#!/usr/bin/env python3
# gguf_retype.py: converts the tensors of a (split) GGUF that match a rule to another type and copies every other
# tensor byte for byte; each tensor stays in its shard. Made for Qwen3.8-Flash-Next's GSQ-RCO files, whose
# hyper-connection projections (`hc_*`) are BF16, which Pascal has no fast path for, and whose router (`ffn_gate_inp`)
# is F32, which llama-quantize never converts: the file's own BF16 and F32 tensors are the source, so no BF16
# download is needed.
#   gguf_retype.py retype --rule 'hc_=Q8_0@BF16' --rule 'ffn_gate_inp=F16@F32' --out-dir <dir> [--prefix <name>] \
#       [--jobs N] [--link-unchanged] SHARD...
#     A rule is <regex>=<type>[@<from type>]: tensors whose name matches the regex (re.search) and, with @, whose type
#     is <from type> are converted; the first matching rule wins. Each conversion is bit for bit what ggml's reference
#     quantizer writes, which is what llama-quantize writes when it has no importance matrix:
#       - to Q8_0 from F32, F16 or BF16 (quantize_row_q8_0_ref): per block of 32, d = amax / 127 and id = 1 / d in
#         float32, q = roundf(x * id) (half away from zero), d stored in F16 (round to nearest even);
#       - to Q5_0 or Q4_0 from F32, F16, BF16 or Q8_0 (quantize_row_q5_0_ref, quantize_row_q4_0_ref): per block of 32,
#         max is the first value of the largest magnitude, d = max / -16 (Q4_0: -8) and id = 1 / d in float32,
#         q = min(31, trunc(x * id + 16.5)) (Q4_0: min(15, trunc(x * id + 8.5))), d stored in F16;
#       - to Q6_K from the same (quantize_row_q6_K_ref): per block of 256, each sub-block of 16 takes the scale that
#         make_qx_quants finds among 19 trial scales (squared error weighted by x²), the 16 scales are stored as int8
#         against the largest, d in F16, and q = nearest(x / (d * scale)) clamped to [-32, 31];
#       - to F16 from F32 or BF16, round to nearest even as ggml_fp32_to_fp16_row; a value beyond F16's range stops
#         the tool (ggml would write inf);
#       - to BF16 from F32, round to nearest even as ggml_compute_fp32_to_bf16 (a NaN stays a quiet NaN); exact for an
#         F32 tensor that holds BF16 values, as the GSQ-RCO routers do (llama-quantize never converts a router).
#     A Q8_0 source is read as dequantize_row_q8_0 reads it (q * d, exact in float32), so Q8_0 -> Q5_0 is what
#     llama-quantize --allow-requantize writes. Rows must hold whole blocks (32 values, Q6_K 256): llama-quantize would
#     switch such a tensor to another type, so give it a rule of its own first (e.g. 'hc_.*_up=Q8_0@BF16').
#     The shards keep their names in --out-dir, or take --prefix in place of the part before -0000k-of-0000N.gguf.
#     The key/value pairs and the order of the tensors are unchanged; offsets are laid out again in the data order,
#     each tensor keeping its slot (its bytes plus the padding up to the next tensor) or, converted, padded to the
#     alignment. A shard with nothing to convert comes out byte for byte as it was, or with --link-unchanged as a
#     symbolic link to the old shard's file (Flash-Next's n-gram table shard: 28.8 GB not copied).
#   gguf_retype.py verify OLD_SHARD... --new NEW_SHARD...
#     checks each new shard against its old one: the same key/value pairs, tensor names, shapes and order, every
#     tensor of unchanged type identical byte for byte, and every converted tensor of the size its type needs.
# Python floats are doubles; each float32 step is done in double and rounded to float32 on its way into an
# array('f'), which gives float32's own result (a product of two floats is exact in double, and a sum or quotient
# rounded twice, to double and then to float, equals the one rounded once). The C code's loops over a block run here
# across all blocks at once, one column of the block at a time, in the C code's order. Tensors are converted in
# pieces of 2^20 values (whole blocks) across the worker processes. Standard library only.
import argparse, math, multiprocessing, os, re, struct, sys
from array import array
from itertools import chain, compress, repeat
from operator import add, and_, gt, mul, not_, or_, truediv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reshard import CHUNK, Header, copy_range, write_all  # noqa: E402

F32, F16, Q4_0, Q5_0, Q8_0, Q6_K, BF16 = 0, 1, 2, 6, 8, 14, 30
TYPES = {"F32": F32, "F16": F16, "Q4_0": Q4_0, "Q5_0": Q5_0, "Q8_0": Q8_0, "Q6_K": Q6_K, "BF16": BF16}
NAMES = {v: k for k, v in TYPES.items()}
FLOATS = (F32, F16, BF16)
SOURCES = {Q8_0: FLOATS, Q5_0: FLOATS + (Q8_0,), Q4_0: FLOATS + (Q8_0,), Q6_K: FLOATS + (Q8_0,), F16: (F32, BF16),
           BF16: (F32,)}
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q4_0: (32, 18), Q5_0: (32, 22), Q8_0: (32, 34), Q6_K: (256, 210)}
PIECE = 1 << 20  # values per conversion job: whole blocks of every type
EPS = array("f", [1e-15])[0]  # GROUP_MAX_EPS
LO = bytes(v & 15 for v in range(256))
LO4 = bytes((v & 15) << 4 for v in range(256))
HI = [bytes((v >> 4 & 3) << k for v in range(256)) for k in (0, 2, 4, 6)]
NEAREST = {v: max(-32, min(31, v)) for v in range(-64, 65)}  # Q6_K's clamp of nearest_int; |iscale * x| < 33


def nbytes(n, typ):
    k, size = BLOCK[typ]
    return n // k * size


def f32(v):
    return array("f", [v])[0]


def floats(raw, typ):
    """The values of `raw` (a tensor of type F32, F16, BF16 or Q8_0) as an array of float32."""
    if typ == F32:
        return array("f", raw)
    if typ == BF16:  # the high half of a float32
        buf = bytearray(2 * len(raw))
        buf[2::4], buf[3::4] = raw[0::2], raw[1::2]
        return array("f", bytes(buf))
    if typ == Q8_0:  # dequantize_row_q8_0: y = qs[j] * d, exact in float32
        nb = len(raw) // 34
        dh, qs = bytearray(2 * nb), bytearray(32 * nb)
        dh[0::2], dh[1::2] = raw[0::34], raw[1::34]
        for j in range(32):
            qs[j::32] = raw[2 + j::34]
        d = struct.unpack(f"<{nb}e", dh)
        return array("f", map(mul, array("b", bytes(qs)), chain.from_iterable(map(repeat, d, repeat(32, nb)))))
    return array("f", struct.unpack(f"<{len(raw) // 2}e", raw))


def bor(*parts):
    """The bytewise OR of byte strings of one length."""
    v = 0
    for p in parts:
        v |= int.from_bytes(p, "little")
    return v.to_bytes(len(parts[0]), "little")


def first_max(cols):
    """Per position of the columns: the first value of the largest magnitude, and that magnitude (0.0 and 0.0 when
    all are zero), as ggml's `if (amax < fabsf(v)) { amax = fabsf(v); max = v; }` from amax = max = 0."""
    mx = amax = [0.0] * len(cols[0])
    for c in cols:
        a = list(map(abs, c))
        mx = [v if b > m else old for v, b, m, old in zip(c, a, amax, mx)]
        amax = list(map(max, amax, a))
    return mx, amax


def q8_0(x):
    """ggml's quantize_row_q8_0_ref on float32 values x (len a multiple of 32): blocks of an F16 scale and 32 int8."""
    nb = len(x) // 32
    amax = [abs(max(max(b), -min(b))) for b in (x[i:i + 32] for i in range(0, len(x), 32))]  # abs: -0.0 -> 0.0
    d = array("f", [a / 127.0 for a in amax])
    inv = array("f", [1.0 / v if v else 0.0 for v in d])
    x0 = array("f", map(mul, x, chain.from_iterable(map(repeat, inv, repeat(32, nb)))))
    # roundf: floor(|x| + 0.5) with x's sign; |x| + 0.5 is exact in double for every float32 x
    q = array("b", map(int, map(math.copysign, map(math.floor, map((0.5).__add__, map(abs, x0))), x0))).tobytes()
    dh = struct.pack(f"<{nb}e", *d)
    out = bytearray(34 * nb)
    out[0::34], out[1::34] = dh[0::2], dh[1::2]
    for j in range(32):
        out[2 + j::34] = q[j::32]
    return out


def q_legacy(x, nmax):
    """quantize_row_q5_0_ref (nmax 16: blocks of an F16 scale, 32 fifth bits, 16 bytes of nibbles) or
    quantize_row_q4_0_ref (nmax 8: an F16 scale and 16 bytes of nibbles) on float32 values x."""
    nb, size = len(x) // 32, 22 if nmax == 16 else 18
    cols = [x[j::32] for j in range(32)]  # column j: value j of every block
    mx, _ = first_max(cols)
    d = array("f", [m / -nmax for m in mx])
    inv = array("f", [1.0 / v if v else 0.0 for v in d])
    cap = bytes(min(2 * nmax - 1, v) for v in range(256))  # MIN(31, (int8_t)(x0 + 16.5f)); x0 + 16.5 >= 0.49
    xi = [bytes(map(cap.__getitem__, map(int, array("f", map(add, array("f", map(mul, c, inv)), repeat(nmax + 0.5))))))
          for c in cols]
    dh = struct.pack(f"<{nb}e", *d)
    out = bytearray(size * nb)
    out[0::size], out[1::size] = dh[0::2], dh[1::2]
    if nmax == 16:  # qh: bit j is value j's fifth bit, bit j + 16 value j + 16's
        qh = [0] * nb
        for j, b in enumerate(xi):
            qh = list(map(or_, qh, map(tuple((v >> 4 & 1) << j for v in range(256)).__getitem__, b)))
        qhb = struct.pack(f"<{nb}I", *qh)
        for k in range(4):
            out[2 + k::22] = qhb[k::4]
    first = 6 if nmax == 16 else 2
    for j in range(16):
        out[first + j::size] = bor(xi[j].translate(LO), xi[j + 16].translate(LO4))
    return out


def q6_k(x):
    """ggml's quantize_row_q6_K_ref on float32 values x (len a multiple of 256): blocks of 128 bytes of low nibbles,
    64 of high bit pairs, 16 int8 scales and an F16 d."""
    n = len(x)
    ns, nb = n // 16, n // 256
    cols = [x[i::16] for i in range(16)]  # column i: value i of every sub-block of 16
    # make_qx_quants(16, 32, x, L, 1, NULL) on every sub-block: the weights are x * x
    mx, amax = first_max(cols)
    zero = [a < EPS for a in amax]  # all zero: L = 0 and scale 0
    mx = [1.0 if z else m for m, z in zip(mx, zero)]
    w = [array("f", map(mul, c, c)) for c in cols]
    wx = [array("f", map(mul, wi, c)) for wi, c in zip(w, cols)]

    def trial(iscale):  # sumlx and suml2 of every sub-block for its trial iscale
        sumlx = suml2 = array("f", bytes(4 * ns))
        for c, wi, wxi in zip(cols, w, wx):
            q = list(map(NEAREST.__getitem__, map(round, array("f", map(mul, iscale, c)))))
            sumlx = array("f", map(add, sumlx, array("f", map(mul, wxi, q))))  # sumlx += w*x[i]*l
            suml2 = array("f", map(add, suml2, array("f", map(mul, array("f", map(mul, wi, q)), q))))  # += w*l*l
        return sumlx, suml2

    iscales = {0: array("f", [-32.0 / m for m in mx])}
    sumlx, suml2 = trial(iscales[0])
    scale = array("f", [a / b if b else 0.0 for a, b in zip(sumlx, suml2)])
    best = array("f", map(mul, scale, sumlx))
    win = [0] * ns  # the trial whose L make_qx_quants keeps: 0 the first, else is
    tenth = f32(0.1)
    for step in (*range(-9, 0), *range(1, 10)):
        c = f32(32 + f32(tenth * step))  # nmax + 0.1f*is
        iscales[step] = iscale = array("f", [-c / m for m in mx])
        sx, s2 = trial(iscale)
        better = map(and_, map(gt, s2, repeat(0.0)), map(gt, array("f", map(mul, sx, sx)), array("f", map(mul, best, s2))))
        for s in compress(range(ns), better):
            scale[s] = sx[s] / s2[s]
            best[s] = scale[s] * sx[s]
            win[s] = step
    for s in compress(range(ns), zero):
        scale[s] = 0.0

    # the block: d from the largest scale, the scales in int8, then every value against its sub-block's d * scale
    dh, sc8 = bytearray(2 * nb), bytearray(16 * nb)
    dsub = array("f", bytes(4 * ns))
    empty = []
    for b in range(nb):
        sc = scale[16 * b:16 * b + 16]
        max_scale = max_abs = 0.0
        for v in sc:
            if abs(v) > max_abs:
                max_abs, max_scale = abs(v), v
        if max_abs < EPS:
            empty.append(b)
            continue
        iscale = f32(-128.0 / max_scale)
        dh[2 * b:2 * b + 2] = struct.pack("<e", f32(1.0 / iscale))
        d = struct.unpack("<e", dh[2 * b:2 * b + 2])[0]
        q = [min(127, round(v)) for v in array("f", map(mul, repeat(iscale, 16), sc))]
        sc8[16 * b:16 * b + 16] = bytes(v & 255 for v in q)
        dsub[16 * b:16 * b + 16] = array("f", map(mul, repeat(d, 16), q))
    safe = array("f", [v if v else 1.0 for v in dsub])
    L = bytearray(n)
    for i, c in enumerate(cols):
        q = map(round, array("f", map(truediv, c, safe)))
        L[i::16] = bytes(map((32).__add__, map(max, repeat(-32), map(min, repeat(31), q))))
    for s in compress(range(ns), map(not_, dsub)):  # d * scale == 0: the sub-block keeps make_qx_quants' L
        if zero[s]:
            L[16 * s:16 * s + 16] = bytes(16)
        else:
            isc = iscales[win[s]][s]
            L[16 * s:16 * s + 16] = bytes(NEAREST[round(f32(isc * v))] + 32 for v in x[16 * s:16 * s + 16])

    out = bytearray(210 * nb)
    Lp = [bytes(L[p::256]) for p in range(256)]  # value p of every block
    for h in (0, 1):
        for k in range(32):
            a = 128 * h + k
            out[64 * h + k::210] = bor(Lp[a].translate(LO), Lp[a + 64].translate(LO4))
            out[64 * h + 32 + k::210] = bor(Lp[a + 32].translate(LO), Lp[a + 96].translate(LO4))
            out[128 + 32 * h + k::210] = bor(*(Lp[a + 32 * g].translate(HI[g]) for g in range(4)))
    for j in range(16):
        out[192 + j::210] = sc8[j::16]
    out[208::210], out[209::210] = dh[0::2], dh[1::2]
    for b in empty:  # memset(&y[i], 0, sizeof(block_q6_K))
        out[210 * b:210 * b + 210] = bytes(210)
    return out


def f16(x):
    try:
        return struct.pack(f"<{len(x)}e", *x)
    except OverflowError:
        raise SystemExit("a value is beyond F16's range")


def bf16(x):
    """ggml_compute_fp32_to_bf16: the high half of each float32, rounded to nearest even; a NaN is kept quiet."""
    return array("H", ((v >> 16) | 64 if v & 0x7FFFFFFF > 0x7F800000 else (v + 0x7FFF + (v >> 16 & 1)) >> 16
                       for v in array("I", x.tobytes()))).tobytes()


QUANT = {Q8_0: q8_0, Q5_0: lambda x: q_legacy(x, 16), Q4_0: lambda x: q_legacy(x, 8), Q6_K: q6_k, F16: f16, BF16: bf16}


def convert(job):
    """One piece of a tensor: (file, byte offset, values, from type, to type) -> the converted bytes."""
    path, start, n, frm, to = job
    with open(path, "rb") as f:
        raw = os.pread(f.fileno(), nbytes(n, frm), start)
    assert len(raw) == nbytes(n, frm), f"{path}: short read at {start}"
    return QUANT[to](floats(raw, frm))


def parse_rule(s):
    m = re.fullmatch(r"(.+)=(\w+)(?:@(\w+))?", s)
    to, frm = (TYPES.get(m[2].upper()), TYPES.get((m[3] or "").upper(), m[3])) if m else (None, None)
    assert to in SOURCES and frm in (None, *NAMES), \
        f"rule {s!r}: <regex>=<{'|'.join(NAMES[t] for t in SOURCES)}>[@<from type>]"
    return re.compile(m[1]), to, frm


def target(rules, name, typ):
    for pat, to, frm in rules:
        if pat.search(name) and frm in (None, typ):
            if typ == to:
                return None
            assert typ in SOURCES[to], f"{name}: no conversion from type {NAMES.get(typ, typ)} to {NAMES[to]}"
            return to
    return None


def out_name(path, prefix):
    base = os.path.basename(path)
    if not prefix:
        return base
    m = re.search(r"-\d{5}-of-\d{5}\.gguf$", base)
    return prefix + (m[0] if m else ".gguf")


def write_header(out, h, infos):
    """h's header with each tensor's (type, offset) from `infos`, in h's tensor order, padded to the alignment."""
    buf = bytearray()
    wstr = lambda s: buf.extend(struct.pack("<Q", len(s)) + s)
    buf += b"GGUF" + struct.pack("<IQQ", h.version, len(h.tensors), len(h.kv))
    for key, t, raw in h.kv:
        wstr(key.encode())
        buf += struct.pack("<I", t) + raw
    for name, dims, _typ, _off in h.tensors:
        wstr(name.encode())
        buf += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", *infos[name])
    buf += b"\0" * ((-len(buf)) % h.align)
    write_all(out.fileno(), bytes(buf))


def cmd_retype(a):
    assert sys.byteorder == "little"
    rules = [parse_rule(r) for r in a.rule]
    hs = [Header(p) for p in a.shards]
    count = hs[0].value("split.count")
    assert count in (None, len(hs)), f"{len(hs)} shards given, split.count is {count}"
    os.makedirs(a.out_dir, exist_ok=True)
    total = {}
    with multiprocessing.Pool(a.jobs) as pool:
        for h in hs:
            path = os.path.join(a.out_dir, out_name(h.path, a.prefix))
            assert os.path.islink(path) or not (os.path.exists(path) and os.path.samefile(path, h.path)), \
                f"{path} is its own source"  # a link from --link-unchanged is replaced, not written through
            order, infos, jobs, pieces, off = sorted(h.tensors, key=lambda x: x[3]), {}, [], {}, 0
            for name, dims, typ, old in order:
                to = target(rules, name, typ)
                n = math.prod(dims)
                if to is None:
                    slot = h.slot[name]
                else:
                    assert dims[0] % BLOCK[to][0] == 0, f"{name}: rows of {dims[0]}, not whole {NAMES[to]} blocks"
                    assert dims[0] % BLOCK[typ][0] == 0 and nbytes(n, typ) <= h.slot[name], \
                        f"{name}: its data runs past the next tensor"
                    slot = nbytes(n, to)
                    starts = range(0, n, PIECE)
                    jobs += [(h.path, h.data0 + old + nbytes(k, typ), min(PIECE, n - k), typ, to) for k in starts]
                    pieces[name] = len(starts)
                    key = f"{NAMES[typ]} -> {NAMES[to]}"
                    c = total.setdefault(key, [0, 0, 0])
                    c[0], c[1], c[2] = c[0] + 1, c[1] + nbytes(n, typ), c[2] + slot
                infos[name] = (typ if to is None else to, off)
                off += slot + (-slot) % h.align
            if not jobs and a.link_unchanged:
                if os.path.lexists(path):
                    os.remove(path)
                os.symlink(os.path.realpath(h.path), path)
                print(f"{path} -> {os.path.realpath(h.path)}: nothing to convert")
                continue
            done = pool.imap(convert, jobs)
            with open(h.path, "rb") as src, open(path + ".part", "wb") as dst:
                write_header(dst, h, infos)
                for name, _dims, typ, old in order:
                    if infos[name][0] == typ:
                        copy_range(src, dst, h.data0 + old, h.slot[name], h.align)
                    else:
                        size = 0
                        for _ in range(pieces[name]):
                            data = next(done)
                            write_all(dst.fileno(), bytes(data))
                            size += len(data)
                        write_all(dst.fileno(), b"\0" * ((-size) % h.align))
            os.replace(path + ".part", path)
            print(f"{path}: {len(h.tensors)} tensors, {len(pieces)} converted, {os.path.getsize(path):,} bytes")
            sys.stdout.flush()
    for key, (n, before, after) in sorted(total.items()):
        print(f"{key}: {n} tensors, {before:,} -> {after:,} bytes ({after - before:+,})")


def cmd_verify(a):
    assert len(a.old) == len(a.new), "as many new shards as old ones"
    for old, new in zip(a.old, a.new):
        ho, hn = Header(old), Header(new)
        assert ho.kv == hn.kv, f"{new}: key/value pairs differ"
        assert [t[:2] for t in ho.tensors] == [t[:2] for t in hn.tensors], f"{new}: tensor names, shapes or order"
        same = conv = 0
        with open(old, "rb") as fo, open(new, "rb") as fn:
            for (name, dims, to_, oo), (_n, _d, tn, on) in zip(ho.tensors, hn.tensors):
                if to_ != tn:
                    n = math.prod(dims)
                    assert tn in SOURCES and to_ in SOURCES[tn], f"{name}: {to_} -> {tn}"
                    assert hn.slot[name] >= nbytes(n, tn), f"{name}: {hn.slot[name]} bytes for {NAMES[tn]}"
                    conv += 1
                    continue
                left, po, pn = ho.slot[name], ho.data0 + oo, hn.data0 + on
                while left:
                    k = min(left, CHUNK)
                    assert os.pread(fo.fileno(), k, po) == os.pread(fn.fileno(), k, pn), f"{name}: bytes differ"
                    left, po, pn = left - k, po + k, pn + k
                same += 1
        print(f"{new}: {same} tensors identical to {old}, {conv} converted")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("retype")
    s.add_argument("--rule", action="append", required=True,
                   help="<regex>=<Q8_0|Q6_K|Q5_0|Q4_0|F16|BF16>[@<from type>], repeatable; the first match wins")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--prefix", help="new file name before -0000k-of-0000N.gguf")
    s.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="worker processes (default 8)")
    s.add_argument("--link-unchanged", action="store_true",
                   help="a shard with nothing to convert becomes a symbolic link to the old shard")
    s.add_argument("shards", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("old", nargs="+")
    v.add_argument("--new", nargs="+", required=True)
    a = ap.parse_args()
    cmd_retype(a) if a.cmd == "retype" else cmd_verify(a)


if __name__ == "__main__":
    main()
