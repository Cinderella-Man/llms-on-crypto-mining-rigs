#!/usr/bin/env python3
# gguf_retype.py: converts the tensors of a (split) GGUF that match a rule to another type and copies every other
# tensor byte for byte; each tensor stays in its shard. Made for Qwen3.8-Flash-Next's GSQ-RCO files, whose
# hyper-connection projections (`hc_*`) are BF16, which Pascal has no fast path for, and whose router (`ffn_gate_inp`)
# is F32, which llama-quantize never converts: the file's own BF16 and F32 tensors are the source, so no BF16
# download is needed.
#   gguf_retype.py retype --rule 'hc_=Q8_0@BF16' --rule 'ffn_gate_inp=F16@F32' --out-dir <dir> [--prefix <name>] \
#       [--jobs N] SHARD...
#     A rule is <regex>=<type>[@<from type>]: tensors whose name matches the regex (re.search) and, with @, whose type
#     is <from type> are converted; the first matching rule wins. Conversions:
#       - to Q8_0 from F32, F16 or BF16, bit for bit as ggml's quantize_row_q8_0_ref (what llama-quantize writes):
#         per block of 32, d = amax / 127 and id = 1 / d in float32, q = roundf(x * id) (half away from zero), d
#         stored in F16 (round to nearest even);
#       - to F16 from F32 or BF16, round to nearest even as ggml_fp32_to_fp16_row; a value beyond F16's range stops
#         the tool (ggml would write inf).
#     The shards keep their names in --out-dir, or take --prefix in place of the part before -0000k-of-0000N.gguf.
#     The key/value pairs and the order of the tensors are unchanged; offsets are laid out again in the data order,
#     each tensor keeping its slot (its bytes plus the padding up to the next tensor) or, converted, padded to the
#     alignment. A shard with nothing to convert comes out byte for byte as it was.
#   gguf_retype.py verify OLD_SHARD... --new NEW_SHARD...
#     checks each new shard against its old one: the same key/value pairs, tensor names, shapes and order, every
#     tensor of unchanged type identical byte for byte, and every converted tensor of the size its type needs.
# Python floats are doubles; each float32 step is done in double and rounded to float32 on its way into an
# array('f'), which gives float32's own result (a product of two floats is exact in double, and a quotient rounded
# twice, to double and then to float, equals the quotient rounded once). Standard library only.
import argparse, math, multiprocessing, os, re, struct, sys
from array import array
from itertools import chain, repeat
from operator import mul

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reshard import CHUNK, Header, copy_range, write_all  # noqa: E402

F32, F16, Q8_0, BF16 = 0, 1, 8, 30
TYPES = {"F32": F32, "F16": F16, "Q8_0": Q8_0, "BF16": BF16}
NAMES = {v: k for k, v in TYPES.items()}
SOURCES = {Q8_0: (F32, F16, BF16), F16: (F32, BF16)}
ESIZE = {F32: 4, F16: 2, BF16: 2}  # bytes per element


def nbytes(n, typ):
    return n // 32 * 34 if typ == Q8_0 else n * ESIZE[typ]


def floats(raw, typ):
    """The values of `raw` (a tensor of type F32, F16 or BF16) as float32, through a memoryview of format 'f'."""
    if typ == F32:
        return memoryview(raw).cast("f")
    if typ == BF16:  # the high half of a float32
        buf = bytearray(2 * len(raw))
        buf[2::4], buf[3::4] = raw[0::2], raw[1::2]
        return memoryview(buf).cast("f")
    return memoryview(array("f", struct.unpack(f"<{len(raw) // 2}e", raw)))


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


def f16(x):
    try:
        return struct.pack(f"<{len(x)}e", *x)
    except OverflowError:
        raise SystemExit("a value is beyond F16's range")


def convert(job):
    """One tensor: (file, byte offset, elements, from type, to type) -> the converted bytes."""
    path, start, n, frm, to = job
    with open(path, "rb") as f:
        raw = os.pread(f.fileno(), n * ESIZE[frm], start)
    assert len(raw) == n * ESIZE[frm], f"{path}: short read at {start}"
    x = floats(raw, frm)
    return q8_0(x) if to == Q8_0 else f16(x)


def parse_rule(s):
    m = re.fullmatch(r"(.+)=(\w+)(?:@(\w+))?", s)
    to, frm = (TYPES.get(m[2].upper()), TYPES.get((m[3] or "").upper(), m[3])) if m else (None, None)
    assert to in SOURCES and frm in (None, *SOURCES[Q8_0]), f"rule {s!r}: <regex>=<Q8_0|F16>[@<F32|F16|BF16>]"
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
            assert not (os.path.exists(path) and os.path.samefile(path, h.path)), f"{path} is its own source"
            order, infos, jobs, off = sorted(h.tensors, key=lambda x: x[3]), {}, [], 0
            for name, dims, typ, old in order:
                to = target(rules, name, typ)
                n = math.prod(dims)
                if to is None:
                    slot = h.slot[name]
                else:
                    assert to != Q8_0 or dims[0] % 32 == 0, f"{name}: rows of {dims[0]}, not a multiple of 32"
                    assert nbytes(n, typ) <= h.slot[name], f"{name}: its data runs past the next tensor"
                    slot = nbytes(n, to)
                    jobs.append((h.path, h.data0 + old, n, typ, to))
                    key = f"{NAMES[typ]} -> {NAMES[to]}"
                    c = total.setdefault(key, [0, 0, 0])
                    c[0], c[1], c[2] = c[0] + 1, c[1] + nbytes(n, typ), c[2] + slot
                infos[name] = (typ if to is None else to, off)
                off += slot + (-slot) % h.align
            done = pool.imap(convert, jobs)
            with open(h.path, "rb") as src, open(path + ".part", "wb") as dst:
                write_header(dst, h, infos)
                for name, _dims, typ, old in order:
                    if infos[name][0] == typ:
                        copy_range(src, dst, h.data0 + old, h.slot[name], h.align)
                    else:
                        data = next(done)
                        write_all(dst.fileno(), bytes(data) + b"\0" * ((-len(data)) % h.align))
            os.replace(path + ".part", path)
            print(f"{path}: {len(h.tensors)} tensors, {len(jobs)} converted, {os.path.getsize(path):,} bytes")
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
    s.add_argument("--rule", action="append", required=True, help="<regex>=<Q8_0|F16>[@<from type>], repeatable")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--prefix", help="new file name before -0000k-of-0000N.gguf")
    s.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="worker processes (default 8)")
    s.add_argument("shards", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("old", nargs="+")
    v.add_argument("--new", nargs="+", required=True)
    a = ap.parse_args()
    cmd_retype(a) if a.cmd == "retype" else cmd_verify(a)


if __name__ == "__main__":
    main()
