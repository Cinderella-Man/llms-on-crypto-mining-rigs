#!/usr/bin/env python3
# gguf_retype_test.py DIR [--quantize <llama-quantize> --ggml <libggml-base.so> --old SHARD... --new SHARD...
#     [--sample N]]: a self-test of gguf_retype.py (beside it).
#   Always, two synthetic files:
#   - a three-shard GGUF of BF16, F32 and raw (I8) tensors of odd sizes, with blocks that test the edge cases
#     (rounding ties, all-zero and all -0.0 blocks, the largest value negative, Q8_0 scales below F16's normal range
#     and below its smallest step, F16 ties in the normal and subnormal ranges): the BF16 `hc_` tensors go to Q8_0 and
#     the F32 `ffn_gate_inp` ones to F16;
#   - a two-shard GGUF of BF16, F32, F16 and Q8_0 tensors that go to Q6_K, Q5_0 and Q4_0, with blocks that test those
#     types' edge cases (the first of two equal magnitudes deciding the scale's sign, the cap at 31 and 15, values one
#     float32 step below a truncation boundary, nearest_int ties, Q6_K sub-blocks that are zero, sub-blocks whose int8
#     scale rounds to 0 and blocks whose F16 d underflows, both of which keep make_qx_quants' own L, an outlier that
#     moves the scale search, a tensor of more than one conversion piece), a 320-wide tensor that a first rule keeps
#     in Q8_0, and a shard with nothing to convert that --link-unchanged links;
#   and checks that (1) the converted bytes equal a line-by-line transcription of ggml's C (quantize_row_q8_0_ref,
#   _q5_0_ref, _q4_0_ref, _q6_K_ref with make_qx_quants and nearest_int, dequantize_row_q8_0,
#   ggml_compute_fp32_to_fp16), (2) every other tensor is unchanged byte for byte and a shard with nothing to convert
#   has its old SHA-256, (3) the headers keep their key/value pairs and carry the new types, with aligned offsets,
#   (4) `verify` passes. Negative controls, which must fail: a rule asking for an impossible conversion, a Q6_K rule on
#   rows of 320, and the same transcriptions with one step changed (Q6_K without make_qx_quants' search, Q5_0's
#   truncation done in double instead of float32), which must differ from the tool's bytes on these blocks.
#   With the options: checks the converted tensors of a real retyped file against llama.cpp itself, every tensor or N
#   of each type spread over the file (--sample). Q8_0, Q6_K, Q5_0 and Q4_0 tensors go in their source type into one
#   GGUF per type with the first old shard's key/value pairs (split keys left out), which llama-quantize converts
#   (--pure, and --allow-requantize for a Q8_0 source); every converted tensor also goes through ggml_quantize_chunk,
#   the function llama-quantize calls, from libggml-base through ctypes, from float32 values that ggml's own
#   to_float makes of a Q8_0 or F16 source (llama-quantize leaves `ffn_gate_inp` as it is, so F16 has only this
#   check). Each must equal its tensor in the new shards byte for byte, and ggml's bytes of one tensor must differ
#   from the tool's of another of the same shape (the negative control).
# Prints "ok" or fails an assertion. Uses DIR for its files and removes them. Standard library only.
import argparse, ctypes, hashlib, math, os, random, shutil, struct, subprocess, sys
from array import array

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

ALIGN = 32
F32, F16, Q4_0, Q5_0, Q8_0, Q6_K, I8, BF16 = 0, 1, 2, 6, 8, 14, 24, 30
NAME = {F32: "F32", F16: "F16", Q4_0: "Q4_0", Q5_0: "Q5_0", Q8_0: "Q8_0", Q6_K: "Q6_K", BF16: "BF16"}
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q4_0: (32, 18), Q5_0: (32, 22), Q8_0: (32, 34), Q6_K: (256, 210)}
TOOL = os.path.join(here, "gguf_retype.py")
EPS = array("f", [1e-15])[0]


def f32(v):
    """A double rounded to float32, as C's float arithmetic leaves it (inf beyond the range)."""
    return array("f", [v])[0]


def bits(v):
    return struct.unpack("<I", array("f", [v]).tobytes())[0]


def from_bits(b):
    return array("f", struct.pack("<I", b))[0]


def size(n, typ):
    return n // BLOCK[typ][0] * BLOCK[typ][1]


def ref_f16(f):
    """ggml_compute_fp32_to_fp16 (ggml-impl.h), line by line."""
    base = f32(f32(abs(f) * 2.0 ** 112) * 2.0 ** -110)
    w = bits(f)
    shl1_w = (w + w) & 0xFFFFFFFF
    sign = w & 0x80000000
    bias = shl1_w & 0xFF000000
    if bias < 0x71000000:
        bias = 0x71000000
    base = f32(from_bits((bias >> 1) + 0x07800000) + base)
    b = bits(base)
    nonsign = ((b >> 13) & 0x00007C00) + (b & 0x00000FFF)
    return (sign >> 16) | (0x7E00 if shl1_w > 0xFF000000 else nonsign)


def from_f16(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def ref_q8_0(vals):
    """quantize_row_q8_0_ref (ggml-quants.c), line by line."""
    out = bytearray()
    for i in range(0, len(vals), 32):
        blk = vals[i:i + 32]
        amax = 0.0
        for v in blk:
            amax = amax if amax > abs(v) else abs(v)
        d = f32(amax / 127)
        id_ = f32(1.0 / d) if d else 0.0
        out += struct.pack("<H", ref_f16(d))
        for v in blk:
            x0 = f32(v * id_)
            out += struct.pack("<b", math.floor(x0 + 0.5) if x0 >= 0 else math.ceil(x0 - 0.5))  # roundf
    return bytes(out)


def ref_dequant_q8_0(raw):
    """dequantize_row_q8_0: y = qs[j] * d."""
    vals = []
    for i in range(0, len(raw), 34):
        d = from_f16(struct.unpack("<H", raw[i:i + 2])[0])
        vals += [f32(q * d) for q in struct.unpack("<32b", raw[i + 2:i + 34])]
    return vals


def ref_q_legacy(vals, nmax, double_trunc=False):
    """quantize_row_q5_0_ref (nmax 16) and quantize_row_q4_0_ref (nmax 8), line by line. double_trunc (a negative
    control) truncates x0 + 16.5 in double instead of rounding it to float first."""
    out = bytearray()
    for i in range(0, len(vals), 32):
        blk = vals[i:i + 32]
        amax = mx = 0.0
        for v in blk:
            if amax < abs(v):
                amax, mx = abs(v), v
        d = f32(mx / -nmax)
        id_ = f32(1.0 / d) if d else 0.0
        out += struct.pack("<H", ref_f16(d))
        qh, qs = 0, bytearray(16)
        for j in range(16):
            x0, x1 = f32(blk[j] * id_), f32(blk[16 + j] * id_)
            t0, t1 = (x0 + nmax + 0.5, x1 + nmax + 0.5) if double_trunc else (f32(x0 + (nmax + 0.5)), f32(x1 + (nmax + 0.5)))
            xi0, xi1 = min(2 * nmax - 1, int(t0)), min(2 * nmax - 1, int(t1))  # (int8_t) truncates toward zero
            if nmax == 16:
                qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4)
                qh |= ((xi0 & 0x10) >> 4) << j
                qh |= ((xi1 & 0x10) >> 4) << (j + 16)
            else:
                qs[j] = xi0 | (xi1 << 4)
        if nmax == 16:
            out += struct.pack("<I", qh)
        out += qs
    return bytes(out)


def ref_nearest(fval):
    """nearest_int (ggml-quants.c): the float sum with 1.5 * 2^23 rounds to an integer."""
    return (bits(f32(fval + 12582912.0)) & 0x007FFFFF) - 0x00400000


def ref_make_qx_quants(x, nmax, rmse_type):
    """make_qx_quants(n, nmax, x, L, rmse_type, NULL), line by line: (L, scale)."""
    mx = amax = 0.0
    for v in x:
        if abs(v) > amax:
            amax, mx = abs(v), v
    if amax < EPS:
        return [0] * len(x), 0.0
    iscale = f32(-nmax / mx)
    if rmse_type == 0:
        return [nmax + max(-nmax, min(nmax - 1, ref_nearest(f32(iscale * v)))) for v in x], f32(1 / iscale)

    def sums(iscale):
        sumlx = suml2 = 0.0
        for v in x:
            l = max(-nmax, min(nmax - 1, ref_nearest(f32(iscale * v))))
            w = f32(v * v)
            sumlx = f32(sumlx + f32(f32(w * v) * l))
            suml2 = f32(suml2 + f32(f32(w * l) * l))
        return sumlx, suml2

    L = [nmax + max(-nmax, min(nmax - 1, ref_nearest(f32(iscale * v)))) for v in x]
    sumlx, suml2 = sums(iscale)
    scale = f32(sumlx / suml2) if suml2 else 0.0
    best = f32(scale * sumlx)
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = f32(-f32(nmax + f32(f32(0.1) * is_)) / mx)
        sumlx, suml2 = sums(iscale)
        if suml2 > 0 and f32(sumlx * sumlx) > f32(best * suml2):
            L = [nmax + max(-nmax, min(nmax - 1, ref_nearest(f32(iscale * v)))) for v in x]
            scale = f32(sumlx / suml2)
            best = f32(scale * sumlx)
    return L, scale


def ref_q6_k(vals, rmse_type=1):
    """quantize_row_q6_K_ref, line by line; rmse_type 0 (a negative control) skips make_qx_quants' search."""
    out = bytearray()
    for i in range(0, len(vals), 256):
        x = vals[i:i + 256]
        L, scales = [0] * 256, [0.0] * 16
        max_scale = max_abs = 0.0
        for ib in range(16):
            L[16 * ib:16 * ib + 16], scales[ib] = ref_make_qx_quants(x[16 * ib:16 * ib + 16], 32, rmse_type)
            if abs(scales[ib]) > max_abs:
                max_abs, max_scale = abs(scales[ib]), scales[ib]
        if max_abs < EPS:
            out += bytes(210)
            continue
        iscale = f32(-128.0 / max_scale)
        d16 = ref_f16(f32(1 / iscale))
        sc8 = [min(127, ref_nearest(f32(iscale * s))) for s in scales]
        for j in range(16):
            d = f32(from_f16(d16) * sc8[j])
            if not d:
                continue
            for ii in range(16):
                L[16 * j + ii] = max(-32, min(31, ref_nearest(f32(x[16 * j + ii] / d)))) + 32
        ql, qh = bytearray(128), bytearray(64)
        for half in (0, 1):
            j = 128 * half
            for l in range(32):
                q1, q2, q3, q4 = (L[j + l + k] & 0xF for k in (0, 32, 64, 96))
                ql[64 * half + l] = q1 | (q3 << 4)
                ql[64 * half + l + 32] = q2 | (q4 << 4)
                qh[32 * half + l] = ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4)
                                     | ((L[j + l + 96] >> 4) << 6))
        out += ql + qh + bytes(v & 255 for v in sc8) + struct.pack("<H", d16)
    return bytes(out)


def ref_convert(raw, frm, to, **kw):
    vals = {F32: lambda: list(struct.unpack(f"<{len(raw) // 4}f", raw)), BF16: lambda: bf16_values(raw),
            F16: lambda: [from_f16(h) for h in struct.unpack(f"<{len(raw) // 2}H", raw)],
            Q8_0: lambda: ref_dequant_q8_0(raw)}[frm]()
    if to == Q8_0:
        return ref_q8_0(vals)
    if to in (Q5_0, Q4_0):
        return ref_q_legacy(vals, 16 if to == Q5_0 else 8, **kw)
    if to == Q6_K:
        return ref_q6_k(vals, **kw)
    return struct.pack(f"<{len(vals)}H", *map(ref_f16, vals))


def bf16(vals):
    """float32 values cut to BF16 (exact for the special values below), as raw bytes."""
    raw = b"".join(struct.pack("<f", v)[2:] for v in vals)
    return raw, [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def write_gguf(path, kv, tensors, pad_last=False):  # tensors: (name, dims, type, bytes)
    with open(path, "wb") as f:
        buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for key, t, raw in kv:
            buf += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", t) + raw
        off = 0
        for name, dims, typ, data in tensors:
            buf += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
            buf += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
            off += len(data) + (-len(data)) % ALIGN
        buf += b"\0" * ((-len(buf)) % ALIGN)
        f.write(buf)
        for i, (_n, _d, _t, data) in enumerate(tensors):
            f.write(data + (b"\0" * ((-len(data)) % ALIGN) if pad_last or i < len(tensors) - 1 else b""))


def split_kv(no, count, total):
    return [("split.no", 2, struct.pack("<H", no)), ("split.count", 2, struct.pack("<H", count)),
            ("split.tensors.count", 5, struct.pack("<i", total))]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_retyped(olds, news, want):
    """Every tensor of the new shards has the type and the leading bytes `want` gives it; headers as the old ones."""
    seen = 0
    for p_old, p_new in zip(olds, news):
        ho, hn = g.Header(p_old), g.Header(p_new)
        assert ho.kv == hn.kv and [t[:2] for t in ho.tensors] == [t[:2] for t in hn.tensors]
        assert hn.data0 % ALIGN == 0
        raw_new = open(p_new, "rb").read()
        for name, _dims, typ, off in hn.tensors:
            wt, data = want[name]
            assert off % ALIGN == 0 and typ == wt, f"{name}: type {typ}, want {wt}"
            assert raw_new[hn.data0 + off:hn.data0 + off + len(data)] == data, f"{name}: bytes differ"
            seen += 1
    assert seen == len(want)


def synthetic(d):
    old, new = os.path.join(d, "old"), os.path.join(d, "new")
    os.makedirs(old, exist_ok=True)
    rnd = random.Random(218)
    # BF16 blocks of 32: zeros, -0.0, ties at d = 1 (amax 127), the largest value negative, scales in F16's subnormal
    # range and below its smallest step, a large scale; then Gaussian blocks of mixed scales
    special = [[0.0] * 32, [-0.0] * 32,
               [127.0] + [s * (k + 0.5) for k in range(0, 124, 8) for s in (1, -1)][:31],
               [rnd.uniform(-1, 1) for _ in range(31)] + [-3.0],
               [rnd.gauss(0, 3e-5) for _ in range(32)], [rnd.gauss(0, 1e-9) for _ in range(32)],
               [rnd.gauss(0, 2e3) for _ in range(32)]]
    hc_up = [v for b in special for v in b] + [rnd.gauss(0, 10 ** rnd.uniform(-4, 1)) for _ in range(64 * 6 - 32 * 7)]
    hc_down = [rnd.gauss(0, 0.02) for _ in range(96 * 3)]
    hc_out = [rnd.gauss(0, 0.5) for _ in range(32 * 5)]
    # F32 router: F16 ties (1 + 2^-11 rounds down to even, 1 + 3 * 2^-11 up), subnormal ties and the largest values
    ties = [1 + 2 ** -11, 1 + 3 * 2 ** -11, -(1 + 2 ** -11), 3 * 2 ** -25, 5 * 2 ** -25, 2 ** -26, -3 * 2 ** -25,
            65504.0, 65519.0, -65504.0, 6.1e-5, 0.0, -0.0]
    gate = ties + [rnd.gauss(0, 0.05) for _ in range(32 * 4 - len(ties))]
    shexp = [rnd.gauss(0, 0.05) for _ in range(32)]
    f32b = lambda v: struct.pack(f"<{len(v)}f", *v)
    raw = lambda n: rnd.randbytes(n)
    t1 = [("blk.0.attn_norm.weight", [61], F32, f32b([rnd.gauss(1, 0.1) for _ in range(61)])),
          ("blk.0.hc_attn_up.weight", [64, 6], BF16, bf16(hc_up)[0]),
          ("blk.0.experts.weight", [4999], I8, raw(4999)),
          ("blk.0.hc_attn_norm.weight", [61], F32, f32b([rnd.gauss(1, 0.1) for _ in range(61)])),
          ("blk.0.ffn_gate_inp.weight", [32, 4], F32, f32b(gate)),
          ("blk.0.ffn_gate_inp_shexp.weight", [32], F32, f32b(shexp))]
    t2 = [("blk.1.hc_ffn_down.weight", [96, 3], BF16, bf16(hc_down)[0]),
          ("blk.1.experts.weight", [3001], I8, raw(3001)),
          ("output_hc_up.weight", [32, 5], BF16, bf16(hc_out)[0]),
          ("blk.1.tail.weight", [777], I8, raw(777))]
    t3 = [("per_layer_token_embd.weight", [70001], I8, raw(70001))]
    total = len(t1) + len(t2) + len(t3)
    olds = [os.path.join(old, f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    write_gguf(olds[0], [("general.architecture", 8, struct.pack("<Q", 4) + b"test")] + split_kv(0, 3, total), t1)
    write_gguf(olds[1], split_kv(1, 3, total), t2)
    write_gguf(olds[2], split_kv(2, 3, total), t3, pad_last=True)

    rules = ["--rule", "hc_=Q8_0@BF16", "--rule", "ffn_gate_inp=F16@F32"]
    subprocess.run([sys.executable, "-I", TOOL, "retype", *rules, "--out-dir", new, "--prefix", "r", "--jobs", "2",
                    *olds], check=True, stdout=subprocess.DEVNULL)
    news = [os.path.join(new, f"r-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    assert sha(news[2]) == sha(olds[2]), "the shard with nothing to convert changed"

    want = {}
    for name, dims, typ, data in t1 + t2 + t3:
        if "hc_" in name and typ == BF16:
            want[name] = (Q8_0, ref_q8_0(bf16_values(data)))
        elif "ffn_gate_inp" in name:
            vals = struct.unpack(f"<{len(data) // 4}f", data)
            want[name] = (F16, struct.pack(f"<{len(vals)}H", *map(ref_f16, vals)))
        else:
            want[name] = (typ, data)
    assert len([n for n in want if want[n][0] in (Q8_0, F16)]) == 5
    check_retyped(olds, news, want)
    out = subprocess.run([sys.executable, "-I", TOOL, "verify", *olds, "--new", *news], check=True,
                         capture_output=True, text=True).stdout
    assert out.count(" converted") == 3 and "3 tensors identical" in out and "2 converted" in out, out
    bad = subprocess.run([sys.executable, "-I", TOOL, "retype", "--rule", "experts=Q8_0", "--out-dir",
                          os.path.join(d, "bad"), *olds], capture_output=True, text=True)
    assert bad.returncode != 0 and "no conversion from type 24" in bad.stderr, bad.stderr


def q8_0_bytes(rnd, nb):
    """Random Q8_0 blocks: F16 scales of mixed magnitude (one zero, one negative), int8 values including -128."""
    out = bytearray()
    for i in range(nb):
        d = 0.0 if i == 1 else (-1 if i == 2 else 1) * 10 ** rnd.uniform(-5, -1)
        q = [-128, 127, 0] + [rnd.randint(-127, 127) for _ in range(29)] if i == 0 else \
            [rnd.randint(-127, 127) for _ in range(32)]
        out += struct.pack("<e", d) + struct.pack("<32b", *q)
    return bytes(out)


def synthetic_low(d):
    """Q6_K, Q5_0 and Q4_0 from BF16, F32, F16 and Q8_0, against the transcriptions; the negative controls."""
    old, new = os.path.join(d, "old-low"), os.path.join(d, "new-low")
    os.makedirs(old, exist_ok=True)
    rnd = random.Random(620)
    gauss = lambda n, s=1.0: [rnd.gauss(0, s) for _ in range(n)]
    # blocks of 32 for Q5_0 / Q4_0 (F32, so float32 steps hold): zeros, -0.0, equal magnitudes of either sign first,
    # the cap (-16 then 16: d = 1, 16 -> 32.5 -> 31), values one float32 step below a truncation boundary at d = 1,
    # a scale whose F16 underflows, large and small Gaussians
    near = [0.5 - 2 ** -25, -0.5 - 2 ** -24, 1.5 - 2 ** -24, 14.5 - 2 ** -21, 2.5, -3.5]
    legacy = [[0.0] * 32, [-0.0] * 32, [3.0] + gauss(30) + [-3.0], [-3.0] + gauss(30) + [3.0],
              [-16.0, 16.0] + [rnd.uniform(-16, 16) for _ in range(30)],
              [-16.0] + near + [k + 0.5 for k in range(-16, 9)],
              [-8.0] + [k + 0.5 for k in range(-8, 8)] + [rnd.uniform(-8, 8) for _ in range(15)],
              gauss(32, 1e-9), gauss(32, 2e3), gauss(32, 0.02)]
    legacy_vals = [f32(v) for b in legacy for v in b] + [f32(v) for v in gauss(32 * 6, 0.05)]  # 16 blocks
    # blocks of 256 for Q6_K: all zero; all -0.0; one zero sub-block; one sub-block whose int8 scale rounds to 0; all
    # values so small that d underflows F16 (every sub-block keeps make_qx_quants' L); a sub-block below EPS among
    # tiny ones; an outlier; nearest_int ties (max -32: iscale 1, values k + 0.5); equal magnitudes of either sign
    sub = lambda s: gauss(16, s)
    blocks = [[0.0] * 256, [-0.0] * 256,
              sum((sub(1.0) if k != 5 else [0.0] * 16 for k in range(16)), []),
              sum((sub(1.0) if k != 9 else sub(1e-4) for k in range(16)), []),
              sum((sub(1e-9) for _ in range(16)), []),
              sum((sub(1e-9) if k != 3 else [1e-16] * 16 for k in range(16)), []),
              gauss(100, 0.02) + [3.0] + gauss(155, 0.02),
              [-32.0] + [k + 0.5 for k in range(-16, 15)] + gauss(224),
              [2.0] + gauss(14, 0.5) + [-2.0] + [-2.0] + gauss(14, 0.5) + [2.0] + gauss(224)]
    q6_vals = [f32(v) for b in blocks for v in b]  # 9 blocks
    f32b = lambda v: struct.pack(f"<{len(v)}f", *v)
    hc_down = bf16(gauss(256 * 6, 0.02) + [5.0] + gauss(255, 0.02))[0]  # 7 blocks of 256
    hc_up = bf16(gauss(320 * 4, 0.02))[0]
    big = bf16(gauss(1024 * 1100, 0.03))[0]  # 1,126,400 values: two conversion pieces
    f16s = struct.pack(f"<{64 * 3}e", *gauss(64 * 3, 0.1))
    t1 = [("blk.0.hc_attn_down.weight", [256, 7], BF16, hc_down),
          ("blk.0.hc_attn_up.weight", [320, 4], BF16, hc_up),
          ("blk.0.ssm_out.weight", [256, 9], F32, f32b(q6_vals)),
          ("blk.0.attn_norm.weight", [61], F32, f32b(gauss(61))),
          ("blk.0.ffn_gate_shexp.weight", [32, 16], F32, f32b(legacy_vals)),
          ("blk.0.ffn_up_shexp.weight", [64, 3], F16, f16s),
          ("blk.0.attn_q.weight", [256, 3], Q8_0, q8_0_bytes(rnd, 24)),
          ("blk.0.attn_k.weight", [64, 5], Q8_0, q8_0_bytes(rnd, 10)),
          ("blk.0.attn_v.weight", [512, 2], Q8_0, q8_0_bytes(rnd, 32)),
          ("blk.0.experts.weight", [4999], I8, rnd.randbytes(4999))]
    t2 = [("blk.1.ffn_down_shexp.weight", [1024, 1100], BF16, big),
          ("blk.1.tail.weight", [777], I8, rnd.randbytes(777))]
    t3 = [("per_layer_token_embd.weight", [7001], I8, rnd.randbytes(7001))]
    total = len(t1) + len(t2) + len(t3)
    olds = [os.path.join(old, f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    write_gguf(olds[0], [("general.architecture", 8, struct.pack("<Q", 4) + b"test")] + split_kv(0, 3, total), t1)
    write_gguf(olds[1], split_kv(1, 3, total), t2)
    write_gguf(olds[2], split_kv(2, 3, total), t3, pad_last=True)
    rules = {"hc_attn_up": Q8_0, "hc_": Q6_K, "ssm_out": Q6_K, "ffn_gate_shexp": Q5_0, "ffn_up_shexp": Q4_0,
             "attn_q": Q5_0, "attn_k": Q4_0, "attn_v": Q6_K, "ffn_down_shexp": Q5_0}
    args = sum((["--rule", f"{k}={NAME[v]}"] for k, v in rules.items()), [])
    subprocess.run([sys.executable, "-I", TOOL, "retype", *args, "--out-dir", new, "--jobs", "3",
                    "--link-unchanged", *olds], check=True, stdout=subprocess.DEVNULL)
    news = [os.path.join(new, os.path.basename(p)) for p in olds]
    assert os.path.islink(news[2]) and os.path.samefile(news[2], olds[2]), "--link-unchanged did not link shard 3"

    want = {}
    for name, dims, typ, data in t1 + t2 + t3:
        to = next((v for k, v in rules.items() if k in name), None)
        want[name] = (to, ref_convert(data, typ, to)) if to is not None else (typ, data)
    assert sum(t == Q6_K for t, _d in want.values()) == 3
    check_retyped(olds, news, want)
    out = subprocess.run([sys.executable, "-I", TOOL, "verify", *olds, "--new", *news], check=True,
                         capture_output=True, text=True).stdout
    assert "2 tensors identical" in out and "8 converted" in out and "1 converted" in out, out

    # negative controls: each must fail, or differ from the tool's bytes
    bad = subprocess.run([sys.executable, "-I", TOOL, "retype", "--rule", "hc_=Q6_K@BF16", "--out-dir",
                          os.path.join(d, "bad-low"), *olds], capture_output=True, text=True)
    assert bad.returncode != 0 and "not whole Q6_K blocks" in bad.stderr, bad.stderr
    got = dict((n, data) for n, (_t, data) in want.items())
    assert ref_convert(f32b(q6_vals), F32, Q6_K, rmse_type=0) != got["blk.0.ssm_out.weight"], \
        "Q6_K without the scale search gives the same bytes: the blocks do not test it"
    assert ref_convert(f32b(legacy_vals), F32, Q5_0, double_trunc=True) != got["blk.0.ffn_gate_shexp.weight"], \
        "Q5_0 truncated in double gives the same bytes: the blocks do not test float32's rounding"
    return 8


def bf16_values(raw):
    return [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def rd(path, off, n):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, n, off)
    finally:
        os.close(fd)


class Traits(ctypes.Structure):  # struct ggml_type_traits (ggml.h)
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


def real(d, a):
    """The converted tensors of a real retyped file against llama-quantize and ggml_quantize_chunk."""
    os.makedirs(d, exist_ok=True)
    conv = []  # (name, dims, old type, new type, old file, old data offset, new file, new data offset)
    for p_old, p_new in zip(a.old, a.new):
        ho, hn = g.Header(p_old), g.Header(p_new)
        for (name, dims, to_, oo), (_n, _d, tn, on) in zip(ho.tensors, hn.tensors):
            if to_ != tn:
                conv.append((name, dims, to_, tn, p_old, ho.data0 + oo, p_new, hn.data0 + on))
    assert conv, "nothing was converted"
    lib = ctypes.CDLL(a.ggml)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    to_float_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)

    def floats(typ, src, n):  # float32 bytes of a source tensor, by ggml's to_float (BF16: the exact widening)
        if typ == F32:
            return src
        if typ == BF16:
            buf = bytearray(2 * len(src))
            buf[2::4], buf[3::4] = src[0::2], src[1::2]
            return bytes(buf)
        x, y = ctypes.create_string_buffer(src, len(src)), ctypes.create_string_buffer(4 * n)
        to_float_t(lib.ggml_get_type_traits(typ).contents.to_float)(x, y, n)
        return y.raw

    for to in sorted({c[3] for c in conv}):
        group = [c for c in conv if c[3] == to]
        if a.sample and len(group) > a.sample:
            group = [group[i * (len(group) - 1) // (a.sample - 1)] for i in range(a.sample)] if a.sample > 1 else group[:1]
        made = {}
        for name, dims, frm, _tn, p_old, po, p_new, pn in group:
            n = math.prod(dims)
            x = ctypes.create_string_buffer(floats(frm, rd(p_old, po, size(n, frm)), n), 4 * n)
            y = ctypes.create_string_buffer(size(n, to))
            done = lib.ggml_quantize_chunk(to, x, y, 0, n // dims[0], dims[0], None)
            assert done == size(n, to), f"{name}: ggml_quantize_chunk wrote {done} bytes"
            made[name] = y.raw
            assert y.raw == rd(p_new, pn, size(n, to)), f"{name}: differs from ggml_quantize_chunk's {NAME[to]}"
        others = [(c, e) for c in group for e in group if c[0] != e[0] and c[1] == e[1]]
        if others:  # the negative control: ggml's bytes of another tensor of this shape differ from the tool's
            c, e = others[0]
            assert made[e[0]] != rd(c[6], c[7], size(math.prod(c[1]), to)), f"{c[0]} and {e[0]} have the same bytes"
        print(f"{NAME[to]}: {len(group)} tensors identical to ggml_quantize_chunk's"
              + (f"; {e[0]}'s differ from {c[0]}'s" if others else ""))
        if to == F16:  # llama-quantize never converts ffn_gate_inp
            continue
        probe, probe_q = os.path.join(d, f"probe-{NAME[to]}.gguf"), os.path.join(d, f"probe-{NAME[to]}-q.gguf")
        kv = [x for x in g.Header(a.old[0]).kv if not x[0].startswith("split.")]
        with open(probe, "wb") as f:  # the header, then each tensor's source bytes
            buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(group), len(kv)))
            for key, t, raw in kv:
                buf += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", t) + raw
            off = 0
            for name, dims, frm, *_x in group:
                k = size(math.prod(dims), frm)
                buf += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
                buf += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", frm, off)
                off += k + (-k) % ALIGN
            f.write(buf + b"\0" * ((-len(buf)) % ALIGN))
            for name, dims, frm, _tn, p_old, po, *_x in group:
                k = size(math.prod(dims), frm)
                f.write(rd(p_old, po, k) + b"\0" * ((-k) % ALIGN))
        log = os.path.join(d, f"llama-quantize-{NAME[to]}.log")
        requant = ["--allow-requantize"] if any(c[2] == Q8_0 for c in group) else []
        with open(log, "w") as lf:
            r = subprocess.run([a.quantize, *requant, "--pure", probe, probe_q, NAME[to], str(a.threads)], stdout=lf,
                               stderr=lf)
        assert r.returncode == 0, f"llama-quantize failed: {open(log).read()[-2000:]}"
        hq = g.Header(probe_q)
        got = {name: (typ, hq.data0 + off) for name, _dims, typ, off in hq.tensors}
        for name, dims, _frm, _tn, _po, _oo, p_new, pn in group:
            typ, off = got[name]
            k = size(math.prod(dims), to)
            assert typ == to, f"llama-quantize wrote {name} as type {typ}"
            assert rd(probe_q, off, k) == rd(p_new, pn, k), f"{name}: differs from llama-quantize's {NAME[to]}"
        print(f"{NAME[to]}: {len(group)} tensors identical to llama-quantize's")
        os.remove(probe)
        os.remove(probe_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--quantize", help="llama-quantize")
    ap.add_argument("--ggml", help="libggml-base.so of the same build")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, help="check this many tensors of each new type, spread over the file")
    ap.add_argument("--old", nargs="+")
    ap.add_argument("--new", nargs="+")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    synthetic(os.path.join(d, "synthetic"))
    synthetic_low(os.path.join(d, "synthetic"))
    if a.old:
        assert a.quantize and a.ggml and a.new and len(a.old) == len(a.new), "--quantize, --ggml, --old and --new"
        real(os.path.join(d, "real"), a)
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
