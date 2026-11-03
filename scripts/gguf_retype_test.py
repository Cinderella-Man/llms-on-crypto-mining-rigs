#!/usr/bin/env python3
# gguf_retype_test.py DIR [--quantize <llama-quantize> --ggml <libggml-base.so> --old SHARD... --new SHARD...]:
# a self-test of gguf_retype.py (beside it).
#   Always: writes a three-shard GGUF of BF16, F32 and raw (I8) tensors of odd sizes, with blocks that test the edge
#   cases (rounding ties, all-zero and all -0.0 blocks, the largest value negative, Q8_0 scales below F16's normal
#   range and below its smallest step, F16 ties in the normal and subnormal ranges), converts the BF16 `hc_` tensors
#   to Q8_0 and the F32 `ffn_gate_inp` ones to F16, and checks that (1) the converted bytes equal a line-by-line
#   transcription of ggml's C (quantize_row_q8_0_ref and ggml_compute_fp32_to_fp16), (2) every other tensor is
#   unchanged byte for byte and the shard with nothing to convert has its old SHA-256, (3) the headers keep their
#   key/value pairs and carry the new types, with aligned offsets, (4) `verify` passes and a rule asking for an
#   impossible conversion fails.
#   With the options: checks the converted tensors of a real retyped file against llama.cpp itself. The Q8_0 ones go
#   in their source type into one GGUF with the first old shard's key/value pairs (split keys left out), which
#   llama-quantize converts to Q8_0; the F16 ones go through ggml_quantize_chunk, the function llama-quantize calls,
#   from libggml-base through ctypes (llama-quantize leaves `ffn_gate_inp` as it is). Each must equal its tensor in
#   the new shards byte for byte.
# Prints "ok" or fails an assertion. Uses DIR for its files and removes them. Standard library only.
import argparse, ctypes, hashlib, math, os, random, shutil, struct, subprocess, sys
from array import array

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

ALIGN = 32
F32, F16, Q8_0, I8, BF16 = 0, 1, 8, 24, 30
TOOL = os.path.join(here, "gguf_retype.py")


def f32(v):
    """A double rounded to float32, as C's float arithmetic leaves it (inf beyond the range)."""
    return array("f", [v])[0]


def bits(v):
    return struct.unpack("<I", array("f", [v]).tobytes())[0]


def from_bits(b):
    return array("f", struct.pack("<I", b))[0]


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
    split = lambda no: [("split.no", 2, struct.pack("<H", no)), ("split.count", 2, struct.pack("<H", 3)),
                        ("split.tensors.count", 5, struct.pack("<i", total))]
    olds = [os.path.join(old, f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    write_gguf(olds[0], [("general.architecture", 8, struct.pack("<Q", 4) + b"test")] + split(0), t1)
    write_gguf(olds[1], split(1), t2)
    write_gguf(olds[2], split(2), t3, pad_last=True)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

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
    seen = 0
    for p_old, p_new in zip(olds, news):
        ho, hn = g.Header(p_old), g.Header(p_new)
        assert ho.kv == hn.kv and [t[:2] for t in ho.tensors] == [t[:2] for t in hn.tensors]
        assert hn.data0 % ALIGN == 0
        raw_new = open(p_new, "rb").read()
        for name, _dims, typ, off in hn.tensors:
            wt, data = want[name]
            assert off % ALIGN == 0 and typ == wt, name
            assert raw_new[hn.data0 + off:hn.data0 + off + len(data)] == data, name
            seen += 1
    assert seen == total
    out = subprocess.run([sys.executable, "-I", TOOL, "verify", *olds, "--new", *news], check=True,
                         capture_output=True, text=True).stdout
    assert out.count(" converted") == 3 and "3 tensors identical" in out and "2 converted" in out, out
    bad = subprocess.run([sys.executable, "-I", TOOL, "retype", "--rule", "experts=Q8_0", "--out-dir",
                          os.path.join(d, "bad"), *olds], capture_output=True, text=True)
    assert bad.returncode != 0 and "no conversion from type 24" in bad.stderr, bad.stderr


def bf16_values(raw):
    return [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]


def rd(path, off, n):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, n, off)
    finally:
        os.close(fd)


def real(d, a):
    """The converted tensors of a real retyped file against llama-quantize (Q8_0) and ggml_quantize_chunk (F16)."""
    os.makedirs(d, exist_ok=True)
    conv = []  # (name, dims, old type, new type, old file, old data offset, new file, new data offset)
    for p_old, p_new in zip(a.old, a.new):
        ho, hn = g.Header(p_old), g.Header(p_new)
        for (name, dims, to_, oo), (_n, _d, tn, on) in zip(ho.tensors, hn.tensors):
            if to_ != tn:
                conv.append((name, dims, to_, tn, p_old, ho.data0 + oo, p_new, hn.data0 + on))
    size = lambda n, t: n // 32 * 34 if t == Q8_0 else n * {F32: 4, F16: 2, BF16: 2}[t]

    q8 = [c for c in conv if c[3] == Q8_0]
    if q8:
        probe, probe_q = os.path.join(d, "probe.gguf"), os.path.join(d, "probe-q8_0.gguf")
        kv = [x for x in g.Header(a.old[0]).kv if not x[0].startswith("split.")]
        with open(probe, "wb") as f:  # the header, then each tensor's source bytes
            buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(q8), len(kv)))
            for key, t, raw in kv:
                buf += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", t) + raw
            off = 0
            for name, dims, to_, *_x in q8:
                n = size(math.prod(dims), to_)
                buf += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
                buf += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", to_, off)
                off += n + (-n) % ALIGN
            f.write(buf + b"\0" * ((-len(buf)) % ALIGN))
            for name, dims, to_, _tn, p_old, po, *_x in q8:
                n = size(math.prod(dims), to_)
                f.write(rd(p_old, po, n) + b"\0" * ((-n) % ALIGN))
        log = os.path.join(d, "llama-quantize.log")
        with open(log, "w") as lf:
            r = subprocess.run([a.quantize, probe, probe_q, "Q8_0", str(a.threads)], stdout=lf, stderr=lf)
        assert r.returncode == 0, f"llama-quantize failed: {open(log).read()[-2000:]}"
        hq = g.Header(probe_q)
        got = {name: (typ, hq.data0 + off) for name, _dims, typ, off in hq.tensors}
        for name, dims, _to, _tn, _po, _oo, p_new, pn in q8:
            typ, off = got[name]
            n = size(math.prod(dims), Q8_0)
            assert typ == Q8_0, f"llama-quantize wrote {name} as type {typ}"
            assert rd(probe_q, off, n) == rd(p_new, pn, n), f"{name}: differs from llama-quantize's Q8_0"
        print(f"Q8_0: {len(q8)} tensors identical to llama-quantize's")

    h16 = [c for c in conv if c[3] == F16]
    if h16:
        lib = ctypes.CDLL(a.ggml)
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        for name, dims, to_, _tn, p_old, po, p_new, pn in h16:
            n = math.prod(dims)
            src = rd(p_old, po, size(n, to_))
            if to_ == BF16:
                buf = bytearray(2 * len(src))
                buf[2::4], buf[3::4] = src[0::2], src[1::2]
                src = bytes(buf)
            assert to_ in (F32, BF16), name
            x = ctypes.create_string_buffer(src, len(src))
            y = ctypes.create_string_buffer(2 * n)
            done = lib.ggml_quantize_chunk(F16, x, y, 0, n // dims[0], dims[0], None)
            assert done == 2 * n and y.raw == rd(p_new, pn, 2 * n), f"{name}: differs from ggml's F16"
        print(f"F16: {len(h16)} tensors identical to ggml_quantize_chunk's")
    assert conv, "nothing was converted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--quantize", help="llama-quantize")
    ap.add_argument("--ggml", help="libggml-base.so of the same build")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--old", nargs="+")
    ap.add_argument("--new", nargs="+")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    synthetic(os.path.join(d, "synthetic"))
    if a.old:
        assert a.quantize and a.ggml and a.new and len(a.old) == len(a.new), "--quantize, --ggml, --old and --new"
        real(os.path.join(d, "real"), a)
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
