#!/usr/bin/env python3
"""Per-layer RMS of the largest input channel of selected matrices, from a GGUF importance matrix.

An imatrix GGUF (llama-imatrix's output) stores, per weight matrix, the sum of squared activations of every input
channel (`<tensor>.in_sum2`) and how many tokens were summed (`<tensor>.counts`). For each matching tensor this prints
the RMS of its largest input channel (and which expert and channel it is), the second largest and the median. Channels
far above the rest are the ones that overflow 16-bit block sums and that plain 4-bit rounding handles worst: on Flint
they sit in the routed down-projections of the last layers (the reason for llama.cpp's MoE down-projection rescale and
for the fork's FP32 activation sums, registry row Q76). Header and float32 arrays only, no numpy.

    scripts/imatrix_rms.py imatrix.gguf [regex]          # default regex: ffn_down
    scripts/imatrix_rms.py imatrix_unsloth.gguf 'ffn_down_exps'
"""
import math
import re
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        (at,) = rd(f, "I")
        (n,) = rd(f, "Q")
        return [rval(f, at) for _ in range(n)]
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return rd(f, fmt)[0]


path = sys.argv[1]
pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"ffn_down")
with open(path, "rb") as f:
    assert f.read(4) == b"GGUF"
    rd(f, "I")
    nt, nkv = rd(f, "QQ")
    align = 32
    for _ in range(nkv):
        k = rstr(f)
        (t,) = rd(f, "I")
        v = rval(f, t)
        if k == "general.alignment":
            align = v
    infos = {}
    for _ in range(nt):
        name = rstr(f)
        (nd,) = rd(f, "I")
        dims = rd(f, "Q" * nd)
        (t,) = rd(f, "I")
        (off,) = rd(f, "Q")
        infos[name] = (dims, t, off)
    base = (f.tell() + align - 1) // align * align

    def load(name):
        dims, t, off = infos[name]
        assert t == 0, (name, t)
        n = math.prod(dims)
        f.seek(base + off)
        return dims, struct.unpack(f"<{n}f", f.read(4 * n))

    rows = []
    for name in sorted(infos, key=lambda s: (int(re.search(r"blk\.(\d+)", s).group(1)) if "blk." in s else -1, s)):
        if not name.endswith(".in_sum2") or not pat.search(name):
            continue
        tname = name[: -len(".in_sum2")]
        dims, s2 = load(name)
        _, cnt = load(tname + ".counts")
        ncols = dims[0]
        nmat = len(s2) // ncols
        best = (0.0, 0, 0)
        for e in range(nmat):
            c = cnt[e] if len(cnt) == nmat else cnt[0]
            if c <= 0:
                continue
            for i in range(ncols):
                ms = s2[e * ncols + i] / c
                if ms > best[0]:
                    best = (ms, e, i)
        # second largest channel of the same matrix
        e = best[1]
        c = cnt[e] if len(cnt) == nmat else cnt[0]
        col = sorted((s2[e * ncols + i] / c for i in range(ncols)), reverse=True)
        print(f"{tname:32s} max channel RMS {math.sqrt(best[0]):10.1f} (matrix {best[1]}, channel {best[2]}), "
              f"2nd {math.sqrt(col[1]):9.1f}, median {math.sqrt(col[len(col) // 2]):7.2f}")
