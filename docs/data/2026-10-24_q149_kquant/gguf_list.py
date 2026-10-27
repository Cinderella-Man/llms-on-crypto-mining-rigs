#!/usr/bin/env python3
"""List every tensor of a GGUF file: name, type, shape, bytes (header only, no numpy)."""
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16", 39: "MXFP4"}
# (block size, bytes per block)
BLK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
       10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 23: (256, 136)}
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    if t == 8:
        rstr(f)
    elif t == 9:
        (at,) = rd(f, "I")
        (n,) = rd(f, "Q")
        if at == 8:
            for _ in range(n):
                rstr(f)
        elif at in SIZES:
            f.seek(SIZES[at] * n, 1)
        else:
            for _ in range(n):
                skip_val(f, at)
    else:
        f.seek(SIZES[t], 1)


with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    rd(f, "I")
    nt, nkv = rd(f, "QQ")
    for _ in range(nkv):
        rstr(f)
        (t,) = rd(f, "I")
        skip_val(f, t)
    for _ in range(nt):
        name = rstr(f)
        (nd,) = rd(f, "I")
        dims = rd(f, "Q" * nd)
        (t,) = rd(f, "I")
        rd(f, "Q")
        n = 1
        for d in dims:
            n *= d
        bs, bb = BLK.get(t, (1, 0))
        print(f"{name}\t{TYPES.get(t, t)}\t{'x'.join(map(str, dims))}\t{n // bs * bb}")
