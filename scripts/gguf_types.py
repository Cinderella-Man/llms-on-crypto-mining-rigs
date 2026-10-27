#!/usr/bin/env python3
"""Lists the tensor types of selected per-layer tensors in GGUF files, with the layers that have each type
(header only, no numpy).

    scripts/gguf_types.py <regex> file.gguf...
    scripts/gguf_types.py '^ffn_' flint-UD-Q4_K_XL.gguf flint-S4L0.gguf
    scripts/gguf_types.py 'ffn_down_exps' *.gguf

The regex is matched against the tensor name after "blk.<layer>.", e.g. "ffn_down_exps.weight"; each file prints one
line per tensor name and type, with its layers as ranges (e.g. "layers 2-9,12,40-48").
"""
import re
import struct
import sys
from collections import defaultdict

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 39: "MXFP4"}
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


def main():
    pat = re.compile(sys.argv[1])
    for path in sys.argv[2:]:
        with open(path, "rb") as f:
            assert f.read(4) == b"GGUF"
            rd(f, "I")
            nt, nkv = rd(f, "QQ")
            for _ in range(nkv):
                rstr(f)
                (t,) = rd(f, "I")
                skip_val(f, t)
            groups = defaultdict(list)
            for _ in range(nt):
                name = rstr(f)
                (nd,) = rd(f, "I")
                dims = rd(f, "Q" * nd)
                (t,) = rd(f, "I")
                rd(f, "Q")
                m = re.match(r"blk\.(\d+)\.(.*)", name)
                if m and pat.search(m.group(2)):
                    groups[(m.group(2), TYPES.get(t, str(t)))].append(int(m.group(1)))
        print(f"== {path.split('/')[-1]}")
        for (tn, ty), layers in sorted(groups.items()):
            print(f"  {tn:28s} {ty:8s} layers {compress(sorted(layers))}")


def compress(ls):
    out, i = [], 0
    while i < len(ls):
        j = i
        while j + 1 < len(ls) and ls[j + 1] == ls[j] + 1:
            j += 1
        out.append(str(ls[i]) if i == j else f"{ls[i]}-{ls[j]}")
        i = j + 1
    return ",".join(out)


if __name__ == "__main__":
    main()
