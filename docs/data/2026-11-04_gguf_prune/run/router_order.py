#!/usr/bin/env python3
# router_order.py A_SHARD... -- B_SHARD...: do two files of one model number their experts alike? Per layer (0, 7, 23,
# 35, 47), the cosine of router row e (`ffn_gate_inp`, F32 or F16) in file A and in file B for e = 0, 9, 100, 255, 372,
# 511, against row e + 1 of B (a neighbour). Equal rows for the same id mean an expert ranking counted on one file
# applies to the other. Reads 30 router rows per layer. Standard library only.
import math, os, struct, sys
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../scripts"))
from gguf_reshard import Header  # noqa: E402

IDS = [0, 9, 100, 255, 372, 511]
LAYERS = [0, 7, 23, 35, 47]


def routers(paths):
    out = {}
    for p in paths:
        h = Header(p)
        for name, dims, typ, off in h.tensors:
            if name.endswith(".ffn_gate_inp.weight"):
                out[int(name.split(".")[1])] = (p, h.data0 + off, dims[0], typ)
    return out


def rows(t, ids):
    path, off, n, typ = t
    size = {0: 4, 1: 2}[typ]
    res = []
    with open(path, "rb") as f:
        for e in ids:
            raw = os.pread(f.fileno(), n * size, off + e * n * size)
            res.append(array("f", raw) if typ == 0 else array("f", struct.unpack(f"<{n}e", raw)))
    return res


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def main():
    i = sys.argv.index("--")
    a, b = routers(sys.argv[1:i]), routers(sys.argv[i + 1:])
    print("layer\tsame_id_cosines\tnext_id_cosines")
    for l in LAYERS:
        ra, rb = rows(a[l], IDS), rows(b[l], IDS + [e + 1 for e in IDS[:-1]])
        same = [cos(ra[k], rb[k]) for k in range(len(IDS))]
        nxt = [cos(ra[k], rb[len(IDS) + k]) for k in range(len(IDS) - 1)]
        print(f"{l}\t{','.join(f'{c:.4f}' for c in same)}\t{','.join(f'{c:.3f}' for c in nxt)}")


if __name__ == "__main__":
    main()
