#!/usr/bin/env python3
"""Which layers route alike, and whether their weights are related.
  python3 -I layer_pairs.py experts.tsv OUT_DIR [Q8_0_SHARD...]
From expert_usage.py's experts.tsv: for every pair of layers, the Spearman correlation of their experts' routings
(expert e of one layer against expert e of the other), and each layer's best partner (pairs.tsv, best.tsv). With the
shards of a Q8_0 file, for each layer and its best partner (plus control pairs): the cosine of the two router rows of
the same expert, and the Pearson correlation of the dequantized weights of experts 0, 100 and 372 (the first 160 rows
of `ffn_gate_exps` and `ffn_down_exps`), against expert 0 of one layer and expert 1 of the other (weights.tsv).
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import collections, glob, math, os, struct, sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

CONTROLS = ((0, 16), (24, 40), (7, 23), (40, 47))
EXPERTS = (0, 100, 372)
ROWS = 160


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: -x[i])
    r = [0] * len(x)
    for k, i in enumerate(o):
        r[i] = k
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


class Q8:
    def __init__(self, shards):
        self.idx = {}
        for p in shards:
            h = Header(p)
            for t in h.tensors:
                self.idx[t[0]] = (p, h, t)

    def read(self, name, start, n):
        p, h, t = self.idx[name]
        with open(p, "rb") as f:
            f.seek(h.data0 + t[3] + start)
            return t, f.read(n)

    def expert(self, name, e):
        t, _ = self.read(name, 0, 0)
        assert t[2] == 8, f"{name}: type {t[2]}, expected Q8_0"
        row = t[1][0] // 32 * 34
        _, raw = self.read(name, e * t[1][1] * row, ROWS * row)
        out = array("f")
        for b in range(0, len(raw), 34):
            d = struct.unpack_from("<e", raw, b)[0]
            out.extend(d * v for v in struct.unpack_from("<32b", raw, b + 2))
        return out

    def router(self, L):
        t, raw = self.read(f"blk.{L}.ffn_gate_inp.weight", 0, 0)
        assert t[2] == 0, "router: expected F32"
        t, raw = self.read(t[0], 0, 4 * t[1][0] * t[1][1])
        a = array("f")
        a.frombytes(raw)
        return t[1][0], a


def main():
    path, out, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    cnt = collections.defaultdict(dict)
    with open(path) as f:
        next(f)
        for line in f:
            x = line.split("\t")
            cnt[int(x[0])][int(x[1])] = float(x[2])
    layers = sorted(cnt)
    R = {L: ranks([cnt[L][e] for e in range(len(cnt[L]))]) for L in layers}
    n = len(R[layers[0]])
    m, v = (n - 1) / 2, sum((k - (n - 1) / 2) ** 2 for k in range(n))
    rho = {(a, b): sum((x - m) * (y - m) for x, y in zip(R[a], R[b])) / v for a in layers for b in layers if a < b}
    with open(os.path.join(out, "pairs.tsv"), "w") as f:
        f.write("layer_a\tlayer_b\tspearman_routings\n")
        for (a, b), r in sorted(rho.items()):
            f.write(f"{a}\t{b}\t{r:.4f}\n")
    best = {}
    with open(os.path.join(out, "best.tsv"), "w") as f:
        f.write("layer\tbest_partner\tspearman\tsecond\tspearman_second\n")
        for a in layers:
            row = sorted(((rho[min(a, b), max(a, b)], b) for b in layers if b != a), reverse=True)
            best[a] = row[0][1]
            f.write(f"{a}\t{row[0][1]}\t{row[0][0]:.3f}\t{row[1][1]}\t{row[1][0]:.3f}\n")
    if not shards:
        return
    q = Q8(sorted(shards))
    pairs = sorted({(min(a, b), max(a, b)) for a, b in best.items() if rho[min(a, b), max(a, b)] > 0.5})
    with open(os.path.join(out, "weights.tsv"), "w") as f:
        f.write("layer_a\tlayer_b\tkind\tspearman_routings\trouter_cos_same_expert\trouter_cos_other_expert\t"
                + "\t".join(f"{k}_{e}_r" for k in ("gate", "down") for e in EXPERTS) + "\tgate_0_vs_1_r\n")
        for kind, group in (("paired", pairs), ("control", CONTROLS)):
            for a, b in group:
                w, ra = q.router(a)
                _, rb = q.router(b)
                ne = len(ra) // w
                cos = lambda x, y: sum(p * r for p, r in zip(x, y)) / math.sqrt(sum(p * p for p in x) * sum(r * r for r in y))
                same = [cos(ra[e * w:(e + 1) * w], rb[e * w:(e + 1) * w]) for e in range(0, ne, 8)]
                other = [cos(ra[e * w:(e + 1) * w], rb[((e + 1) % ne) * w:((e + 1) % ne + 1) * w]) for e in range(0, ne, 8)]
                rs = [pearson(q.expert(f"blk.{a}.ffn_{k}_exps.weight", e), q.expert(f"blk.{b}.ffn_{k}_exps.weight", e))
                      for k in ("gate", "down") for e in EXPERTS]
                ctl = pearson(q.expert(f"blk.{a}.ffn_gate_exps.weight", 0), q.expert(f"blk.{b}.ffn_gate_exps.weight", 1))
                f.write(f"{a}\t{b}\t{kind}\t{rho[a, b]:.3f}\t{sum(same) / len(same):.3f}\t{sum(other) / len(other):.3f}\t"
                        + "\t".join(f"{r:.3f}" for r in rs) + f"\t{ctl:.3f}\n")
                f.flush()


if __name__ == "__main__":
    main()
