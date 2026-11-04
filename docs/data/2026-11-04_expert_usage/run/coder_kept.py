#!/usr/bin/env python3
"""Which of the original model's experts a pruned GGUF kept, per layer, and how much of a measured routing they carry.
  python3 -I coder_kept.py ORIGINAL_SHARD... --pruned PRUNED_SHARD... --experts experts.tsv [--usage-model SHARD...] \
    --out OUT_DIR
Each expert of the pruned file is matched to the original's by its router row (`ffn_gate_inp`, exact bytes) and, as a
check, by the first 4 KiB of its `ffn_down_exps` slice; both files must hold the same tensor types (ISTA's Coder file
and ISTA's IQ3_S do). With expert_usage.py's experts.tsv (the routing of another run of the same base model, whose
shards --usage-model names: its router rows are compared with the original's, every 8th expert, to show that expert
ids mean the same expert in both), it writes:
  kept_ids.tsv  per layer: the original expert ids kept, comma-separated
  kept.tsv      per layer: experts kept, matched by router and by down bytes (agreeing), the share of routings and of
                activation energy the kept set carries, the same for the top N by routings, and the overlap; with
                --usage-model, the cosine of the same expert's router rows in the two models (mean, lowest) and of
                neighbouring experts' (a control)
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import argparse, collections, math, os, re, struct, sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

PREFIX = 4096
STEP = 8


def index(shards):
    idx = {}
    for p in shards:
        h = Header(p)
        for t in h.tensors:
            idx[t[0]] = (p, h, t)
    return idx


def slices(idx, name, n, step, size):
    """n byte strings: `size` bytes at every `step` bytes of the tensor's data"""
    p, h, t = idx[name]
    out = []
    with open(p, "rb") as f:
        for i in range(n):
            f.seek(h.data0 + t[3] + i * step)
            out.append(f.read(size))
    return out


def router_rows(idx, L, ids):
    """the router rows of experts `ids` in layer L as floats (F32, F16 or BF16)"""
    p, h, t = idx[f"blk.{L}.ffn_gate_inp.weight"]
    w, typ = t[1][0], t[2]
    size = {0: 4, 1: 2, 30: 2}[typ]
    rows = []
    with open(p, "rb") as f:
        for e in ids:
            f.seek(h.data0 + t[3] + e * w * size)
            raw = f.read(w * size)
            if typ == 0:
                a = array("f")
                a.frombytes(raw)
            elif typ == 1:
                a = array("f", struct.unpack(f"<{w}e", raw))
            else:
                a = array("f", struct.unpack(f"<{w}f", b"".join(b"\0\0" + raw[i:i + 2] for i in range(0, len(raw), 2))))
            rows.append(a)
    return rows


def cos(x, y):
    return sum(p * q for p, q in zip(x, y)) / math.sqrt(sum(p * p for p in x) * sum(q * q for q in y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", nargs="+")
    ap.add_argument("--pruned", nargs="+", required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--usage-model", nargs="+", help="shards of the model experts.tsv was measured on")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    io, ip = index(a.original), index(a.pruned)
    iu = index(a.usage_model) if a.usage_model else None
    cnt, en = collections.defaultdict(dict), collections.defaultdict(dict)
    with open(a.experts) as f:
        next(f)
        for line in f:
            x = line.split("\t")
            cnt[int(x[0])][int(x[1])] = float(x[2])
            en[int(x[0])][int(x[1])] = float(x[5])
    layers = sorted(int(m.group(1)) for k in ip if (m := re.match(r"^blk\.(\d+)\.ffn_gate_inp\.weight$", k)))
    os.makedirs(a.out, exist_ok=True)
    ids = open(os.path.join(a.out, "kept_ids.tsv"), "w")
    ids.write("layer\tkept_ids\n")
    kt = open(os.path.join(a.out, "kept.tsv"), "w")
    kt.write("layer\tkept\tmatched_router\tmatched_down\tagree\trouting_kept_pct\tenergy_kept_pct\t"
             "topN_routing_kept_pct\ttopN_energy_kept_pct\toverlap_with_topN"
             + ("\tusage_router_cos_same_mean\tusage_router_cos_same_min\tusage_router_cos_next_mean" if iu else "")
             + "\n")
    for L in layers:
        r = f"blk.{L}.ffn_gate_inp.weight"
        to, tp = io[r][2], ip[r][2]
        assert to[2] == tp[2] and to[1][0] == tp[1][0], f"{r}: types or widths differ"
        ne_o, ne_p = to[1][1], tp[1][1]
        row = io[r][1].slot[r] // ne_o
        orig = {b: i for i, b in enumerate(slices(io, r, ne_o, row, row))}
        by_router = [orig.get(b) for b in slices(ip, r, ne_p, row, row)]
        d = f"blk.{L}.ffn_down_exps.weight"
        per_o, per_p = io[d][1].slot[d] // ne_o, ip[d][1].slot[d] // ne_p
        orig_d = {b: i for i, b in enumerate(slices(io, d, ne_o, per_o, PREFIX))}
        by_down = [orig_d.get(b) for b in slices(ip, d, ne_p, per_p, PREFIX)]
        agree = sum(1 for x, y in zip(by_router, by_down) if x is not None and x == y)
        kept = sorted({x for x in by_router if x is not None})
        c, e = cnt[L], en[L]
        T, E = sum(c.values()), sum(e.values())
        topn = sorted(c, key=lambda i: (-c[i], i))[:ne_p]
        ids.write(f"{L}\t{','.join(map(str, kept))}\n")
        kt.write(f"{L}\t{ne_p}\t{sum(x is not None for x in by_router)}\t{sum(x is not None for x in by_down)}\t"
                 f"{agree}\t{100 * sum(c[i] for i in kept) / T:.3f}\t{100 * sum(e[i] for i in kept) / E:.3f}\t"
                 f"{100 * sum(c[i] for i in topn) / T:.3f}\t{100 * sum(e[i] for i in topn) / E:.3f}\t"
                 f"{len(set(kept) & set(topn))}")
        if iu:
            ids_s = list(range(0, ne_o, STEP))
            ro, ru = router_rows(io, L, ids_s), router_rows(iu, L, ids_s)
            rn = router_rows(iu, L, [(e + 1) % ne_o for e in ids_s])
            same = [cos(x, y) for x, y in zip(ro, ru)]
            nxt = [cos(x, y) for x, y in zip(ro, rn)]
            kt.write(f"\t{sum(same) / len(same):.4f}\t{min(same):.4f}\t{sum(nxt) / len(nxt):.4f}")
        kt.write("\n")
        kt.flush()
    ids.close()
    kt.close()


if __name__ == "__main__":
    main()
