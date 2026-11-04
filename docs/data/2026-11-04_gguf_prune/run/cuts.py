#!/usr/bin/env python3
# cuts.py --experts experts.tsv.gz --shards <source shards, comma-separated> DIR...: one row per cut folder (gguf_prune.py's
# --out-dir: its kept.tsv and shards): experts kept, the bytes of every shard written (a linked shard counts as its
# source), the bytes saved against the source, and the share of the ranking table's routings and activation energy
# the kept experts carry, over all layers and in the worst layer. Standard library only.
import argparse, gzip, os


def read_kept(path):
    with open(path) as f:
        return {int(c[0]): [int(v) for v in c[1].split(",")] for c in (l.rstrip("\n").split("\t") for l in f)
                if c[0].isdigit()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    table = {}  # layer -> expert -> (routings, energy)
    with (gzip.open(a.experts, "rt") if a.experts.endswith(".gz") else open(a.experts)) as f:
        head = f.readline().rstrip("\n").split("\t")
        li, ei, ri, gi = (head.index(c) for c in ("layer", "expert", "routings", "energy"))
        for line in f:
            c = line.rstrip("\n").split("\t")
            table.setdefault(int(c[li]), {})[int(c[ei])] = (float(c[ri]), float(c[gi]))
    source = sum(os.path.getsize(s) for s in a.shards.split(","))
    print("folder\tkeep\tshards_bytes\tbytes_saved\twritten_bytes\troutings_kept_pct\tworst_layer_routings_kept_pct\t"
          "worst_layer\tenergy_kept_pct\tworst_layer_energy_kept_pct")
    for d in a.dirs:
        kept = read_kept(os.path.join(d, "kept.tsv"))
        files = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
        total = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        written = sum(os.path.getsize(os.path.join(d, f)) for f in files if not os.path.islink(os.path.join(d, f)))
        r = [0.0] * 4  # routings kept, routings, energy kept, energy
        worst_r, worst_e = (101.0, -1), 101.0
        for layer, ids in kept.items():
            rows = table[layer]
            kr, ke = sum(rows[e][0] for e in ids), sum(rows[e][1] for e in ids)
            tr, te = sum(v[0] for v in rows.values()), sum(v[1] for v in rows.values())
            r = [r[0] + kr, r[1] + tr, r[2] + ke, r[3] + te]
            worst_r, worst_e = min(worst_r, (100 * kr / tr, layer)), min(worst_e, 100 * ke / te)
        keep = len(next(iter(kept.values())))
        print(f"{os.path.basename(d.rstrip('/'))}\t{keep}\t{total}\t{source - total}\t{written}\t{100 * r[0] / r[1]:.2f}\t"
              f"{worst_r[0]:.2f}\t{worst_r[1]}\t{100 * r[2] / r[3]:.2f}\t{worst_e:.2f}")


if __name__ == "__main__":
    main()
