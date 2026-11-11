#!/usr/bin/env python3
"""Do the single switches' KLD changes add up? pmix-iq3-fast is pmix-q2-fast with every IQ3_XXS switch at once (plus
the Q2_0 down projections of groups 1-2 in IQ3_XXS's rounding, which no switch file holds). Its measured change from
the start, per text, against the sum of the scored iq3 switches' changes (galloc.py solve's -switches table).
  additivity.py SOLVE_SWITCHES_TSV PASCAL_MIX_KLD_TSV START_DUMP IQ3_DUMP > additivity.tsv
Standard library only."""
import json, sys


def tsv(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f if l.strip()]


sw, pm, start, iq3 = sys.argv[1:5]
rows = [r for r in tsv(sw) if r["name"].startswith(("guiq3", "dniq3", "denseiq3"))]
kld = {(r["name"], r["text"]): float(r["kld"]) for r in tsv(pm)}
print("text\tswitches_summed\tsum_of_changes\tpmix_iq3_fast_change\twhole_over_sum")
for t in ("agent", "wiki", "code"):
    s = sum(float(r[f"dkld_{t}"]) for r in rows)
    whole = kld[("pmix-iq3-fast", t)] - kld[("pmix-q2-fast", t)]
    print(f"{t}\t{len(rows)}\t{s:.5f}\t{whole:.5f}\t{whole / s:.3f}")
a = {n: ty for n, ty, *_ in json.load(open(start))["tensors"]}
b = {n: ty for n, ty, *_ in json.load(open(iq3))["tensors"]}
diff = sorted(n for n in a if not n.startswith("blk.") and a[n] != b.get(n))
print(f"# tensors outside the blocks whose type differs between the two files: {', '.join(diff) or 'none'}")
