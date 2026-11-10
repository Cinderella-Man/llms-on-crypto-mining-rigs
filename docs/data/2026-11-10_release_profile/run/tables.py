#!/usr/bin/env python3
"""tables.py: the Markdown tables of the release profile, from release_parts.py's parts.tsv (several runs joined)
and pass_times.py's pass_times.tsv.

  python3 -I tables.py --parts parts.tsv --pass-times pass_times.tsv > tables.md

1. The parts of a pass at each k (traces release-k<k>), the cards' GPU time, the unprofiled pass (the first call of
   the wall load's 340-token request / its passes) and their difference: hops, host and gaps. Below it the same
   difference by C120r's convention (traces every-k<k>: every evaluation of the request; the request's prompt time /
   (N / k) of the wall load's 1,680-token request).
2. A pass with one output row against a plain one (output-k4 against release-k4).
3. A five-token pass with the MTP draft loaded against one without (draft-k5-r1, draft-k5-r2 against release-k5).
Standard library only."""
import argparse, csv

ap = argparse.ArgumentParser()
ap.add_argument("--parts", required=True); ap.add_argument("--pass-times", required=True)
a = ap.parse_args()

p, order = {}, []  # trace -> part -> ms
for r in csv.DictReader(open(a.parts), delimiter="\t"):
    p.setdefault(r["trace"], {})[r["part"]] = float(r["ms_per_pass"])
    if r["part"] not in order:
        order.append(r["part"])
t = {(r["arm"], int(r["k"])): r for r in csv.DictReader(open(a.pass_times), delimiter="\t")}
parts = [x for x in order if x != "sum of the parts" and any(v.get(x, 0) > 0.004 for v in p.values())]
ks = sorted(int(x.split("-k")[1]) for x in p if x.startswith("release-k"))


def table(cols, rows):
    print("| " + " | ".join(cols) + " |")
    print("|---" + "|---:" * (len(cols) - 1) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


rows = [[x] + [f"{p[f'release-k{k}'][x]:.2f}" for k in ks] for x in parts]
gpu = {k: p[f"release-k{k}"]["sum of the parts"] for k in ks}
wall = {k: float(t[("wall340", k)]["call1_ms_per_pass"]) for k in ks}
rows.append(["**the cards' GPU time**"] + [f"**{gpu[k]:.2f}**" for k in ks])
rows.append(["the unprofiled pass"] + [f"{wall[k]:.2f}" for k in ks])
rows.append(["**hops, host and gaps**"] + [f"**{wall[k] - gpu[k]:.2f}**" for k in ks])
table(["Part"] + [f"k = {k}" for k in ks], rows)

rows = []
gpu_e = {k: p[f"every-k{k}"]["sum of the parts"] for k in ks}
wall_e = {k: float(t[("wall", k)]["prompt_ms_per_pass"]) for k in ks}
rows.append(["the cards' GPU time, every evaluation of the request"] + [f"{gpu_e[k]:.2f}" for k in ks])
rows.append(["the request's prompt time / (N / k), 1,680 tokens"] + [f"{wall_e[k]:.2f}" for k in ks])
rows.append(["their difference"] + [f"{wall_e[k] - gpu_e[k]:.2f}" for k in ks])
table(["By C120r's convention"] + [f"k = {k}" for k in ks], rows)

if "output-k4" in p:
    cols = ["release-k4", "output-k4"] + (["output-k5"] if "output-k5" in p else [])
    rows = [[x] + [f"{p[c][x]:.2f}" for c in cols] + [f"{p['output-k4'][x] - p['release-k4'][x]:+.2f}"]
            for x in parts + ["sum of the parts"]]
    table(["Part", "plain, 82 passes", "one output row, the k = 4 request's last pass"]
          + (["one output row, the k = 5 request's last pass (4 tokens)"] if "output-k5" in p else []) + ["first − plain"], rows)

if "draft-k5-r2" in p and "release-k5" in p:
    cols = ["release-k5", "draft-k5-r1", "draft-k5-r2"]
    rows = [[x] + [f"{p[c][x]:.2f}" for c in cols] + [f"{p['draft-k5-r2'][x] - p['release-k5'][x]:+.2f}"]
            for x in parts + ["sum of the parts"]]
    table(["Part", "no draft, k = 5", "draft loaded, request 1", "draft loaded, request 2", "request 2 − no draft"], rows)
