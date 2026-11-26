#!/usr/bin/env python3
"""summary.py: every prime of C1139's lanes in one table, from prompt_parts.py's --tsv files (Q1872).

  python3 -I summary.py --base pmix --other gmix LANE=prompt_parts.tsv [LANE=prompt_parts2.tsv ...] > summary.md

Per lane and prime: the tokens read and the main evaluations on card 0 (micro-batches), each file's plain prompt time
and speed, the gap, and its split by prompt_parts.py's pricing (est.): the GPU-time difference priced at f, the
pipeline loss, the routed gate/up excess over Q2_0's rate per type priced at f, and what a kernel taking IQ1_M at
Q2_0's rate with no host wait would win back (its excess at f plus the pipeline loss, which the host profile puts on
the two cards that hold the IQ1_M layers). Standard library only."""
import argparse
import collections
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--other", required=True)
ap.add_argument("lanes", nargs="+")
A = ap.parse_args()
rows = []
types = set()
for spec in A.lanes:
    lane, _, path = spec.partition("=")
    d = collections.defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rq = r["request"]
            if r["section"] == "prompt" and r["key"] == "plain":
                d[rq][("plain", r["file"])] = (int(r["count"]), float(r["ms"]))
            elif r["section"] == "evals":
                d[rq][("evals", r["file"])] = int(r["count"])
            elif r["section"] == "price":
                d[rq][("price", r["key"])] = float(r["ms"])
                if r["key"] not in ("f", "pipeline_loss", "gpu_diff_at_f"):
                    types.add(r["key"])
    for rq, v in d.items():
        rows.append((lane, rq, v))
types = sorted(types)
print("| lane | prime | tokens read | micro-batches (card 0) | `%s` plain ms (tok/s) | `%s` plain ms (tok/s) | gap ms | "
      "GPU difference at f | pipeline loss | %s | IQ1_M kernel at Q2_0's rate, no host wait (est.) |"
      % (A.base, A.other, " | ".join(f"{t} gate/up at f" for t in types)))
print("|---|---|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(types) + "---:|")
for lane, rq, v in rows:
    if ("plain", A.base) not in v or ("plain", A.other) not in v:
        continue
    nb, mb = v[("plain", A.base)]
    no, mo = v[("plain", A.other)]
    gap = mo - mb
    ev = v.get(("evals", A.base), "")
    pipe = v.get(("price", "pipeline_loss"), 0.0)
    gdf = v.get(("price", "gpu_diff_at_f"), 0.0)
    iq1 = v.get(("price", "iq1_m"), 0.0)
    cells = " | ".join(f"{v.get(('price', t), 0.0):+,.0f}" for t in types)
    print(f"| {lane} | {rq} | {nb:,} | {ev} | {mb:,.1f} ({nb * 1000 / mb:.1f}) | {mo:,.1f} ({no * 1000 / mo:.1f}) | "
          f"{gap:+,.1f} ({100 * gap / mb:+.1f} %) | {gdf:+,.0f} | {pipe:+,.0f} | {cells} | "
          f"{iq1 + pipe:+,.0f} ({100 * (iq1 + pipe) / gap:.1f} % of the gap) |")
