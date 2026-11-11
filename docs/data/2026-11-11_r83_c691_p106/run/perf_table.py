#!/usr/bin/env python3
"""perf_table.py: R83 (Q1300). The operator times of 2026-11-11_c691_sampling_launches/run/rig_c691.sh's perf step as
tables: TOP_K at 248,320 columns (k = 20) over one, two and four rows, and the PAD of four such rows, per pass; then
what a verify pass's four sampler chains launch without the switch (four one-row TOP_K and the PAD) against with it
(one four-row TOP_K).

  python3 -I perf_table.py US_RUN --out DIR > perf.md

US_RUN is rig_c691.sh's perf/us_run.tsv: op, pass, case string, us/run (tab-separated, no header).
Out: DIR/perf.tsv (op, rows, one column per pass), DIR/chain.tsv (per pass: four one-row TOP_K, the PAD, their sum,
the four-row TOP_K, the saving). Standard library only."""
import argparse, collections, os, re

ap = argparse.ArgumentParser()
ap.add_argument("us_run"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
NE = re.compile(r"ne(?:_a)?=\[(\d+),(\d+),")
us = collections.OrderedDict()  # (op, rows) -> {pass: us}
passes = set()
for line in open(a.us_run):
    f = line.rstrip("\n").split("\t")
    if len(f) != 4:
        continue
    op, p, case, v = f
    m = NE.search(case)
    if not m:
        continue
    us.setdefault((op, int(m.group(2))), {})[int(p)] = float(v)
    passes.add(int(p))
passes = sorted(passes)
keys = sorted(us, key=lambda k: (k[0] != "TOP_K", k[1]))
with open(os.path.join(a.out, "perf.tsv"), "w") as w:
    w.write("op\trows\t" + "\t".join(f"us_pass{p}" for p in passes) + "\n")
    for k in keys:
        w.write(f"{k[0]}\t{k[1]}\t" + "\t".join(f"{us[k].get(p, float('nan')):.2f}" for p in passes) + "\n")
print("## The operator times, card 0, µs a launch\n")
print("| Op | Rows | " + " | ".join(f"Pass {p}" for p in passes) + " |")
print("|---|---:" + "|---:" * len(passes) + "|")
for k in keys:
    print(f"| {k[0]} | {k[1]} | " + " | ".join(f"{us[k].get(p, float('nan')):.2f}" for p in passes) + " |")

g = lambda op, r, p: us.get((op, r), {}).get(p, float("nan"))
cols = ["pass", "top_k_1row", "four_top_k_1row", "pad_4rows", "off_chain_ops", "top_k_4rows", "on_chain_ops", "saving",
        "top_k_4rows_over_1row"]
rows = []
for p in passes:
    one, four, pad = g("TOP_K", 1, p), g("TOP_K", 4, p), g("PAD", 4, p)
    off = 4 * one + pad
    rows.append(dict(zip(cols, [p, one, 4 * one, pad, off, four, four, off - four, four / one])))
with open(os.path.join(a.out, "chain.tsv"), "w") as w:
    w.write("\t".join(cols) + "\n")
    for x in rows:
        w.write("\t".join(str(x[c]) if c == "pass" else f"{x[c]:.2f}" for c in cols) + "\n")
print("\n## A verify pass's four chains: the launches the switch changes, µs\n")
print("| Pass | One-row TOP_K | Four of them | PAD (4 rows) | Off: four TOP_K + PAD | On: one four-row TOP_K | Saving | Four rows / one row |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for x in rows:
    print(f"| {x['pass']} | {x['top_k_1row']:.2f} | {x['four_top_k_1row']:.2f} | {x['pad_4rows']:.2f} | {x['off_chain_ops']:.2f} | "
          f"{x['top_k_4rows']:.2f} | {x['saving']:.2f} | {x['top_k_4rows_over_1row']:.2f} |")
