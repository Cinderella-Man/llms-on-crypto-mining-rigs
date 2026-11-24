#!/usr/bin/env python3
"""steps_mix.py STEPS_MD RUN RUN ...: the ms a decode step of each named run at one shared mix of contexts.

STEPS_MD is flash_steps.py's table (per run and context bucket: requests, tokens, decode s, tok/s, acceptance, tokens a
step, ms a step). A run's ms a step in each bucket every named run decoded in is weighted by the tokens all the named
runs together generated in that bucket, so runs whose replies were longer or sat deeper in the context compare on the
same depths (C854's decode_by_context.py's mix, applied to the decode step, which takes the draft's acceptance out).
Prints a Markdown table: the mix, then per run the ms a step and its tokens a step at that mix. Standard library only.
"""
import sys

path, names = sys.argv[1], sys.argv[2:]
rows = {}
for line in open(path):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 9 and c[0] in names and c[1] != "all" and c[3].isdigit():
        rows.setdefault(c[0], {})[c[1]] = (int(c[3]), float(c[7]), float(c[8]))  # tokens, tokens a step, ms a step
shared = [b for b in rows[names[0]] if all(b in rows[n] for n in names)]
weight = {b: sum(rows[n][b][0] for n in names) for b in shared}
total = sum(weight.values())
print(f"Shared mix: the {len(shared)} buckets every run decoded in, weighted by the {total:,} tokens the runs "
      "generated there: " + ", ".join(f"{b} {weight[b] / total * 100:.1f} %" for b in shared) + ".")
print()
print("| run | ms a step at the shared mix | tokens a step at the shared mix |")
print("|---|---:|---:|")
for n in names:
    ms = sum(rows[n][b][2] * weight[b] for b in shared) / total
    tps = sum(rows[n][b][1] * weight[b] for b in shared) / total
    print(f"| {n} | {ms:.2f} | {tps:.2f} |")
