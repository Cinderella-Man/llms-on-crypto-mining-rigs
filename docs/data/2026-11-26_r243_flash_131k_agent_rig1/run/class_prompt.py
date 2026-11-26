#!/usr/bin/env python3
"""R243 (Q1921): the line against the preset on prompt reading per bucket of new tokens, by the release check's rule
(bench/agent_check/class_arms.py: a gain or a loss only beyond the larger of the two arms' spreads, the spread being the
difference between an arm's two runs).

    python3 -I class_prompt.py PROMPT_BUCKETS_TSV CONTROL_A ARM_A CONTROL_B ARM_B > class_prompt.tsv

PROMPT_BUCKETS_TSV is run/tables_agent.sh's prompt_buckets.tsv (per run: turns, new tokens, prompt seconds and ms a new
token in each bucket of a turn's new tokens). The measure is ms a new token, better lower; the counts are the turns. A
bucket missing from a run is left out. Rows for "all" tasks, then "without capped" when the file has them. Standard
library only.
"""
import csv
import sys

path, ca, ra, cb, rb = sys.argv[1:6]
val = {}
order = []
for r in csv.DictReader(open(path), delimiter="\t"):
    key = (r["tasks"], r["bucket"])
    if key not in order:
        order.append(key)
    if int(r["turns"]) > 0:
        val[(key, r["run"])] = (float(r["ms_a_token"]), int(r["turns"]))
print("tasks\tbucket\tarm_runs\tbase_runs\tarm_turns\tbase_turns\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for key in order:
    if not all((key, x) in val for x in (ca, ra, cb, rb)):
        continue
    av, bv = [val[(key, x)][0] for x in (ra, rb)], [val[(key, x)][0] for x in (ca, cb)]
    an, bn = [val[(key, x)][1] for x in (ra, rb)], [val[(key, x)][1] for x in (ca, cb)]
    am, bm = sum(av) / 2, sum(bv) / 2
    asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
    cls = "negligible" if abs(d) <= max(asp, bsp) else ("gain" if d < 0 else "loss")
    print(f"{key[0]}\t{key[1]}\t{av[0]:.3f} {av[1]:.3f}\t{bv[0]:.3f} {bv[1]:.3f}\t{an[0]} {an[1]}\t{bn[0]} {bn[1]}\t"
          f"{am:.3f}\t{bm:.3f}\t{asp:.3f}\t{bsp:.3f}\t{d:+.3f}\t{d / bm * 100:+.1f} %\t{cls}")
