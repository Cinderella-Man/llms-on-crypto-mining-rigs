#!/usr/bin/env python3
"""C1175 (Q1908): the arm against the preset on the agent check's decode step per cell bucket and on the first-token
wait of each part of a task, by the release check's rule (bench/agent_check/class_arms.py: a gain or a loss only
beyond the larger of the two arms' spreads, the spread being the difference between an arm's two runs).

    python3 -I class_cells.py STEPS_CELLS_MD FIRST_TOKEN_TSV AGENT_DIR CONTROL_A ARM_A CONTROL_B ARM_B > class_cells.tsv

STEPS_CELLS_MD is run/steps_cells.py's table (ms a decode step per bucket of the request's middle context: under 4,102
cells every arm runs the dense kernels, from 4,102 to 6,153 only _MIN_RATIO=0.5 runs the per-token sparse kernel, from
6,153 every arm runs it, the preset in 3 splits and _NSM=10 in 5). FIRST_TOKEN_TSV is tables_agent.sh's first_token.tsv
(the median and 90th percentile of every turn after a task's first). AGENT_DIR holds each run's turns.csv
(agent/<run>/turns.csv), from which the median first-token wait of the tasks' first turns is taken and both parts' turns are
counted (arm_count, base_count: requests for a step, turns for a first-token wait). A bucket
missing from a run is left out of that measure. Standard library only.
"""
import csv
import statistics
import sys

steps_md, ft_tsv, agent_dir, ca, ra, cb, rb = sys.argv[1:8]
val = {}
for ln in open(steps_md):
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) == 9 and c[0] in (ca, ra, cb, rb):
        val[(c[1], c[0])] = (float(c[8]), int(c[2]))
n_first, n_later = {}, {}
for run in (ca, ra, cb, rb):
    rows = [r for r in csv.DictReader(open(f"{agent_dir}/{run}/turns.csv")) if r["attempt"] == "1" and r["ttft_s"]]
    first = [float(r["ttft_s"]) for r in rows if r["turn"] == "1"]
    n_first[run], n_later[run] = len(first), len(rows) - len(first)
    val[("first token, first turns, median s", run)] = (statistics.median(first), n_first[run])
for r in csv.DictReader(open(ft_tsv), delimiter="\t"):
    if r["run"] in n_later:
        val[("first token, later turns, median s", r["run"])] = (float(r["later_turns_median_s"]), n_later[r["run"]])
        val[("first token, later turns, p90 s", r["run"])] = (float(r["later_turns_p90_s"]), n_later[r["run"]])
measures = sorted({m for m, _ in val}, key=lambda m: (not m[0].isdigit() and m != "all", m == "all", m))
print("measure\tarm_runs\tbase_runs\tarm_count\tbase_count\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for m in measures:
    if not all((m, r) in val for r in (ca, ra, cb, rb)):
        continue
    av, bv = [val[(m, r)][0] for r in (ra, rb)], [val[(m, r)][0] for r in (ca, cb)]
    an, bn = [val[(m, r)][1] for r in (ra, rb)], [val[(m, r)][1] for r in (ca, cb)]
    am, bm = sum(av) / 2, sum(bv) / 2
    asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
    cls = "negligible" if abs(d) <= max(asp, bsp) else ("gain" if d < 0 else "loss")
    label = "ms a decode step, all requests" if m == "all" else (
        m if not m[0].isdigit() else f"ms a decode step, middle context {m} cells")
    print(f"{label}\t{av[0]:.2f} {av[1]:.2f}\t{bv[0]:.2f} {bv[1]:.2f}\t{an[0]} {an[1]}\t{bn[0]} {bn[1]}\t{am:.3f}\t{bm:.3f}\t"
          f"{asp:.2f}\t{bsp:.2f}\t{d:+.3f}\t{d / bm * 100:+.1f} %\t{cls}")
