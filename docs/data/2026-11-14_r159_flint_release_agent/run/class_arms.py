#!/usr/bin/env python3
"""class_arms.py AGENT_TSV NAME=RUN,RUN NAME=RUN,RUN ...: the release check's class of each pair of arms.

AGENT_TSV is agent_tables.py's table (one row a run). Each NAME=RUN,RUN names an arm and its two runs (the rows'
`arm` column), the first NAME is the arm being classed, every later NAME a base it is classed against. Per measure
and base: both runs by value, the two means, each arm's spread (the difference between its two runs), the difference
of the means and its share of the base's mean, and the class. The rule: a gain or a loss only where the difference of
the means is beyond the larger of the two arms' spreads, else negligible. Minutes and first-token seconds are better
lower, tokens a second higher. Prints a TSV. Runs anywhere with Python 3 (no packages).
"""
import csv
import sys

MEASURES = [  # column, label, better
    ("equal_work_min_per_task", "minutes a task at equal work", "lower"),
    ("gen_tok_s", "generation tok/s", "higher"),
    ("prompt_tok_s", "prompt tok/s", "higher"),
    ("ttft_median_s", "first token median s", "lower"),
    ("ttft_p90_s", "first token p90 s", "lower"),
    ("ttft_turn1_median_s", "first token on a task's first turn, median s", "lower"),
]

path, specs = sys.argv[1], sys.argv[2:]
rows = {r["arm"]: r for r in csv.DictReader(open(path), delimiter="\t")}
arms = []
for s in specs:
    name, runs = s.split("=", 1)
    arms.append((name, runs.split(",")))
(name, runs), bases = arms[0], arms[1:]
print("measure\tarm\tbase\tarm_runs\tbase_runs\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for col, label, better in MEASURES:
    a = [float(rows[r][col]) for r in runs]
    ma, sa = sum(a) / len(a), max(a) - min(a)
    for bname, bruns in bases:
        b = [float(rows[r][col]) for r in bruns]
        mb, sb = sum(b) / len(b), max(b) - min(b)
        d = ma - mb
        if abs(d) <= max(sa, sb):
            cls = "negligible"
        else:
            cls = "gain" if (d < 0) == (better == "lower") else "loss"
        print(f"{label}\t{name}\t{bname}\t{' '.join(f'{x:g}' for x in a)}\t{' '.join(f'{x:g}' for x in b)}\t"
              f"{ma:.4g}\t{mb:.4g}\t{sa:.4g}\t{sb:.4g}\t{d:+.4g}\t{d / mb * 100:+.1f} %\t{cls}")
