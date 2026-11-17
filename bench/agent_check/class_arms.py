#!/usr/bin/env python3
"""class_arms.py AGENT_TSV NAME=RUN,RUN NAME=RUN,RUN ...: the release check's class of each pair of arms.

AGENT_TSV is agent_tables.py's table (one row a run). Each NAME=RUN,RUN names an arm and its runs (the rows' `arm`
column; two or more, or one, whose spread is then 0), the first NAME is the arm being classed, every later NAME a base
it is classed against. Per measure and base: the runs by value, the two means, each arm's spread (the difference
between its highest and lowest run), the difference of the means and its share of the base's mean (— when the base's
mean is 0), and the class. The rule: a gain or a loss only where the difference of the means is beyond the larger of
the two arms' spreads, else negligible. Minutes, first-token seconds and the work counts are better lower, tokens a
second higher.

The measures, in this order:
  - time: minutes a task at equal work, generation and prompt tok/s, first token median, p90 and on a task's first
    turn (the measures of docs/data/2026-11-14_r159_flint_release_agent/run/class_arms.py, whose rows these are, cell
    for cell);
  - work (C855, Q1526), from agent_tables.py's work columns: prompt tokens read a task, reply tokens re-read a task,
    the prompt seconds of the turns that re-read a reply a task, and the longest first-token wait. They are each run's
    own counts, so a change that reads fewer tokens (text reuse) is classed here while its minutes at equal work, which
    hold the tokens fixed, stay negligible.
A measure whose column the table lacks (a table made before the work columns) is left out. Prints a TSV. Runs anywhere
with Python 3 (no packages).
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
    ("prompt_read_per_task", "prompt tokens read a task", "lower"),
    ("reread_per_task", "reply tokens re-read a task", "lower"),
    ("reread_s_per_task", "prompt s of the turns that re-read a reply, a task", "lower"),
    ("ttft_max_s", "longest first-token wait s", "lower"),
]

path, specs = sys.argv[1], sys.argv[2:]
with open(path) as fh:
    table = csv.DictReader(fh, delimiter="\t")
    columns = table.fieldnames
    rows = {r["arm"]: r for r in table}
arms = []
for s in specs:
    name, runs = s.split("=", 1)
    arms.append((name, runs.split(",")))
(name, runs), bases = arms[0], arms[1:]
print("measure\tarm\tbase\tarm_runs\tbase_runs\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass")
for col, label, better in MEASURES:
    if col not in columns:
        continue
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
        share = f"{d / mb * 100:+.1f} %" if mb else "—"
        print(f"{label}\t{name}\t{bname}\t{' '.join(f'{x:g}' for x in a)}\t{' '.join(f'{x:g}' for x in b)}\t"
              f"{ma:.4g}\t{mb:.4g}\t{sa:.4g}\t{sb:.4g}\t{d:+.4g}\t{share}\t{cls}")
