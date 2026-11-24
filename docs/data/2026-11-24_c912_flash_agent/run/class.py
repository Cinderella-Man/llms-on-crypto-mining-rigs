#!/usr/bin/env python3
"""class.py AGENT_TSV WORK_TSV STEPS_MD ARM BASE BAND_RUN BAND_RUN: C912's class of Flash-Next's candidate run.

One run each of the candidate (ARM) and the control (BASE) on the same rig and boot, so neither engine has a spread of
its own. The band is the gap between BAND_RUN and BAND_RUN, two runs of the same started command (R227's t1400a and
t1400b, the v5 preset's line on another rig), but never less than FLOOR (default 1 %) of BASE's value: R227's two runs
agreed to the last digit on some measures, closer than two runs on one rig have agreed before (R156's two v4.0 runs:
2.43 and 2.37 minutes a task at equal work), so a band below 1 % would be narrower than the error. A measure is a gain
or a loss only where ARM differs from BASE by more than the band, else negligible (the rule of
bench/agent_check/class_arms.py with the band as the larger spread). The rule was fixed before the control's tasks
ended.
Measures: agent_tables.py's time columns (AGENT_TSV), the work counts of report.py work (WORK_TSV: prompt tokens read
and reply tokens re-read a task, the longest first-token wait) and the ms a decode step over all requests of
flash_steps.py (STEPS_MD, its `all` rows). Minutes, seconds, ms and work counts are better lower, tokens a second
higher. Prints a TSV. Python 3's standard library only.
"""
import csv
import os
import sys

agent_tsv, work_tsv, steps_md, arm, base, b1, b2 = sys.argv[1:8]
runs = (arm, base, b1, b2)
floor = float(os.environ.get("FLOOR", "1")) / 100
agent = {r["arm"]: r for r in csv.DictReader(open(agent_tsv), delimiter="\t")}
work = {r["arm"]: r for r in csv.DictReader(open(work_tsv), delimiter="\t")}
steps = {}
for line in open(steps_md):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 9 and cells[1] == "all":
        steps[cells[0]] = float(cells[8])

value = {}
for r in runs:
    a, w = agent[r], work[r]
    t = int(w["tasks"])
    value[r] = [
        ("minutes a task at equal work", float(a["equal_work_min_per_task"]), "lower"),
        ("generation tok/s", float(a["gen_tok_s"]), "higher"),
        ("ms a decode step, all requests", steps[r], "lower"),
        ("prompt tok/s", float(a["prompt_tok_s"]), "higher"),
        ("first token median s", float(a["ttft_median_s"]), "lower"),
        ("first token p90 s", float(a["ttft_p90_s"]), "lower"),
        ("first token on a task's first turn, median s", float(a["ttft_turn1_median_s"]), "lower"),
        ("prompt tokens read a task", int(w["prompt_read"]) / t, "lower"),
        ("reply tokens re-read a task", int(w["reread_n"]) / t, "lower"),
        ("longest first-token wait s", float(w["ttft_max_s"]), "lower"),
    ]
print(f"measure\t{arm}\t{base}\t{b1}\t{b2}\tdiff\tdiff_pct\tgap_{b1}_{b2}\tband\tclass")
for i, (label, _, better) in enumerate(value[arm]):
    va, vb, x1, x2 = (value[r][i][1] for r in runs)
    d, gap = va - vb, abs(x1 - x2)
    band = max(gap, floor * abs(vb))
    if abs(d) <= band:
        cls = "negligible"
    else:
        cls = "gain" if (d < 0) == (better == "lower") else "loss"
    pct = f"{d / vb * 100:+.1f} %" if vb else "—"
    print(f"{label}\t{va:.4g}\t{vb:.4g}\t{x1:.4g}\t{x2:.4g}\t{d:+.4g}\t{pct}\t{gap:.4g}\t{band:.4g}\t{cls}")
