#!/usr/bin/env python3
"""plan_table.py: the micro-batches of every prompt call size of C222's and R225's replays, before and after cuts=1.

Inputs: the rig's own lines (each replay folder's requests/load-A.tsv and taper_lines.txt, one "ubatch taper:" line
per request in order, the setting in the request's arm column) and run.sh's plan check on a CPU
(test-ubatch-taper --plan-check, n_rs_seq 3 as Flash-Next's line: plans-rs3.tsv). Per setting and call size:
  rig        the pieces the rig's server logged (every distinct plan seen)
  before     the CPU plan check without cuts=1 (the build's old plan)
  rig_same   whether before is the rig's plan
  after      the CPU plan check with cuts=1
  pieces_before, pieces_after  micro-batches in the call
  bad_before, bad_after  pieces under 9 tokens that are not a whole segment between the in-call cuts (4 + 512 and 4
             tokens before the call's end), as "size@offset"
Writes plans.tsv (the table) and prints the summary per setting (Markdown).

    python3 -I plan_table.py --rig ../2026-11-20_c222_flash_microbatch ../2026-11-20_r225_flash_taper1400 \
        --cpu plans-rs3.tsv --out plans.tsv > plans_summary.md

Standard library only.
"""
import argparse
import collections
import csv
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--rig", nargs="+", required=True, help="replay data folders (requests/load-A.tsv, taper_lines.txt)")
ap.add_argument("--cpu", required=True, help="the plan check's TSV (test-ubatch-taper --plan-check --tsv)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

MIN, N_UBATCH = 9, 512
pat = re.compile(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$")

rig = collections.defaultdict(set)  # (setting, n) -> plans
for folder in a.rig:
    reqs = list(csv.DictReader(open(os.path.join(folder, "requests", "load-A.tsv")), delimiter="\t"))
    lines = [ln.strip() for ln in open(os.path.join(folder, "taper_lines.txt")) if ln.strip()]
    assert len(reqs) == len(lines), (folder, len(reqs), len(lines))
    for r, ln in zip(reqs, lines):
        m = pat.search(ln)
        n = int(m.group(1))
        assert n == int(r["prompt_n"]), (folder, r, ln)
        rig[(r["arm"], n)].add(m.group(5).strip())

cpu = {}
for r in csv.DictReader(open(a.cpu), delimiter="\t"):
    cpu[(r["spec"], int(r["call_tokens"]))] = r["logged"]


def sizes(plan):
    out = []
    for t in plan.split():
        n, _, c = t.partition("x")
        out += [int(n)] * (int(c) if c else 1)
    return out


def bad(n, plan):
    bounds = sorted({0, n} | {n - off for off in (4 + N_UBATCH, 4) if n > off})
    res, pos = [], 0
    for s in sizes(plan):
        if s < MIN and not (pos in bounds and pos + s in bounds and not any(pos < b < pos + s for b in bounds)):
            res.append(f"{s}@{pos}")
        pos += s
    return res


settings = sorted({s for s, _ in rig}, key=lambda s: (int(re.search(r":(\d+)", s).group(1)), s))
rows = []
for s in settings:
    for n in sorted({n for t, n in rig if t == s}):
        r = sorted(rig[(s, n)])
        before = cpu.get((s, n), "-")
        after = cpu.get((s + ",cuts=1", n), "-")
        rows.append({"setting": s, "call_tokens": n, "rig": " | ".join(r), "before": before,
                     "rig_same": "yes" if r == [before] else "no", "after": after,
                     "changed": "yes" if after != before else "no",
                     "pieces_before": len(sizes(before)) if before != "-" else "-",
                     "pieces_after": len(sizes(after)) if after != "-" else "-",
                     "bad_before": " ".join(bad(n, before)) or "-", "bad_after": " ".join(bad(n, after)) or "-"})

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

print("| Setting | Call sizes | CPU plan = rig's | Calls with a piece under 9 tokens beside a cut, before (size@offset) | After (cuts=1) | Calls whose plan changes | Micro-batches over the call sizes, before and after |")
print("|---|---:|---:|---|---:|---:|---:|")
for s in settings:
    rs = [r for r in rows if r["setting"] == s]
    nb = [r for r in rs if r["bad_before"] != "-"]
    na = [r for r in rs if r["bad_after"] != "-"]
    nc = sum(r["changed"] == "yes" for r in rs)
    which = ", ".join(f"{r['call_tokens']:,} ({r['bad_before']})" for r in nb) or "none"
    pb = sum(r["pieces_before"] for r in rs if r["pieces_before"] != "-")
    pa = sum(r["pieces_after"] for r in rs if r["pieces_after"] != "-")
    print(f"| `{s}` | {len(rs)} | {sum(r['rig_same'] == 'yes' for r in rs)} of {len(rs)} | {len(nb)}: {which} | {len(na)} | {nc} | {pb:,}, {pa:,} |")
