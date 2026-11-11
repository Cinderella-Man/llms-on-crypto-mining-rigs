#!/usr/bin/env python3
"""r82_counts.py: R82 (Q1297), how each graph evaluation ran in a graph-profile load, prompt against generation.

  python3 -I r82_counts.py --load <name>=<load dir> [--load ...] --out <tsv> [--odd <tsv>]

A load dir is one rig_taper.sh OUT with load-A/server.log and prof-graph.csv (GGML_CUDA_PROFILE_MODE=graph), as
timeline.py (docs/data/2026-11-10_prompt_timeline/run/) reads it; the request windows, the main/draft split and the
clock offset are timeline.py's. Unlike timeline.py's counts.tsv, an evaluation whose GPU time was not ready keeps the
"how" the fork wrote for it (eager, capture or replay). Per load and kind (main or draft):
  prompt    evaluations inside a request's prompt window (timeline.py's: the main micro-batches queued between the
            request's launch and launch + prompt time, and the draft evaluations between the first of them and the end)
  first     of the prompt's main evaluations, each card's first in the load
  other     everything else: generation (decode and verify steps, the draft's steps) and the replay's short prompts
            that fall outside a window
Writes one row per (load, kind, part, how) with the count; --odd lists every main prompt evaluation after a card's
first that ran eagerly (request, micro-batch, card, launch groups). Standard library only.
"""
import argparse
import bisect
import csv
import os
import re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="append", required=True, help="name=dir")
ap.add_argument("--out", required=True)
ap.add_argument("--odd", default="")
ap.add_argument("--cards", type=int, default=9)
ap.add_argument("--tol", type=float, default=1.5)
a = ap.parse_args()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def stamp(line):  # "M.SS.mmm.uuu ..." -> ms since the server's start
    m = re.match(r"(\d+)\.(\d+)\.(\d+)\.(\d+) ", line)
    return (int(m[1]) * 60 + int(m[2])) * 1000 + int(m[3]) + int(m[4]) / 1000 if m else None


def parse_log(path):
    tasks, reallocs, cur = [], [], None
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        t = stamp(line)
        if t is None:
            continue
        if "graph reallocation:" in line:
            reallocs.append(t)
        if "launch_slot_" in line and "processing task" in line:
            cur = {"launch": t, "prompt_ms": None}
            tasks.append(cur)
        elif cur is not None and "prompt eval time" in line:
            cur["prompt_ms"] = float(re.search(r"prompt eval time =\s*([\d.]+) ms", line)[1])
    return tasks, reallocs


def parse_trace(path):
    recs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            recs.append({"dev": int(r["device"]), "t": float(r["host_ms"]), "how": r["how"],
                         "groups": int(r["groups"]), "eval_groups": int(r["eval_groups"])})
    recs.sort(key=lambda x: (x["t"], x["dev"]))
    return recs


def classify(recs):
    mb, expect = -1, 0
    for r in recs:
        if r["dev"] == expect:
            if expect == 0:
                mb += 1
            r["main"], r["mb"] = True, mb
            expect = (expect + 1) % a.cards
        else:
            r["main"], r["mb"] = False, None


def align(recs, lines):
    ev = sorted(r["t"] for r in recs if not r["main"])
    if not ev or not lines:
        return None

    def matched(off):
        n = 0
        for L in lines:
            j = bisect.bisect_left(ev, L - off - a.tol)
            if j < len(ev) and ev[j] <= L - off + a.tol:
                n += 1
        return n

    return max((L - e for L in lines[:12] for e in ev), key=matched)


rows, odd = [], []
for spec in a.load:
    name, d = spec.split("=", 1)
    tasks, reallocs = parse_log(os.path.join(d, "load-A", "server.log"))
    recs = parse_trace(os.path.join(d, "prof-graph.csv"))
    classify(recs)
    off = align(recs, reallocs)
    part = {}
    for i, t in enumerate(tasks):
        if off is None or t["prompt_ms"] is None:
            continue
        w0, w1 = t["launch"] - off, t["launch"] - off + t["prompt_ms"]
        mbs = sorted({x["mb"] for x in recs if x["main"] and w0 <= x["t"] <= w1})
        mine = [k for k, x in enumerate(recs) if x["main"] and x["mb"] in set(mbs)]
        if not mine:
            continue
        first, j_last = recs[mine[0]]["t"], mine[-1]
        end = recs[j_last + 1]["t"] if j_last + 1 < len(recs) else w1
        for k, x in enumerate(recs):
            if (x["main"] and x["mb"] in set(mbs)) or (not x["main"] and first <= x["t"] <= end):
                part[k] = (i + 1, mbs.index(x["mb"]) if x["main"] else None)
    seen = set()
    c = Counter()
    for k, x in enumerate(recs):
        kind = "main" if x["main"] else "draft"
        if k in part:
            p = "prompt"
            if x["main"] and x["dev"] not in seen:
                seen.add(x["dev"])
                p = "first"
            elif x["main"] and x["how"] == "eager":
                odd.append({"load": name, "request": part[k][0], "micro_batch": part[k][1], "card": x["dev"],
                            "eval_groups": x["eval_groups"], "groups": x["groups"]})
        else:
            p = "other"
        c[(kind, p, x["how"])] += 1
    for (kind, p, how), n in sorted(c.items()):
        rows.append({"load": name, "evaluations": kind, "part": p, "how": how, "n": n})
    print(f"{name}: {len(recs)} evaluations, clock offset {('%.1f ms' % off) if off is not None else '-'}, "
          f"{len(tasks)} requests, {len(part)} evaluations in prompt windows")


def write(fn, rs):
    cols = list(rs[0]) if rs else ["load"]
    with open(fn, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rs:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


write(a.out, rows)
if a.odd:
    write(a.odd, odd)
