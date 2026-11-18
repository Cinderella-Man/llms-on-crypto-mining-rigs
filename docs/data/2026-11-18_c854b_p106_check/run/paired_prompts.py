#!/usr/bin/env python3
"""paired_prompts.py REF=TURNS.csv RUN=TURNS.csv [KEYS=TURNS.csv ...]: RUN's prompt time against REF's, request by request.

The requests are matched as in same_prompts.py: the same (task, turn, prompt tokens read) and the same tokens from
the cache. With KEYS (one or more further runs' tables) only the requests matched in those runs too are used, so the
set is the one a wider table shows. Prints one Markdown row a matched request (REF ms, RUN ms, RUN / REF), then the
totals, how many requests RUN read slower, the same and faster than REF, and the median of the ratios; the same for
the first turns alone, and for the requests with a cached prefix (every request but a session's first, which reads
its whole prompt on cards that have just loaded). CACHE_TOL=N (default 0) matches the cached tokens within N instead of
exactly, as in same_prompts.py; the table then shows REF's count. Standard library only:
python3 -I paired_prompts.py ref=a.csv run=b.csv [keys=c.csv ...]
"""
import csv
import os
import statistics
import sys

TOL = int(os.environ.get("CACHE_TOL", "0"))


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["attempt"] != "1" or not r["prompt_n"]:  # a request without timings (cut by the task's cap)
            continue
        rows[(r["task_id"], int(r["turn"]), int(r["prompt_n"]))] = (int(r["cache_n"]), float(r["prompt_ms"]))
    return rows


args = [a.split("=", 1) for a in sys.argv[1:]]
(rn, rp), (sn, sp) = args[:2]
runs = [load(p) for _, p in args]
common = [k for k in sorted(set.intersection(*(set(r) for r in runs)))
          if max(r[k][0] for r in runs) - min(r[k][0] for r in runs) <= TOL]
ref, run = runs[0], runs[1]
print(f"| task | turn | prompt tokens read | from the cache | {rn} prompt ms | {sn} prompt ms | {sn} / {rn} |")
print("|---|---:|---:|---:|---:|---:|---:|")
for k in common:
    a, b = ref[k][1], run[k][1]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {ref[k][0]} | {a:.1f} | {b:.1f} | {b / a:.4f} |")
print()
print(f"| requests | prompt tokens read | {rn} s | {sn} s | {sn} / {rn} | {sn} slower / same / faster | median of the ratios |")
print("|---|---:|---:|---:|---:|---|---:|")
for label, keys in (("all matched", common), ("first turns", [k for k in common if k[1] == 1]),
                    ("with a cached prefix (all but a session's cold first request)", [k for k in common if ref[k][0] > 0])):
    if not keys:
        continue
    a = sum(ref[k][1] for k in keys)
    b = sum(run[k][1] for k in keys)
    rat = [run[k][1] / ref[k][1] for k in keys]
    up = sum(x > 1 for x in rat)
    eq = sum(x == 1 for x in rat)
    print(f"| {label}, {len(keys)} | {sum(k[2] for k in keys)} | {a / 1000:.2f} | {b / 1000:.2f} | {b / a:.4f} | "
          f"{up} / {eq} / {len(keys) - up - eq} | {statistics.median(rat):.4f} |")
