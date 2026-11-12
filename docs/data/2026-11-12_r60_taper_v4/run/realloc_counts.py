#!/usr/bin/env python3
"""realloc_counts.py: the target graph's reallocations per request, from llama-server logs (-lv 5).

    python3 -I realloc_counts.py name=server.log [name=server.log ...] > reallocs.tsv

Every request's print_timing block logs its prompt (`prompt eval time = X ms / N tokens`), the graphs reused and the
target context's running count of graph reallocations (`graph reallocs = S / D`: S with the graph's size unchanged,
D changed; both count since the load). The difference to the previous request's count is that request's
reallocations. A reallocation makes the scheduler wait for every card before it moves the buffers, so a layer-split
prompt call loses the overlap of its micro-batches there. Per log and bucket of prompt tokens (R90's buckets: under
512, 512-1,023, 1,024-1,999, 2,000 and more): the requests, their prompt seconds, the reallocations of each kind and
the requests with at least one. Standard library only.
"""
import re, sys

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))
PE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GR = re.compile(r"graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
GU = re.compile(r"graphs reused =\s*([0-9]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


print("log\tprompt_tokens\trequests\tprompt_s\treallocs_same\treallocs_changed\trequests_with_realloc\tgraphs_reused")
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    reqs, cur, prev = [], None, (0, 0)
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = PE.search(line)
        if m:
            cur = dict(ms=float(m.group(1)), n=int(m.group(2)), reused=0, same=0, diff=0)
            continue
        if cur is None:
            continue
        m = GU.search(line)
        if m:
            cur["reused"] = int(m.group(1))
            continue
        m = GR.search(line)
        if m:
            s, d = int(m.group(1)), int(m.group(2))
            cur["same"], cur["diff"] = s - prev[0], d - prev[1]
            prev = (s, d)
            reqs.append(cur)
            cur = None
    for lo, hi in BUCKETS:
        b = [r for r in reqs if lo <= r["n"] < hi]
        if not b:
            continue
        print(f"{name}\t{bname(lo, hi)}\t{len(b)}\t{sum(r['ms'] for r in b) / 1000:.1f}\t{sum(r['same'] for r in b)}\t"
              f"{sum(r['diff'] for r in b)}\t{sum(1 for r in b if r['same'] + r['diff'] > 0)}\t"
              f"{sum(r['reused'] for r in b)}")
