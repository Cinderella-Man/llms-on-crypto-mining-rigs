#!/usr/bin/env python3
"""r225_plans.py: did the server read each pass's calls as its taper setting says? Per pass and call size, the
calls the server logged ("ubatch taper:" lines, matched to requests in order as c222_tables.py does) and the largest
micro-batch among them. Under short=N:64 a call of 601 to N tokens should read in 64s (largest 67: the taper merges a
piece under its 9-token floor into its neighbour) and a longer one in 512s; the classes split at 1,023 and 1,400.

    python3 -I r225_plans.py --load r225/load-A > plans_by_limit.tsv

Every request is counted, the warm-up pass and the unmeasured prefix requests ("reset") included (column kind).
Standard library only.
"""
import argparse, collections, csv, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True, help="the load folder (requests.tsv and server.log)")
a = ap.parse_args()
reqs = list(csv.DictReader(open(os.path.join(a.load, "requests.tsv")), delimiter="\t"))
ansi = re.compile(r"\x1b\[[0-9;]*m")
pend, k = [], 0
for line in open(os.path.join(a.load, "server.log"), errors="replace"):
    line = ansi.sub("", line)
    m = re.search(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$", line)
    if m:
        pend.append((int(m.group(1)), m.group(5).strip()))
        continue
    if "prompt eval time =" in line:
        if k < len(reqs):
            reqs[k]["calls"] = pend
        pend, k = [], k + 1
if k != len(reqs):
    print(f"# warning: {k} timing lines in the server log, {len(reqs)} requests")


def sizes(plan):
    out = []
    for t in plan.split():
        n, _, c = t.partition("x")
        out += [int(n)] * (int(c) if c else 1)
    return out


CLASSES = ((9, 600), (601, 1023), (1024, 1400), (1401, 2048), (2049, 10 ** 9))
res = collections.defaultdict(lambda: [0, 0])
for r in reqs:
    for n, pl in r.get("calls", []):
        lo, hi = next(c for c in CLASSES if c[0] <= n <= c[1]) if n >= 9 else (0, 8)
        e = res[(int(r["pass"]), r["arm"], r["kind"], lo, hi)]
        e[0] += 1
        e[1] = max(e[1], max(sizes(pl)))
print("pass\tsetting\tkind\tcall_tokens\tcalls_logged\tlargest_ubatch")
for (p, arm, kind, lo, hi), (n, mx) in sorted(res.items()):
    print(f"{p}\t{arm}\t{kind}\t{lo}-{hi if hi < 10 ** 9 else 'up'}\t{n}\t{mx}")
