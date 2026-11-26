#!/usr/bin/env python3
"""replies.py: the generation requests' replies of deep-lane loads side by side, so loads of one file can be checked
for the same replies (C1139).

  python3 -I replies.py NAME=requests.csv [NAME=requests.csv ...] > replies.md

Each requests CSV is one load's (bench/deep_lane, deep_client.py): one row per request. For every decode and sample
request (by depth and pass) the table shows the first 8 hex digits of each load's ids_md5 (the reply's token ids; the
text's md5 where the server returned no ids), and whether all loads given agree. Standard library only."""
import csv
import sys

loads = []
for a in sys.argv[1:]:
    name, _, path = a.partition("=")
    with open(path, newline="") as f:
        loads.append((name, {(r["depth"], r["pass"]): (r["ids_md5"] or r["text_md5"]) for r in csv.DictReader(f)
                             if r["kind"] in ("decode", "sample")}))
keys = []
for _, d in loads:
    for k in d:
        if k not in keys:
            keys.append(k)
keys.sort(key=lambda k: (int(k[0]), k[1]))
print("| depth | pass | " + " | ".join(f"`{n}`" for n, _ in loads) + " | same |")
print("|---:|---|" + "---|" * len(loads) + "---|")
for k in keys:
    vals = [d.get(k, "") for _, d in loads]
    same = "yes" if len(set(vals)) == 1 and vals[0] else "no"
    print(f"| {k[0]} | {k[1]} | " + " | ".join(v[:8] or "-" for v in vals) + f" | {same} |")
