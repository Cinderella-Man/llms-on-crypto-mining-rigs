#!/usr/bin/env python3
"""rename_arm.py RES OUT OLD NEW: an agent-check run's turns.csv and attempts.csv with the harness's arm renamed.

RES is the client's res/ folder (or any folder with the two files), OUT the folder to write. In turns.csv the `arm`
column, in attempts.csv the `file` column and the `tag` column's first part (`<arm>/<task>/a1/t1`) are renamed from
OLD to NEW, only where they equal OLD; every other value is copied as it is. Standard library only.
"""
import csv
import os
import sys

res, out, old, new = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for name, cols in (("turns.csv", ("arm",)), ("attempts.csv", ("file",))):
    with open(os.path.join(res, name), newline="") as fh:
        rd = csv.DictReader(fh)
        fields, rows = rd.fieldnames, list(rd)
    n = 0
    for r in rows:
        for c in cols:
            if r.get(c) == old:
                r[c] = new
                n += 1
        if name == "attempts.csv" and r.get("tag", "").startswith(old + "/"):
            r["tag"] = new + r["tag"][len(old):]
    with open(os.path.join(out, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows, {n} renamed {old} -> {new}")
