#!/usr/bin/env python3
"""merge_named.py DEST SRC:ARM:NAME...: arms of several agent checks in one folder, each under a name of its own.

docs/data/2026-11-05_keepalive/run/agent_tables.py and docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py read one
collected folder (res/turns.csv, res/attempts.csv, rig/server-<arm>.log). Two checks that ran an arm of the same name
(R90's gains and R72's) cannot share it as they are, so each SRC:ARM:NAME copies arm ARM of the collected folder SRC
into DEST as NAME: its rows of res/turns.csv (column arm) and res/attempts.csv (column file, and the arm in front of
column tag), and its server log. The columns are the union of every source's. Standard library only.

    python3 -I merge_named.py merged <R90's folder>:v3:r90v3 <R90's folder>:gains:r90gains <R72's folder>:gains:gains
"""
import csv
import os
import shutil
import sys

dest, specs = sys.argv[1], [s.rsplit(":", 2) for s in sys.argv[2:]]
if not specs or any(len(s) != 3 for s in specs):
    sys.exit(__doc__)
names = [s[2] for s in specs]
if len(set(names)) != len(names):
    sys.exit(f"a name is given twice: {names}")
os.makedirs(os.path.join(dest, "res"), exist_ok=True)
os.makedirs(os.path.join(dest, "rig"), exist_ok=True)
for fname, col in (("turns.csv", "arm"), ("attempts.csv", "file")):
    fields, rows = [], []
    for src, arm, name in specs:
        part = [r for r in csv.DictReader(open(os.path.join(src, "res", fname))) if r[col] == arm]
        if not part:
            sys.exit(f"no rows of {arm} in {src}/res/{fname}")
        for r in part:
            r[col] = name
            if r.get("tag", "").startswith(arm + "/"):
                r["tag"] = name + r["tag"][len(arm):]
        fields += [k for k in part[0] if k not in fields]
        rows += part
        print(f"{fname}: {len(part)} rows of {arm} in {src} as {name}")
    with open(os.path.join(dest, "res", fname), "w", newline="") as f:
        w = csv.DictWriter(f, fields, restval="")
        w.writeheader()
        w.writerows(rows)
for src, arm, name in specs:
    shutil.copyfile(os.path.join(src, "rig", f"server-{arm}.log"), os.path.join(dest, "rig", f"server-{name}.log"))
