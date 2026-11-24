#!/usr/bin/env python3
"""C1004 (Q1737): the three passes' tables (c1004_tables.py's tokens.tsv, profile.tsv, anchors.tsv and waits.tsv) into
one file each, with the pass as the first column.

Usage: python3 -I combine.py <out folder> <pass>=<tables folder> [<pass>=<tables folder> ...]
"""
import os
import sys

OUT, PASSES = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
os.makedirs(OUT, exist_ok=True)
for name in ("tokens.tsv", "profile.tsv", "anchors.tsv", "waits.tsv"):
    rows, header = [], None
    for p, d in PASSES:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        header = header or "pass\t" + lines[0]
        rows += [f"{p}\t{line}" for line in lines[1:]]
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join([header] + rows) + "\n")
    print(f"{name}: {len(rows)} rows")
