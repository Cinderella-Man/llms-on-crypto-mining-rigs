#!/usr/bin/env python3
"""merge_arms.py OUT NAME=DIR...: several release_check.sh folders, one arm each, as one folder with named arms.

run/agent_flint.sh (docs/data/2026-11-05_release_regression) runs every Flint arm as the harness's arm "rel", each in
its own folder. agent_tables.py (docs/data/2026-11-05_keepalive/run) wants the arms side by side in one folder. Per
NAME=DIR, this takes DIR's res/turns.csv and res/attempts.csv with the arm SRC_ARM (default rel) renamed NAME (the
turns' arm, the attempts' file and the first part of their tag) and DIR/rig/server-SRC_ARM.log as rig/server-NAME.log,
and writes them to OUT/res/turns.csv, OUT/res/attempts.csv and OUT/rig/. Runs anywhere with Python 3 (no packages).
"""
import csv
import os
import shutil
import sys

SRC = os.environ.get("SRC_ARM", "rel")


def rows(path, key, name):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        out = []
        for row in r:
            if row[key] != SRC:
                continue
            row[key] = name
            if row.get("tag", "").startswith(SRC + "/"):
                row["tag"] = name + row["tag"][len(SRC):]
            out.append(row)
        return r.fieldnames, out


def main():
    out, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    os.makedirs(os.path.join(out, "res"), exist_ok=True)
    os.makedirs(os.path.join(out, "rig"), exist_ok=True)
    for fname, key in (("turns.csv", "arm"), ("attempts.csv", "file")):
        cols, allrows = None, []
        for name, d in pairs:
            c, rr = rows(os.path.join(d, "res", fname), key, name)
            cols = cols or c
            allrows += rr
        with open(os.path.join(out, "res", fname), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(allrows)
        print(f"{fname}: {len(allrows)} rows")
    for name, d in pairs:
        log = os.path.join(d, "rig", f"server-{SRC}.log")
        if os.path.exists(log):
            shutil.copyfile(log, os.path.join(out, "rig", f"server-{name}.log"))
        else:
            print(f"no server log for {name}: {log}")


if __name__ == "__main__":
    main()
