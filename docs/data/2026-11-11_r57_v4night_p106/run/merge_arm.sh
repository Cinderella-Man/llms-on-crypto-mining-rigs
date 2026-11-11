#!/usr/bin/env bash
# merge_arm.sh: one folder for docs/data/2026-11-05_keepalive/run/agent_tables.py with this run's arms and one arm of an
# earlier agent check, so the tables put a second run of the reference beside this one. R57 adds v3.0's t1swift
# (2026-11-06: the same rig, file, preset and build as arm v3rel; docs/data/2026-11-05_release_flashnext). Runs where
# both collected folders are, with Python 3 (no packages).
# Env: NOW (this run's collected folder: res/turns.csv, res/attempts.csv, rig/server-<arm>.log), OLD (the earlier run's),
#   OLD_ARM (t1swift), DEST (a new folder; it gets res/ and rig/ with the rows and logs of every arm of NOW plus OLD_ARM)
set -euo pipefail
: "${NOW:?}" "${OLD:?}" "${DEST:?}"
OLD_ARM=${OLD_ARM:-t1swift}
mkdir -p "$DEST/res" "$DEST/rig"
# turns.csv names the arm in column arm, attempts.csv in column file; the columns are the union of both files'
python3 -I - "$NOW" "$OLD" "$OLD_ARM" "$DEST" << 'EOF'
import csv, os, sys
now, old, arm, dest = sys.argv[1:]
for name, col in (("turns.csv", "arm"), ("attempts.csv", "file")):
    a = list(csv.DictReader(open(os.path.join(now, "res", name))))
    b = [r for r in csv.DictReader(open(os.path.join(old, "res", name))) if r[col] == arm]
    if not a or not b:
        sys.exit(f"no rows of {now} or of {arm} in {old}/res/{name}")
    if any(r[col] == arm for r in a):
        sys.exit(f"{arm} is already in {now}/res/{name}")
    fields = list(a[0].keys()) + [k for k in b[0].keys() if k not in a[0]]
    with open(os.path.join(dest, "res", name), "w", newline="") as f:
        w = csv.DictWriter(f, fields, restval="")
        w.writeheader()
        w.writerows(a + b)
    print(f"{name}: {len(a)} rows of {now}, {len(b)} of {arm}")
EOF
cp "$NOW"/rig/server-*.log "$DEST/rig/"
cp "$OLD/rig/server-$OLD_ARM.log" "$DEST/rig/"
