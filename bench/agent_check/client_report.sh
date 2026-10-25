#!/usr/bin/env bash
# client_report.sh OUT ARMS: release_check.sh's reports, written into OUT (the client's results folder): compare.md
# (arms side by side, ARMS = arm,arm,… with the reference first; per-request rows in res/turns.csv), table.md
# (res/attempts.csv) and gonogo.md (with the rig's meminfo.log and server logs when they are in OUT/rig). Runs in the
# sandbox image (release_check.sh starts it there), or anywhere with Python 3.
set -uo pipefail
OUT=${1:?usage: client_report.sh OUT ARMS}
ARMS=${2:?usage: client_report.sh OUT ARMS}
R=$(cd "$(dirname "$0")" && pwd)/report.py
cd "$OUT" || exit 1
python3 -I "$R" compare res --log proxy.jsonl --arms "$ARMS" > compare.md || exit 1
python3 -I "$R" table res --log proxy.jsonl --csv res/attempts.csv > table.md || exit 1
GONOGO=(--log proxy.jsonl --rtt 0.3)
[ -f rig/meminfo.log ] && GONOGO+=(--meminfo rig/meminfo.log)
ls rig/server-*.log > /dev/null 2>&1 && GONOGO+=(--server-log rig/server-*.log)
python3 -I "$R" gonogo res "${GONOGO[@]}" > gonogo.md || true
cat compare.md
