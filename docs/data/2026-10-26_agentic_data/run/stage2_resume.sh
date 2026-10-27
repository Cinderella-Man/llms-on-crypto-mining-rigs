#!/usr/bin/env bash
# Run 1, stage 2 resumed after its tasks gate stopped run.sh (2026-10-27 ~02:00). tasks.py from-sessions (59777e6)
# built tasks on every stage-1 session whose tests passed, also on sessions the sessions gate excluded, and the tasks
# gate failed those rows (vb: 219 of 687, from 20 parents); run.sh stops on a failed gate. This keeps the rows the gate
# passed (the full file stays as tasks.all.jsonl, the failed gate record as tasks.gate.all.*), then runs run.sh with
# STEPS="s2run gate export checks", which gates the cut file again before any session. Fixed in e1176b5 (tasks.py
# --gate, step s2run), so later runs do not need it.
#   RUN1=<folder with run1-<writer>.env and the run folders> bash stage2_resume.sh q8|va|vb
#   (after that writer's run.sh has stopped; the env file sets WORK and SRC, the bench/agentic_data folder used)
set -euo pipefail
A=${RUN1:?folder with run1-<writer>.env}
W=$1
. "$A/run1-$W.env"
T=$WORK/s2/tasks
for p in $(pgrep -f agentic_data/run.sh || true); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "WORK=$WORK"; then echo "run.sh still runs for $W"; exit 1; fi
done
grep -q "^FAIL: tasks" <(tail -5 "$WORK/run.log") || { echo "$W: the last step is not a failed tasks gate"; exit 1; }
[ -f "$T/tasks.all.jsonl" ] || cp "$T/tasks.jsonl" "$T/tasks.all.jsonl"
for x in json tsv; do [ -f "$WORK/s2/tasks.gate.all.$x" ] || cp "$WORK/s2/tasks.gate.$x" "$WORK/s2/tasks.gate.all.$x"; done
python3 -I - "$T/tasks.all.jsonl" "$WORK/s2/tasks.gate.all.tsv" "$T/tasks.jsonl" <<'EOF'
import json, sys
src, tsv, out = sys.argv[1:]
lines = open(tsv).read().splitlines()
head = lines[0].split("\t")
ok = {r["task_id"] for r in (dict(zip(head, x.split("\t"))) for x in lines[1:]) if r["check"] == "ok"}
rows = open(src).readlines()
kept = [r for r in rows if json.loads(r)["task_id"] in ok]
open(out, "w").writelines(kept)
print(f"kept {len(kept)} of {len(rows)} stage-2 tasks (the rows the tasks gate passed)")
EOF
export STEPS="s2run gate export checks"
echo "$(date +%T) resumed: STEPS=$STEPS (stage2_resume.sh)" | tee -a "$WORK/run.log" >> "$WORK/run.out"
nohup setsid bash "$SRC/run.sh" >> "$WORK/run.out" 2>&1 < /dev/null &
echo "resumed $W"
