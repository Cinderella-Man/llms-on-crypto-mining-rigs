#!/usr/bin/env bash
# pilot.sh OUT: the rig pilot of the agent check (~1 h 50 min), one command on the laptop. Environment: rig_lib.sh.
#  1. v2 with UD-Q4_K_XL at 262,144 tokens, -b 2048 -ub 512, starts on the rig (monitors and watchdog on).
#  2. Smoke task (task 1 of the run order), then Q132's checkpoint check: the cache path of every turn from the proxy
#     rows and the server log's checkpoint lines, printed before anything else runs. Stops here if the task could not
#     be scored.
#  3. Tasks 1-6 on v2 (task 1 is already done).
#  4. Replays on v2 of the two shortest v2 traces among tasks 1-3: streamed (as pi sends them) and not streamed.
#  5. v1 as released (-b 512 -ub 128) replaces v2; tasks 1-3 on v1; the same two traces replayed on v1.
#  6. Everything on the rig stops, its logs come back to OUT/rig, and the go/no-go list is written to OUT/gonogo.md.
# Each attempt is capped at PILOT_TIME_CAP_MIN minutes (default 20) and 40 turns. Resumable: run it again with the
# same OUT and finished attempts and replayed traces are skipped.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: pilot.sh OUT}")
mkdir -p "$OUT"
# shellcheck source=rig_lib.sh
. "$(dirname "$0")/rig_lib.sh"
disk_check || exit 1
trap 'server_stop > /dev/null 2>&1; laptop_stop' EXIT
tunnel_start
proxy_start || exit 1

say "pilot: step 1, v2"
server_start v2-ud || exit 1
say "pilot: step 2, smoke task and the checkpoint check (Q132)"
CAP=(--time-cap-min "${PILOT_TIME_CAP_MIN:-20}")
run_arm v2-ud 1 "${CAP[@]}"
python3 -I "$AC_DIR/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" > "$OUT/smoke.md"
ssh "$RIG" "grep -iE 'checkpoint|re-processing|backend sampling' $RIG_LOGDIR/server-v2-ud.log | tail -40" \
  > "$OUT/smoke_server_lines.txt"
cat "$OUT/smoke.md" | tee -a "$OUT/run.log"
say "server lines on checkpoints and sampling: $(wc -l < "$OUT/smoke_server_lines.txt") (in smoke_server_lines.txt)"
if ! ls "$OUT"/res/v2-ud/*/a1/result.json > /dev/null 2>&1 || grep -q infra_exhausted "$OUT"/res/v2-ud/*/a1/result.json; then
  say "smoke task not scored; stopping the pilot"
  rig_end
  exit 1
fi
say "pilot: step 3, tasks 1-6 on v2"
run_arm v2-ud 6 "${CAP[@]}"
TRACES=$(python3 -I "$AC_DIR/replay.py" pick --log "$OUT/proxy.jsonl" --results "$OUT/res" --arm v2-ud --n 2 \
  --tasks 1-3 --shortest)
echo "$TRACES" > "$OUT/replay_traces.txt"
say "pilot: step 4, replays of $TRACES on v2"
replay_run v2-ud "$TRACES"
replay_run v2-ud-nostream "$TRACES" --stream off
server_stop
say "pilot: step 5, v1"
server_start v1-ud || exit 1
run_arm v1-ud 3 "${CAP[@]}"
replay_run v1-ud "$TRACES"
server_stop
say "pilot: step 6, logs and go/no-go"
rig_end
python3 -I "$AC_DIR/report.py" table "$OUT/res" --log "$OUT/proxy.jsonl" > "$OUT/table.md"
python3 -I "$AC_DIR/replay.py" summary "$OUT/replay.jsonl" --log "$OUT/proxy.jsonl" --reference v2-ud > "$OUT/replay.md"
python3 -I "$AC_DIR/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" --replay "$OUT/replay.jsonl" \
  --replay-label v2-ud --meminfo "$OUT/rig/meminfo.log" --server-log "$OUT"/rig/server-*.log \
  --expect-min v2-ud=3.9,v1-ud=5.5 > "$OUT/gonogo.md"
cat "$OUT/table.md" "$OUT/replay.md" "$OUT/gonogo.md" | tee -a "$OUT/run.log"
