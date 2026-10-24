#!/usr/bin/env bash
# full_run.sh OUT [live] [replay]: the full rig run of the agent check (~20 h), one command on the laptop.
# Environment: rig_lib.sh. Both stages by default; each is resumable (finished attempts and replayed traces are
# skipped), so a stopped run continues with the same command.
#   live    all 82 tasks once on each engine with UD-Q4_K_XL, in crossover order so drift over the day falls equally
#           on both: v2 on the first half of the run order (H1, tasks 1-41), v1 on H1, v1 on H2 (42-82), v2 on H2.
#           Every prefix of the run order is balanced by track and size, so the halves are too.
#   replay  8 traces picked from v2's H1 attempts across their range of live minutes, 4 of them (the 1st, 3rd, 5th
#           and 7th shortest) for the variants; in priority order, so the least important come last:
#           v2 (8), v1 (4), v2 at v1's -b 512 -ub 128 (4), v2 not streamed (4), v2 with the reasoning dropped from
#           the history (4), v2 with 6 checkpoints and no minimum spacing (4), the lead file on v2 (8, if MODEL_K2H),
#           stock llama.cpp at v1's flags (4, if STOCK_BIN), v2 with -t 1 (4, unless SKIP_T1=1).
# Then the rig's logs come back to OUT/rig and OUT/table.md, OUT/replay.md and OUT/gonogo.md are written.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: full_run.sh OUT [live] [replay]}")
shift
STAGES=${*:-live replay}
mkdir -p "$OUT"
# shellcheck source=rig_lib.sh
. "$(dirname "$0")/rig_lib.sh"
disk_check || exit 1
trap 'server_stop > /dev/null 2>&1; laptop_stop' EXIT
tunnel_start
proxy_start || exit 1

block() {  # block ARM TASKS: start the arm's server, run the tasks, stop it
  disk_check || exit 1
  server_start "$1" || exit 1
  run_arm "$1" "$2"
  server_stop
}

if [[ " $STAGES " == *" live "* ]]; then
  say "live: v2 H1, v1 H1, v1 H2, v2 H2"
  block v2-ud 1-41
  server_start v1-ud || exit 1
  run_arm v1-ud 1-41
  run_arm v1-ud 42-82
  server_stop
  block v2-ud 42-82
fi

if [[ " $STAGES " == *" replay "* ]]; then
  [ -s "$OUT/replay_traces8.txt" ] || python3 -I "$AC_DIR/replay.py" pick --log "$OUT/proxy.jsonl" \
    --results "$OUT/res" --arm v2-ud --n 8 --tasks 1-41 > "$OUT/replay_traces8.txt"
  T8=$(cat "$OUT/replay_traces8.txt")
  T4=$(echo "$T8" | tr ',' '\n' | sed -n '1p;3p;5p;7p' | paste -sd,)
  say "replay traces: $T8 (variants on $T4)"
  rblock() {  # rblock ARM LABEL TRACES [replay options]
    local arm=$1 label=$2 traces=$3
    shift 3
    disk_check || exit 1
    server_start "$arm" || exit 1
    replay_run "$label" "$traces" "$@"
    server_stop
  }
  server_start v2-ud || exit 1
  replay_run v2-ud "$T8"
  replay_run v2-ud-nostream "$T4" --stream off
  replay_run v2-ud-noreasoning "$T4" --drop-reasoning
  server_stop
  rblock v1-ud v1-ud "$T4"
  rblock v2-ud-v1flags v2-ud-v1flags "$T4"
  rblock v2-ud-ckpt v2-ud-ckpt "$T4"
  [ -n "${MODEL_K2H:-}" ] && rblock v2-k2h v2-k2h "$T8"
  [ -n "${STOCK_BIN:-}" ] && rblock stock-ud stock-ud "$T4"
  [ "${SKIP_T1:-0}" = 1 ] || rblock v2-ud-t1 v2-ud-t1 "$T4"
fi

rig_end
python3 -I "$AC_DIR/report.py" table "$OUT/res" --log "$OUT/proxy.jsonl" > "$OUT/table.md"
[ -s "$OUT/replay.jsonl" ] && python3 -I "$AC_DIR/replay.py" summary "$OUT/replay.jsonl" --log "$OUT/proxy.jsonl" \
  --reference v2-ud > "$OUT/replay.md"
python3 -I "$AC_DIR/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" --meminfo "$OUT/rig/meminfo.log" \
  --server-log "$OUT"/rig/server-*.log > "$OUT/gonogo.md"
cat "$OUT/table.md" "$OUT/replay.md" 2> /dev/null | tee -a "$OUT/run.log"
