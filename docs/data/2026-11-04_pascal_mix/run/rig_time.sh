#!/usr/bin/env bash
# rig_time.sh: a Pascal mix against the file it was made from, on one rig, the same build and the same boot. Runs ON
# the rig, detached; each step after a logged quiet check (no llama process, the cards empty, host load and memory)
# and under mem_guard.sh. Per model, in the order of MODELS:
#   fit   llama-server loads the model at the 64k line (CTX, UB, TS; -fa on -fit off -lm dio --no-host, no draft),
#         answers one short request (REQUEST), and the cards' memory and the host's MemAvailable are logged once the
#         server is healthy: the fit probe on the real weights;
#   wall  llama-bench -p KS -n 0 -ts WALL_TS, REPS repetitions after a warm-up (../../2026-11-03_verify_cost/run/
#         verify_cost.sh, STEPS=wall): the one-token pass and the k-curve, T(1) ... T(8), on the split of the
#         2026-11-04_kcurve_p106 table.
# Then ../../2026-11-04_kcurve_p106/run/tk.py's table of both (rig_time.tsv) and the file DONE in OUT.
#   ssh $RIG "BIN=<build>/bin MODELS='<label>=<first shard> <label>=<first shard>' OUT=<work dir> \
#     setsid nohup bash rig_time.sh > <work dir>/rig_time.out 2>&1 < /dev/null &"
# Needs beside it (or in GUARD, REQUEST, VERIFY, TK): mem_guard.sh and smoke_request.json
# (../../2026-11-03_flash_next_mtp/run/), verify_cost.sh, tk.py. Optional: CTX (65536), UB (512), TS
# (6,5,5,6,5,5,6,5,6), WALL_TS (6,6,6,6,5,6,5,6,3), KS (1,2,3,4,5,6,7,8), REPS (5), PORT (8096), STEPS ("fit wall").
set -uo pipefail
: "${BIN:?}" "${MODELS:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
GUARD=${GUARD:-$HERE/mem_guard.sh} REQUEST=${REQUEST:-$HERE/smoke_request.json}
VERIFY=${VERIFY:-$HERE/verify_cost.sh} TK=${TK:-$HERE/tk.py}
CTX=${CTX:-65536} UB=${UB:-512} TS=${TS:-6,5,5,6,5,5,6,5,6} WALL_TS=${WALL_TS:-6,6,6,6,5,6,5,6,3}
KS=${KS:-1,2,3,4,5,6,7,8} REPS=${REPS:-5}
PORT=${PORT:-8096} STEPS=${STEPS:-fit wall}
mkdir -p "$OUT"
rm -f "$OUT/DONE"

quiet() {
  echo "== quiet before $1 $(date -Is): load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); \
cards $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); \
$(grep -E '^(MemAvailable|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
}

tk=()
for spec in $MODELS; do
  label=${spec%%=*} model=${spec#*=}
  for step in $STEPS; do
    pkill -x llama-server; pkill -x llama-bench; sleep 2
    case $step in
    fit)
      quiet "$label fit"
      LOG=$OUT/mem_guard-$label-fit.log PROC=llama-server setsid nohup chrt -f 10 bash "$GUARD" > /dev/null 2>&1 < /dev/null &
      sleep 1
      t0=$(date +%s)
      "$BIN/llama-server" -m "$model" -ngl 99 -fa on -c "$CTX" -np 1 -b 2048 -ub "$UB" -ts "$TS" -fit off -lm dio \
        --no-host --ctx-checkpoints 4 --cache-ram 0 --jinja --no-webui --host 127.0.0.1 --port "$PORT" \
        > "$OUT/server-$label.log" 2>&1 < /dev/null &
      srv=$!
      for i in $(seq 1 450); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2
      done
      if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        echo "$label: loaded in $(( $(date +%s) - t0 )) s at -c $CTX -ub $UB -ts $TS"
        curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
          -d @"$REQUEST" > "$OUT/reply-$label.json"
        echo "$label: request $(wc -c < "$OUT/reply-$label.json") bytes; cards (index, used, total MiB) \
$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';') \
$(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"
      else
        echo "$label: FAILED to load (server log: $OUT/server-$label.log)"; tail -n 5 "$OUT/server-$label.log"
      fi
      kill $srv 2> /dev/null; wait $srv 2> /dev/null
      kill "$(cat "$OUT/mem_guard-$label-fit.log.pid")" 2> /dev/null; grep KILL "$OUT/mem_guard-$label-fit.log"
      ;;
    wall)
      # verify_cost.sh logs its own quiet check
      BIN=$BIN MODEL=$model DRAFT=$model OUT=$OUT/$label STEPS=wall KS=$KS REPS=$REPS TS=$WALL_TS GUARD=$GUARD \
        REQUEST=$REQUEST bash "$VERIFY"
      tk+=("$label=$OUT/$label/bench-wall.json")
      ;;
    esac
  done
done
pkill -x llama-server; pkill -x llama-bench
[ ${#tk[@]} -gt 0 ] && python3 -I "$TK" "${tk[@]}" > "$OUT/rig_time.tsv"
echo "== end $(date -Is)"
date -Is > "$OUT/DONE"
