#!/usr/bin/env bash
# verify_cost.sh: Q583's three loads on one rig, run ON the rig and detached, each under mem_guard.sh (Q558) and each
# after a logged quiet check (no llama process, the cards empty, the host's load and memory):
#   draft   llama-server with S15's MTP draft line and GGML_CUDA_PROFILE, one drafting request (REQUEST, a chat
#           completion body; the reply's timings carry draft_n and draft_n_accepted), the server's log and profile;
#   prof    llama-bench, k-token passes of the main model alone (-p KS -n 0, REPS repetitions after one warm-up),
#           with GGML_CUDA_PROFILE: where a k-token pass's GPU time goes;
#   wall    the same llama-bench without the profiler: the wall time of a k-token pass.
# llama-bench's passes ask for the last token's logits only; a verify needs all k (the output layer over k rows),
# which the draft load's own passes show. Ends with "== end" and the file DONE in OUT.
#   ssh $RIG "BIN=<build>/bin MODEL=<first shard> DRAFT=<draft gguf> OUT=<work dir> \
#     setsid nohup bash verify_cost.sh > <work dir>/verify_cost.out 2>&1 < /dev/null &"
# mem_guard.sh and smoke_request.json are the ones in ../../2026-11-03_flash_next_mtp/run/ (pass GUARD and REQUEST
# when this folder is copied elsewhere). Environment, optional: TS (S15's split), KS (1,2,3,4,5), REPS (5), PORT (8096),
# STEPS ("draft prof wall").
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
GUARD=${GUARD:-$HERE/mem_guard.sh}
REQUEST=${REQUEST:-$HERE/smoke_request.json}
TS=${TS:-6,6,6,6,5,6,5,6,3}
KS=${KS:-1,2,3,4,5}
REPS=${REPS:-5}
PORT=${PORT:-8096}
STEPS=${STEPS:-draft prof wall}
mkdir -p "$OUT"
rm -f "$OUT/DONE"

quiet() {
  echo "== quiet before $1 $(date -Is): load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); \
cards $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); \
$(grep -E '^(MemAvailable|CmaFree|Shmem|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
}
guard_on() { LOG=$OUT/mem_guard-$1.log PROC=$2 setsid nohup chrt -f 10 bash "$GUARD" > /dev/null 2>&1 < /dev/null & sleep 1; }
guard_off() { kill "$(cat "$OUT/mem_guard-$1.log.pid")" 2> /dev/null; grep KILL "$OUT/mem_guard-$1.log"; }

for step in $STEPS; do
  pkill -x llama-server; pkill -x llama-bench; sleep 2
  quiet "$step"
  case $step in
  draft)
    guard_on draft llama-server
    t0=$(date +%s)
    GGML_CUDA_PROFILE=$OUT/prof-draft.csv "$BIN/llama-server" -ngl 99 -fa on -c 65536 -np 1 -bs \
      --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 \
      --min-p 0 --metrics --no-webui -b 2048 -ub 512 -ts "$TS" -fit off -lm dio \
      -md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 \
      --no-host -m "$MODEL" --host 127.0.0.1 --port "$PORT" > "$OUT/server-draft.log" 2>&1 < /dev/null &
    srv=$!
    for i in $(seq 1 450); do
      curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2
    done
    echo "load $(( $(date +%s) - t0 )) s; cards $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')"
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
      # the profile's evaluations before the request are the warm-up's: mark where the request starts
      echo "request starts after $(( $(wc -l < "$OUT/prof-draft.csv") - 1 )) profile lines, $(date -Is)"
      curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d @"$REQUEST" > "$OUT/reply-draft.json"
      echo "request done $(date -Is), $(wc -c < "$OUT/reply-draft.json") bytes, $(( $(wc -l < "$OUT/prof-draft.csv") - 1 )) profile lines"
      curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-draft.txt"
    else
      echo "server did not start"; tail -20 "$OUT/server-draft.log"
    fi
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server
    guard_off draft
    ;;
  prof|wall)
    guard_on "$step" llama-bench
    envs=()
    [ "$step" = prof ] && envs=(GGML_CUDA_PROFILE="$OUT/prof-bench.csv")
    t0=$(date +%s)
    env "${envs[@]}" "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -b 2048 -ub 512 -ts "${TS//,//}" -lm dio \
      --no-host 1 -p "$KS" -n 0 -r "$REPS" -o json -oe md > "$OUT/bench-$step.json" 2> "$OUT/bench-$step.log" < /dev/null
    echo "llama-bench $step: exit $?, $(( $(date +%s) - t0 )) s"
    grep -E '^\|' "$OUT/bench-$step.log"
    guard_off "$step"
    ;;
  esac
done
quiet end
echo "== end $(date -Is)"
touch "$OUT/DONE"
