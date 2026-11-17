#!/usr/bin/env bash
# rig_kq_ab.sh: C861 (a) on one P106 rig, run ON the rig and detached. GGML_CUDA_MMQ_KQ_FP32SCALE (the K-quant prompt
# kernels' scale multiply in FP32) off and on, on the one release file with K-quant tensors (Flash-Next's v4 file,
# UkisAI's Swift-1.5 GSQ-RCO Q2_0), with the v4 preset's environment (PRESET_ENV from SETTINGS). Arms:
#   off  GGML_CUDA_MMQ_KQ_FP32SCALE=0
#   on   GGML_CUDA_MMQ_KQ_FP32SCALE=1                                          (Q4_K, Q5_K, Q6_K: the default types)
#   all  GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all     (Q3_K too)
# Steps, each arm in turn, every process after a logged quiet check and under ram_watchdog.sh:
#   1. test-backend-ops -o MUL_MAT,MUL_MAT_ID -b CUDA0 -p 'type_a=q[3-6]_K' (card 0; cases of 9 tokens and more take
#      the prompt kernels): every case must pass in every arm. tbo-<arm>.log
#   2. llama-bench, prompts of 64, 128 and 512 tokens at micro-batches of 64 (the preset's taper for prompts of up to
#      600 tokens) and 512, -n 0 -r REPS, ROUNDS times in the order off on all. bench-<arm>-<round>.jsonl and .log
#   3. llama-completion, one greedy reply of 128 tokens to PROMPT_FILE (~600 tokens, read through the prompt kernels):
#      the same bytes in every arm, since the FP32 form is exact. greedy-<arm>.out and .err; greedy.tsv (arm, md5)
# Ends with "== end" and the file DONE in OUT.
#   ssh $RIG "BIN=<build>/bin MODEL=<first shard> OUT=<work dir>/out \
#     setsid nohup bash <work dir>/run/rig_kq_ab.sh > <work dir>/rig_kq_ab.out 2>&1 < /dev/null &"
# Environment, optional: SETTINGS (settings.env next to this script: release/flash-next-v4/settings.env), WATCHDOG
# (ram_watchdog.sh next to it: scripts/ram_watchdog.sh), PROMPT_FILE (prompt.txt next to it), TS (6,6,6,6,5,6,5,6,3),
# REPS (5), ROUNDS (2), ARMS ("off on all"), STEPS ("tbo bench greedy").
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SETTINGS=${SETTINGS:-$HERE/settings.env}
WATCHDOG=${WATCHDOG:-$HERE/ram_watchdog.sh}
PROMPT_FILE=${PROMPT_FILE:-$HERE/prompt.txt}
TS=${TS:-6,6,6,6,5,6,5,6,3}
REPS=${REPS:-5}
ROUNDS=${ROUNDS:-2}
ARMS=${ARMS:-off on all}
STEPS=${STEPS:-tbo bench greedy}
# shellcheck source=/dev/null
. "$SETTINGS"
: "${PRESET_ENV:?SETTINGS must set PRESET_ENV}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"

quiet() {
  echo "== quiet before $1 $(date -Is): load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(ps -eo comm | grep -c '^llama-\|^test-backend'); \
cards $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); \
$(grep -E '^(MemAvailable|CmaFree|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
}

arm_env() {
  case $1 in
    off) echo "GGML_CUDA_MMQ_KQ_FP32SCALE=0" ;;
    on)  echo "GGML_CUDA_MMQ_KQ_FP32SCALE=1" ;;
    all) echo "GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all" ;;
    *)   return 1 ;;
  esac
}

# run TAG PROC STDOUT STDERR cmd...: one process in the background under the watchdog, its output to the two files;
# waits for it
run() {
  local tag=$1 proc=$2 out=$3 log=$4; shift 4
  pkill -x llama-server; pkill -x llama-bench; pkill -x llama-completion; sleep 2
  quiet "$tag"
  local t0; t0=$(date +%s)
  "$@" > "$out" 2> "$log" < /dev/null &
  local pid=$!
  PID=$pid PROC=$proc LOG="/dev/shm/c861a-watchdog-$tag.tsv" setsid nohup chrt -f 10 bash "$WATCHDOG" > /dev/null 2>&1 < /dev/null &
  wait "$pid"
  local rc=$?
  sleep 2
  mv "/dev/shm/c861a-watchdog-$tag.tsv" "$OUT/watchdog-$tag.tsv" 2> /dev/null
  grep KILL "$OUT/watchdog-$tag.tsv" 2> /dev/null
  echo "$tag: exit $rc, $(( $(date +%s) - t0 )) s"
}

for step in $STEPS; do
  case $step in
  tbo)
    for arm in $ARMS; do
      sw=$(arm_env "$arm") || { echo "unknown arm $arm"; continue; }
      echo "== tbo $arm $(date -Is): env $sw"
      # shellcheck disable=SC2086
      run "tbo-$arm" test-backend-ops "$OUT/tbo-$arm.log" "$OUT/tbo-$arm.err" \
        env $sw "$BIN/test-backend-ops" -o MUL_MAT,MUL_MAT_ID -b CUDA0 -p 'type_a=q[3-6]_K'
      grep -E 'tests passed|FAIL' "$OUT/tbo-$arm.log" | tail -n 5
    done ;;
  bench)
    for r in $(seq 1 "$ROUNDS"); do
      for arm in $ARMS; do
        sw=$(arm_env "$arm") || continue
        echo "== bench $arm round $r $(date -Is): env $PRESET_ENV $sw"
        # shellcheck disable=SC2086
        run "bench-$arm-$r" llama-bench "$OUT/bench-$arm-$r.jsonl" "$OUT/bench-$arm-$r.log" \
          env $PRESET_ENV $sw "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -b 4096 -ub 64,512 -ts "${TS//,//}" -lm dio \
            --no-host 1 -p 64,128,512 -n 0 -r "$REPS" --progress -o jsonl -oe md
        grep -E '^\|' "$OUT/bench-$arm-$r.log"
      done
    done ;;
  greedy)
    printf 'arm\tmd5\tbytes\tprompt_eval\n' > "$OUT/greedy.tsv"
    for arm in $ARMS; do
      sw=$(arm_env "$arm") || continue
      echo "== greedy $arm $(date -Is): env $PRESET_ENV $sw"
      # shellcheck disable=SC2086
      run "greedy-$arm" llama-completion "$OUT/greedy-$arm.out" "$OUT/greedy-$arm.err" \
        env $PRESET_ENV $sw "$BIN/llama-completion" -m "$MODEL" -ngl 99 -fa on -ts "$TS" -lm dio --no-host \
          -c 4096 -b 4096 -ub 512 -n 128 --temp 0 --seed 1 -no-cnv --no-display-prompt -f "$PROMPT_FILE"
      printf '%s\t%s\t%s\t%s\n' "$arm" "$(md5sum < "$OUT/greedy-$arm.out" | cut -d' ' -f1)" "$(wc -c < "$OUT/greedy-$arm.out")" \
        "$(grep -m1 'prompt eval time' "$OUT/greedy-$arm.err" | sed 's/^.*prompt eval time *= *//')" >> "$OUT/greedy.tsv"
    done
    cat "$OUT/greedy.tsv"
    if [ "$(tail -n +2 "$OUT/greedy.tsv" | cut -f2 | sort -u | wc -l)" = 1 ]; then echo "greedy: SAME"; else echo "greedy: DIFFERS"; fi ;;
  esac
done
pkill -x llama-bench
quiet end
echo "== end $(date -Is)"
touch "$OUT/DONE"
