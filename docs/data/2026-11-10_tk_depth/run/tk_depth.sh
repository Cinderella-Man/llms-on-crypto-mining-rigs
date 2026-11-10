#!/usr/bin/env bash
# tk_depth.sh: Q873 (T(k) at agent depth), run ON the rig and detached. One llama-bench per arm, k-token passes of the
# main model at three depths of context (-p KS -n 0 -d DEPTHS -r REPS, Q583's line), with the release preset's
# environment (PRESET_ENV from SETTINGS) and the arm's sparse-attention switch:
#   off      GGML_CUDA_FATTN_SPARSE_PASCAL=0  (every attention call on the dense kernels)
#   sparse2  GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5  (the sparse kernel for every call it
#            takes, with no cache-size threshold; NSM=5 is the P106-090's own SM count)
# Each arm starts after a logged quiet check (no llama process, the cards empty, the host's load and memory) and runs
# under ram_watchdog.sh (PROC=llama-bench). llama-bench processes a depth once and restores the saved state for every
# later repetition and k at that depth; if it ever processes a depth a second time (the restore failed) the script
# stops that arm, since every repetition would then read the whole depth again. Output per arm in OUT:
# bench-<arm>.jsonl (one line per test, every repetition's time in samples_ns), bench-<arm>.log (llama-bench's stderr:
# progress and its Markdown table), watchdog-<arm>.tsv (written on /dev/shm while it runs). Ends with "== end" and
# the file DONE in OUT.
#   ssh $RIG "BIN=<build>/bin MODEL=<first shard> OUT=<work dir>/out \
#     setsid nohup bash <work dir>/run/tk_depth.sh > <work dir>/tk_depth.out 2>&1 < /dev/null &"
# Environment, optional: SETTINGS (settings.env next to this script), WATCHDOG (ram_watchdog.sh next to it), TS
# (6,6,6,6,5,6,5,6,3), KS (1,2,3,4,5,6,7,8), DEPTHS (0,8192,32768), REPS (5), ARMS ("off sparse2").
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SETTINGS=${SETTINGS:-$HERE/settings.env}
WATCHDOG=${WATCHDOG:-$HERE/ram_watchdog.sh}
TS=${TS:-6,6,6,6,5,6,5,6,3}
KS=${KS:-1,2,3,4,5,6,7,8}
DEPTHS=${DEPTHS:-0,8192,32768}
REPS=${REPS:-5}
ARMS=${ARMS:-off sparse2}
# shellcheck source=/dev/null
. "$SETTINGS"
: "${PRESET_ENV:?SETTINGS must set PRESET_ENV}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
# the depths llama-bench processes from scratch: each nonzero depth once
NDEPTH=$(tr ',' '\n' <<< "$DEPTHS" | grep -cv '^0$')

quiet() {
  echo "== quiet before $1 $(date -Is): load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(ps -eo comm | grep -c '^llama-'); \
cards $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); \
$(grep -E '^(MemAvailable|CmaFree|SwapFree):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
}

# llama-bench's stderr, line by line, into the log; stops llama-bench if a depth is processed more than once
watch_err() {
  local log=$1 n=0 line
  while IFS= read -r line; do
    printf '%s\n' "$line" >> "$log"
    case $line in
      *"depth run "*"(cached)"*) ;;
      *"depth run "*)
        n=$((n + 1))
        if [ "$n" -gt "$NDEPTH" ]; then
          echo "== $(date -Is) a depth was processed again (state restore failed): stopping llama-bench" | tee -a "$log"
          pkill -x llama-bench
        fi ;;
    esac
  done
}

for arm in $ARMS; do
  case $arm in
    off)     sw="GGML_CUDA_FATTN_SPARSE_PASCAL=0" ;;
    sparse2) sw="GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  pkill -x llama-server; pkill -x llama-bench; sleep 2
  quiet "$arm"
  echo "== arm $arm $(date -Is): env $PRESET_ENV $sw"
  : > "$OUT/bench-$arm.log"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  env $PRESET_ENV $sw "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -b 2048 -ub 512 -ts "${TS//,//}" -lm dio \
    --no-host 1 -p "$KS" -n 0 -d "$DEPTHS" -r "$REPS" --progress -o jsonl -oe md \
    > "$OUT/bench-$arm.jsonl" 2> >(watch_err "$OUT/bench-$arm.log") < /dev/null &
  pid=$!
  PID=$pid PROC=llama-bench LOG="/dev/shm/tk_depth-watchdog-$arm.tsv" setsid nohup chrt -f 10 bash "$WATCHDOG" > /dev/null 2>&1 < /dev/null &
  wait "$pid"
  rc=$?
  sleep 2
  echo "llama-bench $arm: exit $rc, $(( $(date +%s) - t0 )) s, $(wc -l < "$OUT/bench-$arm.jsonl") tests"
  mv "/dev/shm/tk_depth-watchdog-$arm.tsv" "$OUT/watchdog-$arm.tsv"; grep KILL "$OUT/watchdog-$arm.tsv"
  grep -E '^\|' "$OUT/bench-$arm.log"
done
pkill -x llama-bench
quiet end
echo "== end $(date -Is)"
touch "$OUT/DONE"
