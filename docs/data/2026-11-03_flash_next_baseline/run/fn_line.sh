#!/usr/bin/env bash
# One server line of Q215's single-request part, run ON the rig and detached, so that a dropped ssh connection
# cannot stop it: llama-server with the line's build and settings, its load time, memory per card and on the host,
# the load log's buffer and feature lines, then fn_requests.py (beside this script) against it on 127.0.0.1, the
# context-checkpoint lines from the server log (restores carry their own ms), and, with GGML_CUDA_PROFILE in ENVS,
# the kernel profile copied next to the output. Ends with "== end <label>".
#   ssh $RIG "LABEL=v3 BUILD=<build folder> BUILDS=<builds> MODEL=<first shard> CTX=65536 \
#     ARGS='-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off' MODE=speed OUT=<work dir> \
#     setsid nohup bash fn_line.sh > <work dir>/line-v3.out 2>&1 < /dev/null &"
set -uo pipefail
: "${LABEL:?}" "${BUILD:?}" "${BUILDS:?}" "${MODEL:?}" "${CTX:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8095}
LOG=$OUT/server-$LABEL.log
echo "== $LABEL: $BUILD -c $CTX ${ARGS:-} [env ${ENVS:-}] mode ${MODE:-speed} $(date -Is)"
pkill -x llama-server; sleep 2
env ${ENVS:-} $BUILDS/$BUILD/bin/llama-server -m "$MODEL" -ngl 99 -fa on -c "$CTX" ${ARGS:-} -np 1 \
  --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
srv=$!
t0=$(date +%s)
for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
echo "load $(( $(date +%s) - t0 )) s"
grep -E 'model buffer size|KV buffer size|RS buffer size|compute buffer size|n_ctx *=|lazy|ple|checkpoints enabled|pipeline|graph splits|out of memory|failed|error' "$LOG" \
  | grep -v 'unused tensor' | cut -c1-220 | head -60
echo "cards: $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tr '\n' ';')"
free -m | sed -n 2p
if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
  python3 -I "$HERE/fn_requests.py" "$PORT" "${MODE:-speed}" "${SSD_DEV:-sda}" "${USB_DEV:-sdb}"
  grep -E 'context checkpoint' "$LOG" | cut -c1-240 | tail -40
  echo "cards after: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')"
else
  echo "server did not start"; tail -20 "$LOG"
fi
kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server
for e in ${ENVS:-}; do
  case $e in GGML_CUDA_PROFILE=*) p=${e#GGML_CUDA_PROFILE=}; [ -f "$p" ] && mv "$p" "$OUT/prof-$LABEL.csv" ;; esac
done
echo "== end $LABEL $(date -Is)"
