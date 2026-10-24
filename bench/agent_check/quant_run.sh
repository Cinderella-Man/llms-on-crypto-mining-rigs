#!/usr/bin/env bash
# quant_run.sh OUT: the agent check on a GPU machine with Docker (the quant machine): one llama-server with NP slots,
# the task set ATTEMPTS times, pi in one container per attempt. Its first use is the calibration (UD-Q4_K_XL, two
# passes: the defaults), which sizes pass rates, turns and tokens per task before any rig time; the same script runs
# the file comparison (ARM=q-k2h MODEL=… ATTEMPTS=1-10, and the same for UD). Quality and token counts only: the
# proxy log carries llama-server's timings, but times on a shared multi-GPU machine are never speed results.
#
# Environment (paths as seen inside the server container unless noted):
#   MODEL         the GGUF (UD-Q4_K_XL)                         LLAMA_SERVER  the llama-server binary
#   MOUNTS        docker -v arguments that make both visible     GPUS          device list (default 0,1)
#   PORT          llama-server's port on 127.0.0.1 (required)    PROXY_PORT    the proxy's port (default PORT+1)
#   NP            server slots = attempts in flight (default 6)  CTX           total context, -kvu (default 262144)
#   ATTEMPTS      default 1-2                                    TASKS         default all
#   ARM           the arm's name in the results (default calib-ud)
#   SERVER_IMAGE  default nvidia/cuda:12.9.2-devel-ubuntu24.04   IMAGE         sandbox image (default agent-check:0.87.1)
#   LABEL         label on every container (default pascal-quants=1)
# Steps: build the sandbox image (docker/Dockerfile, cached), validate every task in it (stops here if one fails),
# start the server, the proxy and run.py, then write OUT/table.md and OUT/attempts.csv. Resumable: running it again
# with the same OUT skips finished attempts. Every container is removed at the end, also on failure.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: quant_run.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?}" "${LLAMA_SERVER:?}" "${PORT:?}"
GPUS=${GPUS:-0,1}; PROXY_PORT=${PROXY_PORT:-$((PORT + 1))}; NP=${NP:-6}; CTX=${CTX:-262144}
ATTEMPTS=${ATTEMPTS:-1-2}; TASKS=${TASKS:-all}; MOUNTS=${MOUNTS:-}; ARM=${ARM:-calib-ud}
SERVER_IMAGE=${SERVER_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
IMAGE=${IMAGE:-agent-check:0.87.1}; LABEL=${LABEL:-pascal-quants=1}
NAME=accal-srv-${GPUS//,/}
PREFIX=accal-$$
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/quant_run.log"; }

log "building $IMAGE"
docker build -q --label "$LABEL" -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE" >> "$OUT/quant_run.log" 2>&1 \
  || { log "image build failed"; exit 1; }
log "validating the tasks in the image"
python3 "$HERE/validate.py" --sandbox docker --image "$IMAGE" --docker-label "$LABEL" --docker-prefix "$PREFIX" \
  --jobs 4 --out "$OUT/validation.json" > "$OUT/validation.txt" 2>&1 \
  || { log "task validation failed (see validation.txt)"; tail -5 "$OUT/validation.txt"; exit 1; }
tail -1 "$OUT/validation.txt" | tee -a "$OUT/quant_run.log"

cleanup() {
  docker logs "$NAME" 2>&1 | grep -iE "error|warn|nan|fail|abort|checkpoint|re-processing|backend sampling" \
    | grep -viE "eval time|tokens per second|t/s" > "$OUT/server.log"
  docker rm -f "$NAME" > /dev/null 2>&1
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2> /dev/null
  docker ps -aq --filter "name=^$PREFIX-" | xargs -r docker rm -f > /dev/null 2>&1
  log "stopped"
}
trap cleanup EXIT
docker rm -f "$NAME" > /dev/null 2>&1
# shellcheck disable=SC2086
docker run -d --name "$NAME" --label "$LABEL" --network host --gpus "\"device=$GPUS\"" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID $MOUNTS "$SERVER_IMAGE" "$LLAMA_SERVER" \
  -m "$MODEL" -ngl 99 -fa on -np "$NP" -kvu -c "$CTX" -b 2048 -ub 512 --ctx-checkpoints 4 --cache-ram 0 \
  --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui \
  --host 127.0.0.1 --port "$PORT" > /dev/null || { log "server did not start"; exit 1; }
log "server $NAME on GPUs $GPUS, port $PORT; waiting for /health"
until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10
  [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2> /dev/null)" = running ] \
    || { log "server died while loading"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
done
python3 -I "$HERE/proxy.py" --listen "127.0.0.1:$PROXY_PORT" --upstream "127.0.0.1:$PORT" --log "$OUT/proxy.jsonl" \
  >> "$OUT/proxy.out" 2>&1 &
PROXY_PID=$!
sleep 1
log "running tasks $TASKS, attempts $ATTEMPTS, $NP in flight"
python3 "$HERE/run.py" --arm "$ARM" --base-url "http://127.0.0.1:$PROXY_PORT/v1" --out "$OUT/res" --tasks "$TASKS" \
  --attempts "$ATTEMPTS" --parallel "$NP" --context 262144 --work-root "$OUT/work" \
  --sandbox docker --image "$IMAGE" --docker-label "$LABEL" --docker-prefix "$PREFIX" >> "$OUT/run.log" 2>&1
rc=$?
python3 -I "$HERE/report.py" table "$OUT/res" --log "$OUT/proxy.jsonl" --csv "$OUT/attempts.csv" > "$OUT/table.md"
cat "$OUT/table.md" | tee -a "$OUT/quant_run.log"
rm -rf "$OUT/work"
exit $rc
