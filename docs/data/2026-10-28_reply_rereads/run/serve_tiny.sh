#!/usr/bin/env bash
# The v2.0 llama-server (CPU build from build_cpu.sh) on a tiny random Flint GGUF that carries Flint's own vocabulary
# and chat template, with the rig's chat flags (--jinja --reasoning-format deepseek). Only /apply-template, /tokenize
# and /detokenize are used, so the weights do not matter. Listens on 127.0.0.1:$PORT.
#   WORK=... MODEL=/path/tiny-flint.gguf PORT=18431 bash serve_tiny.sh
set -euo pipefail
: "${WORK:?set WORK}" "${MODEL:?set MODEL}"
PORT=${PORT:-18431}
docker rm -f pq-rereads-tiny >/dev/null 2>&1 || true
docker run -d --rm --cpu-shares=256 --label pascal-quants=1 --name pq-rereads-tiny --network host \
    -v "$WORK/out-cpu/build/bin:/bin-llama:ro" -v "$MODEL:/model.gguf:ro" nvidia/cuda:12.9.2-devel-ubuntu24.04 \
    /bin-llama/llama-server -m /model.gguf -c 4096 -np 1 -t 2 --jinja --reasoning-format deepseek \
    --host 127.0.0.1 --port "$PORT" --no-webui
for i in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "up on $PORT"; exit 0; }; sleep 1; done
docker logs pq-rereads-tiny 2>&1 | tail -20; exit 1
