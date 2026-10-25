#!/usr/bin/env bash
# Drafter training text from vLLM in one command: start the server (serve_vllm.sh) unless it already answers, wait until
# it does, then run generate.py --api vllm over the prompt files in order, in a client container.
#
# usage: GPUS=2,5 MODEL_DIR=.. HF_DIR=.. WORK=.. DATA=/data/q20 OUT=gen-V1 PROMPTS="prompts-ext.jsonl prompts-ext2.jsonl" \
#        bash run_generation.sh
#   DATA     folder with the prompt files and the claims/ folder shared with other generation copies; the answers go to
#            DATA/OUT.jsonl, progress to DATA/OUT.log
#   CONC     concurrent requests (default 160: ~60-75 % of the KV cache of two 24 GB cards, no preemption)
#   MAXP     longest prompt in tokens (default 6144; prompt + 2,048 must fit MAX_LEN)
# Stop: bash stop_generation.sh (no new requests, running answers finish, then the server stops).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(dirname "$here")
: "${DATA:?}" "${OUT:?}" "${PROMPTS:?}" "${HF_DIR:?}"
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:1440cb63fac5fa19040c295fcd5c6e6d551a506aa72a68fe3425cfd6ef8cf4b8}
PORT=${PORT:-18560}
CLIENT=${CLIENT:-pq-vllm-cli}
if ! curl -sf -m 5 "127.0.0.1:$PORT/health" >/dev/null; then
    bash "$here/serve_vllm.sh"
    for _ in $(seq 1 180); do curl -sf -m 5 "127.0.0.1:$PORT/health" >/dev/null && break; sleep 10; done
    curl -sf -m 5 "127.0.0.1:$PORT/health" >/dev/null || { echo "server did not come up" >&2; exit 1; }
fi
rm -f "$DATA/STOP-$OUT"
docker rm -f "$CLIENT" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$CLIENT" --label pascal-quants=1 --cpu-shares=256 --network host --entrypoint bash ${MOUNTS:-} \
    -v "$src:$src:ro" -v "$DATA:$DATA" -v "$HF_DIR:$HF_DIR:ro" "$IMAGE" -c "cd $src; for P in $PROMPTS; do \
      [ -e $DATA/STOP-$OUT ] && break; echo \"\$(date -u +%FT%TZ) client on \$P\" >> $DATA/$OUT.log; \
      python3 generate.py http://127.0.0.1:$PORT $DATA/\$P $DATA/$OUT.jsonl $DATA/claims --conc ${CONC:-160} \
        --n-predict 2048 --max-prompt ${MAXP:-6144} --stop-file $DATA/STOP-$OUT --api vllm --tokenizer $HF_DIR \
        >> $DATA/$OUT.log 2>&1; done"
echo "client $CLIENT writing $DATA/$OUT.jsonl; progress in $DATA/$OUT.log, server rate in docker logs ${NAME:-pq-vllm-gen}"
