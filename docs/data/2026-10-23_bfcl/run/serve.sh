#!/bin/bash
# Q30 BFCL v3 check (2026-10-23): start one llama-server container (pascal-quants label) and wait until it answers
# /health. WORK: folder holding the llama.cpp builds; MODELS: folder holding the GGUF files; BIN: default llama-server.
# serve.sh NAME PORT GPUS TS MODEL [BIN] [ENV...]
set -e
: "${WORK:?}" "${MODELS:?}"
NAME=$1 PORT=$2 GPUS=$3 TS=$4 MODEL=$5 BIN=${6:-${BIN:?}}
shift 6 || shift $#
ENVS=()
for e in "$@"; do ENVS+=(-e "$e"); done
docker run -d --name "$NAME" --label pascal-quants=1 --network host --gpus "\"device=$GPUS\"" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${ENVS[@]}" \
  -v "$WORK:$WORK:ro" -v "$MODELS:$MODELS:ro" \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  "$BIN" -m "$MODEL" -c 16384 -np 2 -ngl 99 -fa on -ts "$TS" --cache-ram 0 --host 127.0.0.1 --port "$PORT"
until curl -sf "127.0.0.1:$PORT/health" >/dev/null; do
  sleep 5
  [ "$(docker inspect -f '{{.State.Status}}' "$NAME")" = running ] || { docker logs "$NAME" 2>&1 | tail -20; exit 1; }
done
echo "$NAME ready on $PORT"
