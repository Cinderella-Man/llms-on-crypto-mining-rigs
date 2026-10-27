#!/usr/bin/env bash
# Q41 pilot (2026-10-23/24), the job each pilot file ran on the quant machine. WORK: work dir with harness/ (bench/ at
# 39ecdd2), venv/ (the client's Python), bfcl-v4/ (the unpacked bfcl-eval wheel), e1_pilot.jsonl, template.jinja,
# raw/, logs/; BIN: llama-server under WORK; MODELS: the folder holding MODEL; GPUS: the GPU set, e.g. 3,4.
# pilot.sh TAG MODEL [ENV=VAL ...]: llama-server with the same flags for every file, then BFCL v4 E1 (376 tasks) and E2
# (25 scenarios x 4 categories), K = 5, the makers' sampling.
# The server log keeps only warnings and errors: no timing lines (speed is measured on the rig only).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODELS:?}" "${GPUS:?}"
TAG=$1; MODEL=$2; shift 2
V=$WORK; PORT=18190; NAME=pq-bfclv4-srv
envs=(); for e in "$@"; do envs+=(-e "$e"); done
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --label pascal-quants=1 --network host --gpus "\"device=$GPUS\"" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${envs[@]}" \
  -v "$WORK:$WORK:ro" -v "$MODELS:$MODELS:ro" \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 "$BIN" \
  -m "$MODEL" -ngl 99 -fa on -np 8 -kvu -c 262144 -b 2048 -ub 512 --host 127.0.0.1 --port $PORT >/dev/null
cleanup() {
  docker logs $NAME 2>&1 | grep -iE "error|warn|nan|fail|abort" | grep -viE "eval time|tokens per second|t/s" \
    > $V/logs/server-$TAG.txt
  docker rm -f $NAME >/dev/null 2>&1
}
trap cleanup EXIT
until curl -sf 127.0.0.1:$PORT/health >/dev/null; do
  sleep 10
  [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" = running ] || { echo "server died"; exit 1; }
done
client() { docker run --rm --label pascal-quants=1 --network host -e BFCL_ROOT=/v4/bfcl-v4 -v $V:/v4 \
  -w /v4/harness/bfcl_flint python:3.12-slim /v4/venv/bin/python "$@"; }
rc=0
client run.py http://127.0.0.1:$PORT /v4/e1_pilot.jsonl /v4/raw/e1-$TAG.jsonl --attempts 1-5 --conc 8 || rc=1
client multi_turn.py run http://127.0.0.1:$PORT /v4/template.jinja /v4/raw/e2-$TAG.jsonl --scenarios 25 \
  --attempts 1-5 --conc 8 || rc=1
exit $rc
