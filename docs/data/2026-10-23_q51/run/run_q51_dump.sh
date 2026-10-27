#!/usr/bin/env bash
# Q51 (2026-10-23) re-run of the routing pass only (every token requests logits, so the last layer is recorded too), then the analysis.
set -euo pipefail
WORK=${WORK:?WORK=<work dir: src/llama.cpp, src/q51 = a copy of bench/expert_overlap, work/q51/prompts.jsonl>}
W=$WORK/work/q51
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
MODEL_DIR=${MODELS:?MODELS=<directory holding flint-Q8_0.gguf>}
echo "== build $(date +%T)"
docker run --rm --name pq-q51-build --label pascal-quants=1 -v "$WORK":/w $IMG bash -euo pipefail -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache python3 > /dev/null
  cd /w/src/llama.cpp && rm -rf examples/route-dump && cp -r /w/src/q51/route_dump examples/route-dump
  export CCACHE_DIR=/w/builds/ccache
  cmake --build build-cuda -j 22 --target llama-route-dump 2>&1 | grep -E "error|Error" | head -20 || true
  ls -la build-cuda/bin/llama-route-dump
'
echo "== route dump $(date +%T)"
rm -f $W/routes.bin
docker run --rm --name pq-q51-dump --label pascal-quants=1 --gpus '"device=3,4"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e ROUTE_IN=/w/work/q51/seqs.txt -e ROUTE_OUT=/w/work/q51/routes.bin \
  -v "$WORK":/w -v $MODEL_DIR:/models:ro $IMG \
  /w/src/llama.cpp/build-cuda/bin/llama-route-dump -m /models/flint-Q8_0.gguf -ngl 99 -c 16384 -b 512 -ub 512 -fa on \
  > $W/dump.log 2>&1
tail -2 $W/dump.log
echo "== analyze $(date +%T)"
docker run --rm --name pq-q51-ana --label pascal-quants=1 -v "$WORK":/w python:3.12-slim \
  python -I /w/src/q51/analyze.py /w/work/q51/routes.bin /w/work/q51/gen.jsonl /w/work/q51/q51.json \
  > $W/analysis.txt 2>&1
cat $W/analysis.txt
echo "== DONE $(date +%T)"
