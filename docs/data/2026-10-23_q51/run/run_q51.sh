#!/usr/bin/env bash
# Q51 (2026-10-23) on a CUDA machine, one step after another: build llama-route-dump, generate with Flint Q8_0 on
# two V100s, stop the server, then run the routing pass over the generated sequences. Quality/statistics only, no timing.
set -euo pipefail
WORK=${WORK:?WORK=<work dir: src/llama.cpp, src/q51 = a copy of bench/expert_overlap, work/q51/prompts.jsonl>}
W=$WORK/work/q51
SRC=$WORK/src
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
MODEL_DIR=${MODELS:?MODELS=<directory holding flint-Q8_0.gguf>}
MODEL=/models/flint-Q8_0.gguf
PORT=18451
mkdir -p $W
echo "== build $(date +%T)"
docker rm -f pq-q51-build >/dev/null 2>&1 || true
docker run --rm --name pq-q51-build --label pascal-quants=1 -v "$WORK":/w $IMG bash -euo pipefail -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache python3 > /dev/null
  cd /w/src/llama.cpp && git config --global --add safe.directory /w/src/llama.cpp
  rm -rf examples/route-dump && cp -r /w/src/q51/route_dump examples/route-dump
  grep -q "add_subdirectory(route-dump)" examples/CMakeLists.txt || echo "add_subdirectory(route-dump)" >> examples/CMakeLists.txt
  export CCACHE_DIR=/w/builds/ccache
  cmake -B build-cuda > /dev/null
  cmake --build build-cuda -j 22 --target llama-route-dump 2>&1 | grep -E "error|Error" | head -20 || true
  ls -la build-cuda/bin/llama-route-dump
'
echo "== serve $(date +%T)"
docker rm -f pq-q51-srv >/dev/null 2>&1 || true
docker run -d --name pq-q51-srv --label pascal-quants=1 --network host --gpus '"device=3,4"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -v "$WORK":/w:ro -v $MODEL_DIR:/models:ro $IMG \
  /w/src/llama.cpp/build-cuda/bin/llama-server -m $MODEL -c 65536 -np 4 -ngl 99 -fa on --jinja \
  --cache-ram 0 --host 127.0.0.1 --port $PORT > /dev/null
until curl -sf 127.0.0.1:$PORT/health > /dev/null; do
  sleep 5
  [ "$(docker inspect -f '{{.State.Status}}' pq-q51-srv)" = running ] || { docker logs pq-q51-srv 2>&1 | tail -30; exit 1; }
done
echo "== generate $(date +%T)"
docker run --rm --name pq-q51-gen --label pascal-quants=1 --network host -v "$WORK":/w python:3.12-slim \
  python -I /w/src/q51/generate.py http://127.0.0.1:$PORT /w/work/q51/prompts.jsonl /w/work/q51/gen.jsonl \
  /w/work/q51/seqs.txt 4 3072
docker logs pq-q51-srv > $W/server.log 2>&1 || true
docker stop pq-q51-srv > /dev/null && docker rm pq-q51-srv > /dev/null
echo "== route dump $(date +%T)"
docker run --rm --name pq-q51-dump --label pascal-quants=1 --gpus '"device=3,4"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e ROUTE_IN=/w/work/q51/seqs.txt -e ROUTE_OUT=/w/work/q51/routes.bin \
  -v "$WORK":/w -v $MODEL_DIR:/models:ro $IMG \
  /w/src/llama.cpp/build-cuda/bin/llama-route-dump -m $MODEL -ngl 99 -c 16384 -b 2048 -ub 512 -fa on \
  > $W/dump.log 2>&1
tail -3 $W/dump.log
ls -la $W
echo "== DONE $(date +%T)"
