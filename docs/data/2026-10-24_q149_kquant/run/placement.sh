#!/usr/bin/env bash
# Q149 (2026-10-24): where the e2e runs put the Q6_K output head (-ot output\.weight=CUDA0): one verbose load, one
# token; read the buffer lines of $WORK/e2e/logs/placement.log.
# Usage: the environment of e2e.sh (BUILD, MODEL, WORK, GPU), then placement.sh
W=${WORK:?WORK=<work dir>}
BUILD=$(realpath "${BUILD:?BUILD=<unpacked build with bin/>}")
MODEL=$(realpath "${MODEL:?MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf>}")
mkdir -p $W/e2e/logs $W/nvcache
docker run --rm --name q149-e2e --gpus "\"device=${GPU:-0}\"" -v "$BUILD:/b:ro" -v "$MODEL:/m/model.gguf:ro" \
    -v "$W/nvcache:/nv" -e CUDA_CACHE_PATH=/nv nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 400 \
    /b/bin/llama-completion -m /m/model.gguf -ngl 2 -ot 'output\.weight=CUDA0' -t 8 -fa on -p hi -n 1 -no-cnv -lv 4 \
    > $W/e2e/logs/placement.log 2>&1
echo "exit $?" >> $W/e2e/logs/placement.log
