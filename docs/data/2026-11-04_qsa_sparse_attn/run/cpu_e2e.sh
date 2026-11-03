#!/usr/bin/env bash
# CPU-only end-to-end check of the branch's host-side changes (the QSA k-pool inputs): builds test-llama-archs and
# test-llama-context without CUDA from a llama.cpp tree and runs them. test-llama-archs builds a small random qwen4exp
# whose attention layers are all QSA layers (compress ratio 4) and checks the CPU logits against a second CPU context,
# a state roundtrip and a chunked decode against one batch, so the k-pool inputs (pools, pads, tails) of every ubatch
# take part; test-llama-context then runs its memory cases on the same random models. Run it on the branch and on its
# base; the qwen4exp rows must be OK on both, with the same numbers.
#
# Usage: SRC=<llama.cpp tree> OUT=<dir> [JOBS=6] [DOCKER_ARGS=...] cpu_e2e.sh
# Output: $OUT/build (the CPU build), $OUT/archs.txt, $OUT/archs_stats.txt (with LLAMA_QSA_INPUT_STATS=1),
# $OUT/models (the random qwen4exp files, test-llama-archs -o) and $OUT/context.txt (test-llama-context on each).
set -euo pipefail
SRC=$(realpath "${SRC:?SRC=<llama.cpp tree>}")
mkdir -p "${OUT:?OUT=<dir>}"
OUT=$(realpath "$OUT")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
docker run --rm ${DOCKER_ARGS:-} -e JOBS="${JOBS:-6}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$SRC:/src:ro" -v "$OUT:/out" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF > /out/cmake.log
    cmake --build /out/build -j "$JOBS" --target test-llama-archs test-llama-context > /out/build.log
    cd /out/build/bin
    ./test-llama-archs -a "^qwen4exp$" -s 42 > /out/archs.txt 2>&1 || echo "test-llama-archs rc=$?" >> /out/archs.txt
    LLAMA_QSA_INPUT_STATS=1 ./test-llama-archs -a "^qwen4exp$" -s 42 -v 4 > /out/archs_stats.txt 2>&1 \
        || echo "rc=$?" >> /out/archs_stats.txt
    rm -rf /out/models && mkdir -p /out/models
    ./test-llama-archs -a "^qwen4exp$" -s 42 -o /out/models > /out/models.txt 2>&1 || echo "rc=$?" >> /out/models.txt
    : > /out/context.txt
    for m in /out/models/*.gguf; do
        echo "== $(basename $m)" >> /out/context.txt
        ./test-llama-context -m "$m" -t 4 >> /out/context.txt 2>&1 || echo "test-llama-context rc=$?" >> /out/context.txt
    done
    chown -R "$HOST_UID:$HOST_GID" /out
'
sed 's/\x1b\[[0-9;]*m//g' "$OUT/archs.txt" | grep -i "qwen4exp\|rc=" || true
grep -E "^==|OK|FAIL|SKIP|passed|rc=" "$OUT/context.txt" | tail -12
