#!/usr/bin/env bash
# Makes custom GGUF quantizations on any x86-64 Linux PC with Docker, using llama.cpp's llama-quantize on the CPU.
# Useful on this rig because the formats that decode fastest on Pascal 5-SM cards (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) are
# not always published with an importance matrix, while the published "dynamic" quants use slow K-quant blocks.
#
#   scripts/quantize_docker.sh <llama.cpp source dir> <models dir> <input.gguf> <output.gguf> <type> [imatrix file]
#   (paths relative to <models dir>; requantizing from Q8_0 is allowed and adds only a small error)
#
# Env: EXTRA (extra llama-quantize options; default keeps the output head in Q8_0, which is fast on these cards).
set -euo pipefail

SRC=$(realpath "$1"); MODELS=$(realpath "$2"); IN=$3; OUT=$4; TYPE=$5; IMATRIX=${6:-}
EXTRA=${EXTRA:---output-tensor-type q8_0}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}

docker run --rm -e IN="$IN" -e OUT="$OUT" -e TYPE="$TYPE" -e IMATRIX="$IMATRIX" -e EXTRA="$EXTRA" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -v "$SRC:/src" -v "$MODELS:/models" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake build-essential > /dev/null
    cmake -S /src -B /src/build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
    cmake --build /src/build-cpu -j "$(nproc)" --target llama-quantize > /dev/null
    mkdir -p "$(dirname "/models/$OUT")"
    args="--allow-requantize $EXTRA"
    [ -n "$IMATRIX" ] && args="$args --imatrix /models/$IMATRIX"
    /src/build-cpu/bin/llama-quantize $args "/models/$IN" "/models/$OUT" "$TYPE" "$(nproc)" 2>&1 | tail -4
    chown "$HOST_UID:$HOST_GID" "/models/$OUT" "$(dirname "/models/$OUT")"
    chown -R "$HOST_UID" /src/build-cpu
    ls -l "/models/$OUT"
'
