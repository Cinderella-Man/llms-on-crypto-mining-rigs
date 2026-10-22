#!/usr/bin/env bash
# Builds llama.cpp's ggml libraries (static, CUDA) for the bit-exactness check, without touching the source tree:
# the source is mounted read-only and everything is written to <out dir>/build.
#
# Usage: verify/build_ggml.sh <llama.cpp source dir> <out dir> [61-virtual|120]
#   The source must be commit d77256999 with patches/q4_1_64-type.diff applied.
#   61-virtual  compute_61 PTX only: checks the Pascal code path on a newer GPU (driver JIT)
#   120         native code for an RTX 50-series GPU
set -euo pipefail
SRC=$(realpath "$1")
mkdir -p "$2"
OUT=$(realpath "$2")
ARCHS=${3:-61-virtual}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
docker run --rm -v "$SRC:/src:ro" -v "$OUT:/out" -e ARCHS="$ARCHS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_FA=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets > /out/cmake.log
    cmake --build /out/build -j "$(nproc)" --target ggml-cuda ggml-cpu ggml-base ggml > /out/build.log
    chown -R "$HOST_UID:$HOST_GID" /out
    echo "ggml libraries in /out/build/ggml/src"
'
