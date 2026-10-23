#!/usr/bin/env bash
# Builds llama.cpp's test-backend-ops and a few tools from a local source tree for the GPU of this PC plus compute_61
# PTX, so that the Pascal code paths can be tested here: the CUDA driver JIT-compiles the compute_61 PTX when
# CUDA_FORCE_PTX_JIT=1 is set. Uses CUDA 12.9 in Docker (CUDA 13 can't target Pascal); the binaries then run on the host
# with the CUDA 12 runtime libraries this script copies out of the image.
#
# Usage: scripts/build_llamacpp_tests_docker.sh <llama.cpp source dir> <out dir> [host arch, default 120-real]
# Then, e.g. for the fork's Pascal matrix-vector kernels:
#   export LD_LIBRARY_PATH=<out dir>/cuda12-libs
#   GGML_CUDA_MMVQ_PASCAL=2 <out dir>/build/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION
#   CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2 <out dir>/build/bin/test-backend-ops -b CUDA0 -o MUL_MAT -p 'n=1,'
# The source tree is mounted read-only; ccache lives in <out dir>/ccache, so rebuilds take about a minute.
set -euo pipefail
SRC=$(realpath "$1")
mkdir -p "$2"
OUT=$(realpath "$2")
HOST_ARCH=${3:-120-real}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT/ccache" "$OUT/cuda12-libs"
docker run --rm -v "$SRC:/src:ro" -v "$OUT:/out" -e ARCHS="61-virtual;$HOST_ARCH" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git > /dev/null
    git config --global --add safe.directory "*"
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_FA=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1
    cmake --build /out/build -j "$(nproc)" \
        --target test-backend-ops test-export-graph-ops llama-bench llama-perplexity llama-server > /out/build.log 2>&1 \
        || { tail -40 /out/build.log; exit 1; }
    cp -L /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 /out/cuda12-libs/
    chown -R "$HOST_UID:$HOST_GID" /out
    echo "built: /out/build/bin; runtime libraries: /out/cuda12-libs"
'
