#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25). Run from a copy in the check dir D (README.md): it builds ./src into ./out.
# Q20/D3: test build of the fork's pascal-p106-v2-drafter work tree (./src, the fork checkout) for the RTX 3090
# (sm_86) and V100 (sm_70) plus compute_61 PTX (the rig's code paths under CUDA_FORCE_PTX_JIT=1), flash attention on
# (the drafter's attention uses it), -j 8, labelled container. ccache in ./out/ccache.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/ccache out/cuda12-libs
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-build -v "$PWD/src:/src:ro" -v "$PWD/out:/out" \
    -e ARCHS="61-virtual;70-real;86-real" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target llama-pipe-drafter-check llama-pipespec llama-server llama-completion \
        > /out/build.log 2>&1 || { grep -E "error|Error" /out/build.log | head -60; tail -20 /out/build.log; exit 1; }
    cp -L /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 /out/cuda12-libs/
    grep -E "warning" /out/build.log | grep -i "pipe-drafter\|pipespec\|drafter-check\|server-context" | head -30 || true
    echo "built: out/build/bin"
'
