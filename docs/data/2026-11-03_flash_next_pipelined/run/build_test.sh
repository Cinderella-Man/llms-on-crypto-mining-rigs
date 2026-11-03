#!/usr/bin/env bash
# Q550: test build of fork branch pascal-p106-v3-hybrid (./src, from `git archive`) for the RTX 3090 (sm_86)
# plus compute_61 PTX, -j 8, low CPU share, in a container: test-backend-ops, test-llama-archs (writes the
# random qwen4exp model), llama-completion, llama-pipespec, llama-server, llama-pipe-recurrent-check. ccache in
# ./out/ccache. Run it next to ./src (e.g. `git archive` of the branch).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out/ccache out/cuda12-libs
docker run --rm --cpu-shares=256 --name pq-build-test -v "$PWD/src:/src:ro" -v "$PWD/out:/out" \
    -e ARCHS="61-virtual;86-real" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target test-backend-ops test-llama-archs llama-perplexity llama-completion \
        llama-pipespec llama-server llama-pipe-recurrent-check > /out/build.log 2>&1 \
        || { grep -E "error|Error" /out/build.log | head -40; tail -20 /out/build.log; exit 1; }
    cp -L /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 /out/cuda12-libs/
    grep -E "warning" /out/build.log | grep -i "llama-graph\|llama-memory\|llama-context\|llama-kv\|qwen4exp" | head -20 || true
    echo "built: out/build/bin"
'
