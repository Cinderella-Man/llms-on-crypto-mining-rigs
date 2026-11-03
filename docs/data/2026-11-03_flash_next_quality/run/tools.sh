#!/usr/bin/env bash
# llama-imatrix, llama-quantize and llama-gguf-split for compute 8.6 from the Miner Llama v3 base (37e717ff6), the
# commit whose llama-perplexity scores every file here, built in Docker (CUDA 12.9) at a low CPU share with the flags of
# ../../2026-11-03_table_bits/run/build.sh. BUILD_FROM (and CCACHE_FROM) name an existing build folder (and ccache) of
# that commit and those flags, made by that script: copied first, only the three tools compile.
#   (env: WORK SRC [BUILD_FROM] [CCACHE_FROM] [JOBS]) bash tools.sh   ->  $WORK/tools/bin/{llama-imatrix,...}
set -euo pipefail
. "$(dirname "$0")/common.sh"
: "${WORK:?}" "${SRC:?}"
T=$WORK/tools
mkdir -p "$T/bin"
[ -n "${BUILD_FROM:-}" ] && [ ! -d "$T/build" ] && cp -a "$BUILD_FROM" "$T/build"
[ -n "${CCACHE_FROM:-}" ] && [ ! -d "$T/ccache" ] && cp -a "$CCACHE_FROM" "$T/ccache"
docker run --rm $DOCKER_BASE --name "pq-$TAG-tools" -e JOBS="${JOBS:-8}" -v "$T:/work" -v "$SRC:/src:ro" "$IMAGE" \
    bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    git config --global --add safe.directory /src
    cmake -S /src -B /work/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
    nice -n 19 cmake --build /work/build -j "$JOBS" --target llama-imatrix llama-quantize llama-gguf-split > /dev/null
    cp /work/build/bin/llama-imatrix /work/build/bin/llama-quantize /work/build/bin/llama-gguf-split /work/bin/
    { echo "commit: $(git -C /src rev-parse HEAD) ($(git -C /src log -1 --format=%ci))"
      echo "arch: sm_86, GGML_NATIVE=ON"; echo "built: $(date -Is)"; } > /work/bin/BUILDINFO.txt
    cat /work/bin/BUILDINFO.txt'
ls -l "$T/bin"
