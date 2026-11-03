#!/usr/bin/env bash
# The two programs this folder runs on the quality machine's Ampere cards, built in Docker (CUDA 12.9):
#   - llama-perplexity (and llama-gguf) from a llama.cpp checkout at the Miner Llama v3 base commit, which knows
#     Flash-Next (`qwen4exp`), for compute 8.6 with the host's own CPU features: the rig build (compute 6.1 PTX, no
#     AVX) also runs on Ampere through the driver's JIT, but its batched MMQ cases abort there;
#   - table_bits (table_bits.cpp, next to this script): the n-gram table census and round trips, on the CPU.
#   WORK=<build folder> SRC=<llama.cpp checkout> [ARCH=86] [JOBS=8] [ONLY=table_bits] \
#     [DOCKER_ARGS="--label ... --cpu-shares 256"] bash build.sh
# Output: $WORK/bin/{llama-perplexity,llama-gguf,table_bits} and $WORK/bin/BUILDINFO.txt.
set -euo pipefail
: "${WORK:?}" "${SRC:?}"
ARCH=${ARCH:-86}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/bin" "$WORK/ccache"
cp "$HERE/table_bits.cpp" "$WORK/"
docker run --rm ${DOCKER_ARGS:-} -e ARCH="$ARCH" -e JOBS="${JOBS:-8}" -e ONLY="${ONLY:-}" -v "$WORK:/work" \
    -v "$SRC:/src" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    g++ -O3 -march=native -fopenmp -std=c++17 -Wall -o /work/bin/table_bits /work/table_bits.cpp
    [ "$ONLY" = table_bits ] && exit 0
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    git config --global --add safe.directory /src
    cmake -S /src -B /work/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
    cmake --build /work/build -j "$JOBS" --target llama-perplexity llama-gguf > /dev/null
    cp /work/build/bin/llama-perplexity /work/build/bin/llama-gguf /work/bin/
    { echo "commit: $(git -C /src rev-parse HEAD) ($(git -C /src log -1 --format=%ci))"
      echo "arch: sm_$ARCH, GGML_NATIVE=ON"; echo "image: '"$IMAGE"'"; echo "built: $(date -Is)"
      echo "table_bits.cpp md5: $(md5sum /work/table_bits.cpp | cut -d" " -f1)"; } > /work/bin/BUILDINFO.txt
    cat /work/bin/BUILDINFO.txt
'
