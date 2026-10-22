#!/usr/bin/env bash
# Builds llama.cpp for the P106 rig on any x86-64 Linux PC with Docker, then packs the binaries into a tarball you
# copy to the rig. Building on the rig itself works too, but its 2-core Celeron is slow at it.
#
# The build targets exactly what the rig can run:
#   - CUDA 12.9 inside the container (CUDA 13 can't build for Pascal), code for compute capability 6.1 only;
#   - CPU code without AVX/AVX2/FMA/F16C, because the rig's Celeron has none of them (an AVX binary dies with
#     "Illegal instruction");
#   - Ubuntu 24.04 userland, so the binaries match the rig's glibc;
#   - llama/ggml linked statically; only the CUDA runtime libraries (libcudart, libcublas) come from the rig's own
#     CUDA 12.9 installation.
#
# Usage: scripts/build_llamacpp_docker.sh [git-ref] [--cuda-graphs-pascal]
#   git-ref               llama.cpp branch, tag or commit (default: master)
#   --cuda-graphs-pascal  also apply llama.cpp PR #27721, which re-enables CUDA graphs on Pascal (compute 6.1)
# Output: $WORK/llama.cpp-<date>-<sha>[-graphs61].tar.gz with bin/ and BUILDINFO.txt (WORK defaults to ~/llama-builds).
# On the rig: tar xzf <tarball> -C /opt && /opt/<name>/bin/llama-server --list-devices
set -euo pipefail

REF=${1:-master}
PATCH=${2:-}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
WORK=${WORK:-$HOME/llama-builds}
mkdir -p "$WORK/ccache"

docker run --rm -e REF="$REF" -e PATCH="$PATCH" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache curl > /dev/null

    [ -d /work/src/.git ] || git clone -q https://github.com/ggml-org/llama.cpp /work/src
    cd /work/src
    git config --global --add safe.directory /work/src
    git fetch -q origin && git checkout -q --force "$REF" && git reset -q --hard
    git clean -qfdx
    [ "$REF" = master ] && git reset -q --hard origin/master
    sha=$(git rev-parse --short=9 HEAD)
    name=llama.cpp-$(date +%Y%m%d)-$sha

    if [ "$PATCH" = --cuda-graphs-pascal ]; then
        curl -sL https://patch-diff.githubusercontent.com/raw/ggml-org/llama.cpp/pull/27721.diff | git apply -v \
            || sed -i "s/devices\[cuda_ctx->device\].cc < GGML_CUDA_CC_VOLTA) {/devices[cuda_ctx->device].cc < GGML_CUDA_CC_DP4A) {/" ggml/src/ggml-cuda/ggml-cuda.cu
        grep -q "cc < GGML_CUDA_CC_DP4A) {" ggml/src/ggml-cuda/ggml-cuda.cu || { echo "patch failed"; exit 1; }
        name=$name-graphs61
    fi

    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
    cmake --build build -j "$(nproc)" --target llama-bench llama-server llama-cli llama-completion \
        llama-perplexity llama-batched-bench llama-gguf > /dev/null

    out=/work/$name
    rm -rf "$out" && mkdir -p "$out/bin"
    cp build/bin/llama-* "$out/bin/"
    {
        echo "name: $name"
        echo "ref: $REF"
        echo "commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
        echo "patch: ${PATCH:-none}"
        echo "image: '"$IMAGE"'"
        echo "built: $(date -Is)"
        echo "avx instructions in llama-bench: $(objdump -d build/bin/llama-bench | grep -c ymm)"
        echo "highest glibc symbol: $(objdump -T build/bin/llama-bench | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        echo "dynamic libraries:"; ldd build/bin/llama-bench | sed "s/^/  /"
    } > "$out/BUILDINFO.txt"
    tar czf "/work/$name.tar.gz" -C /work "$name"
    chown -R "$HOST_UID:$HOST_GID" /work
    cat "$out/BUILDINFO.txt"
'
