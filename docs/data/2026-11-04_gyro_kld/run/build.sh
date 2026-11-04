#!/usr/bin/env bash
# agentionai's llama.cpp fork (the one engine that reads the Gyro trellis types TQ2_T, TQK6/TQK7 and TQP5-7), built
# from source at a pinned commit in Docker, for one backend:
#   cuda    CUDA 12.9 for compute 8.6 (RTX 3080 / 3090), the backend our KL-divergence bases were made with
#   vulkan  Vulkan (Ubuntu 24.04's loader, headers and glslc); on NVIDIA cards it runs with the driver's Vulkan ICD,
#           which the NVIDIA Container Toolkit mounts when NVIDIA_DRIVER_CAPABILITIES includes graphics
# Only llama-perplexity (and llama-gguf, for the headers) are built, with llama/ggml linked statically.
#   (env: WORK [REF] [BACKEND] [JOBS] [DOCKER_ARGS]) bash build.sh
# Output: $WORK/<backend>/bin/ and $WORK/<backend>/BUILDINFO.txt; the source tree is $WORK/src, ccache $WORK/ccache.
set -euo pipefail
: "${WORK:?}"
REF=${REF:-9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f}
BACKEND=${BACKEND:-cuda}
case $BACKEND in
    cuda) IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
          PKGS="" FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF" ;;
    vulkan) IMAGE=${IMAGE:-ubuntu:24.04}
          PKGS="libvulkan-dev glslc spirv-headers" FLAGS="-DGGML_VULKAN=ON" ;;
    *) echo "BACKEND is cuda or vulkan" >&2; exit 1 ;;
esac
mkdir -p "$WORK/ccache"
docker run --rm ${DOCKER_ARGS:-} --name "pq-gyro-build-$BACKEND" -e REF="$REF" -e BACKEND="$BACKEND" -e PKGS="$PKGS" \
    -e FLAGS="$FLAGS" -e JOBS="${JOBS:-8}" -e IMAGE="$IMAGE" -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache $PKGS > /dev/null
    git config --global --add safe.directory /work/src
    [ -d /work/src/.git ] || git clone -q https://gitlab.com/agentionai-group/llama.cpp.git /work/src
    cd /work/src
    git cat-file -e "$REF^{commit}" 2> /dev/null || git fetch -q origin
    git checkout -q --force "$REF"
    b=/work/build-$BACKEND
    cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF $FLAGS \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
    cmake --build $b -j "$JOBS" --target llama-perplexity llama-gguf > /dev/null
    out=/work/$BACKEND
    rm -rf $out && mkdir -p $out/bin
    cp $b/bin/llama-perplexity $b/bin/llama-gguf $out/bin/
    {
        echo "backend: $BACKEND"
        echo "commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci)) $(git describe --tags --always)"
        echo "image: $IMAGE"
        echo "flags: $FLAGS"
        echo "built: $(date -Is)"
        echo "dynamic libraries:"; ldd $out/bin/llama-perplexity | sed "s/ (0x[0-9a-f]*)//; s/^/  /"
    } > $out/BUILDINFO.txt
    cat $out/BUILDINFO.txt
'
