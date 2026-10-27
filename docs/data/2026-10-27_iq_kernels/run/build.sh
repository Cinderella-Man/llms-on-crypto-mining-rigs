#!/usr/bin/env bash
# Builds the fork's branch pascal-p106-v3-iqmmvq (IQ2_S / IQ3_XXS Pascal matrix-vector kernels) on a machine with
# Docker, from a source copy in $WORK/src (e.g. `git archive HEAD | ssh <machine> "tar -x -C $WORK/src"`, with the
# commit in $WORK/src/COMMIT). One build serves both the rig and a test GPU: code for compute capability 6.1 (the P106
# runs its SASS) plus compute_61 PTX and sm_70 code, so a V100 runs test-backend-ops natively and with
# CUDA_FORCE_PTX_JIT=1 (the 6.1 code path). The other flags are those of scripts/build_llamacpp_docker.sh (static, no
# AVX, CUDA 12.9 on Ubuntu 24.04). CPU only, at a low CPU share, -j 8, in a labelled container.
#   ssh <build machine> "WORK=<folder> NAME=llama.cpp-<date>-<sha>-<branch> bash -s" < run/build.sh
# Output: $WORK/out/$NAME/{bin,BUILDINFO.txt} and $WORK/out/$NAME.tar.gz (md5 printed).
set -euo pipefail
WORK=${WORK:?folder holding src/}
NAME=${NAME:?NAME=llama.cpp-<date>-<sha>-<branch>}
ARCHS=${ARCHS:-61;70-real}
LABEL=${LABEL:-pascal-quants=1}
mkdir -p "$WORK/out/ccache"
docker run --rm --cpu-shares=256 --label "$LABEL" --name v3iq-build -e NAME="$NAME" -e ARCHS="$ARCHS" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -v "$WORK/src:/src:ro" -v "$WORK/out:/out" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git binutils > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target test-backend-ops llama-server llama-bench llama-cli llama-completion \
        llama-batched-bench llama-perplexity > /out/build.log 2>&1 \
        || { grep -E "error" /out/build.log | head -40; tail -20 /out/build.log; exit 1; }
    grep -iE "warning" /out/build.log | grep -iE "mmvq-pascal" | head -20 || true
    out=/out/$NAME
    rm -rf "$out" && mkdir -p "$out/bin"
    cp /out/build/bin/llama-* /out/build/bin/test-backend-ops "$out/bin/"
    {
        echo "name: $NAME"
        echo "ref: pascal-p106-v3-iqmmvq (source copy)"
        echo "commit: $(cat /src/COMMIT)"
        echo "image: nvidia/cuda:12.9.2-devel-ubuntu24.04"
        echo "cuda architectures: $ARCHS"
        echo "built: $(date -Is)"
        echo "avx instructions in llama-bench: $(objdump -d /out/build/bin/llama-bench | grep -c ymm)"
        echo "highest glibc symbol: $(objdump -T /out/build/bin/llama-bench | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
    } > "$out/BUILDINFO.txt"
    tar czf "/out/$NAME.tar.gz" -C /out "$NAME"
    md5sum "/out/$NAME.tar.gz"
    chown -R "$HOST_UID:$HOST_GID" /out
    cat "$out/BUILDINFO.txt"
'
