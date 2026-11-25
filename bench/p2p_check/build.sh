#!/usr/bin/env bash
# Builds build/p2p_check inside the CUDA 12.9 container (CUDA 13 can no longer target Pascal): sm_61 machine code plus
# compute_61 PTX and sm_86, host code without AVX (the rig's Celeron has none), the CUDA runtime linked statically.
#
# Usage: bench/p2p_check/build.sh   (env IMAGE overrides the container image, DOCKER_ARGS adds docker run options)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p p2p_check/build

# shellcheck disable=SC2086
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w/p2p_check "$IMAGE" bash -euo pipefail -c '
    nvcc -O3 -std=c++17 \
        -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_86,code=sm_86 \
        -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" \
        -o build/p2p_check src/main.cu
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)"
        echo "avx (ymm) instructions in the binary: $(objdump -d build/p2p_check | grep -c ymm || true)"
        echo "md5: $(md5sum build/p2p_check | cut -d" " -f1)"
    } > build/BUILDINFO.txt
    cat build/BUILDINFO.txt
'
