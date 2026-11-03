#!/usr/bin/env bash
# Builds build/megakernel_p106 inside the CUDA 12.9 container (CUDA 13 can no longer target Pascal).
#
# The binary carries:
#   - sm_61 machine code, which the P106 rig runs;
#   - compute_61 PTX, so a newer GPU can run the exact Pascal code path through the driver's JIT
#     (CUDA_FORCE_PTX_JIT=1) for correctness runs;
#   - sm_86 (RTX 30-series) machine code for correctness runs (its timings mean nothing for Pascal).
# Host code is built without AVX (-march=x86-64 -mno-avx), because the rig's Celeron has none, and the CUDA runtime
# is linked statically, so the binary needs only the NVIDIA driver on the machine that runs it.
#
# Usage: bench/megakernel_p106/build.sh   (env IMAGE overrides the container image, DOCKER_ARGS adds docker run options)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p megakernel_p106/build

# shellcheck disable=SC2086
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w/megakernel_p106 "$IMAGE" bash -euo pipefail -c '
    nvcc -O3 -std=c++17 -use_fast_math \
        -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_86,code=sm_86 \
        -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" \
        -Xptxas -v -o build/megakernel_p106 src/main.cu 2> build/ptxas.log \
        || { cat build/ptxas.log; exit 1; }
    cuobjdump -res-usage build/megakernel_p106 > build/res-usage.txt
    cuobjdump -sass -arch sm_61 build/megakernel_p106 > build/sm61.sass
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)"
        echo "avx (ymm) instructions in the binary: $(objdump -d build/megakernel_p106 | grep -c ymm || true)"
        echo "highest glibc symbol: $(objdump -T build/megakernel_p106 | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        echo "dynamic libraries:"; ldd build/megakernel_p106 | sed "s/^/  /"
        echo "md5: $(md5sum build/megakernel_p106 | cut -d" " -f1)"
    } > build/BUILDINFO.txt
    cat build/BUILDINFO.txt
'
