#!/usr/bin/env bash
# Builds build/insn_probe inside the CUDA 12.9 container (CUDA 13 can no longer target Pascal): sm_61 machine code for
# the P106 rig, compute_61 PTX (so a newer GPU can run the Pascal code path through the driver's JIT for a smoke test),
# host code without AVX, the CUDA runtime linked statically. build/sass_counts.txt counts, per kernel, the instruction
# it is meant to time in the sm_61 SASS: 64 per loop iteration (8 chains x 8 steps; 128 F2F for f2f, 3 XMAD per
# step for imad), times the compiler's unrolling, so a count below 64 means the compiler merged steps and the kernel
# would time something else.
#
# Usage: bench/insn_probe/build.sh   (env IMAGE overrides the container image, DOCKER_ARGS adds docker run options)
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
# shellcheck disable=SC2086
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    nvcc -O3 -std=c++17 -gencode arch=compute_61,code=[sm_61,compute_61] -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" -Xptxas -v \
        -o build/insn_probe src/main.cu 2> build/ptxas.log || { cat build/ptxas.log; exit 1; }
    cuobjdump -sass -arch sm_61 build/insn_probe > build/sm61.sass
    bash sass_counts.sh build/sm61.sass build/ptxas.log > build/sass_counts.txt
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)"
        echo "avx (ymm) instructions in the binary: $(objdump -d build/insn_probe | grep -c ymm || true)"
        echo "highest glibc symbol: $(objdump -T build/insn_probe | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        echo "dynamic libraries:"; ldd build/insn_probe | sed "s/^/  /"
    } > build/BUILDINFO.txt
    cat build/BUILDINFO.txt build/sass_counts.txt
'
