#!/usr/bin/env bash
# Builds build/mmvq_kcol (src/kcol_main.cu: the k-column products of src/kcol_mm.cuh at 1 to 8 columns) inside the
# CUDA 12.9 container, like build.sh: sm_61 machine code for the P106, compute_61 PTX for correctness runs on a newer
# GPU (CUDA_FORCE_PTX_JIT=1), sm_120 for an RTX 50-series card; host code without AVX, the CUDA runtime linked
# statically.
#
# Usage: bench/mmvq_pascal/build_kcol.sh       (env IMAGE overrides the container image, DOCKER_ARGS adds run arguments)
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build

docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    nvcc -O3 -std=c++17 -use_fast_math \
        -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_120,code=sm_120 \
        -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" \
        -Xptxas -v -o build/mmvq_kcol src/kcol_main.cu 2> build/ptxas_kcol.log \
        || { cat build/ptxas_kcol.log; exit 1; }
    cuobjdump -res-usage build/mmvq_kcol > build/res-usage_kcol.txt
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)"
        echo "avx (ymm) instructions in the binary: $(objdump -d build/mmvq_kcol | grep -c ymm || true)"
        echo "highest glibc symbol: $(objdump -T build/mmvq_kcol | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
    } > build/BUILDINFO_kcol.txt
    cat build/BUILDINFO_kcol.txt
'
