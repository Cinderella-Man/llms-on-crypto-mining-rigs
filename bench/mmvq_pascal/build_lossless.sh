#!/usr/bin/env bash
# Builds the lossless BF16 tools (Q621) inside the CUDA 12.9 container, like build.sh:
#   build/lossless_entropy: CPU only (src/lossless_entropy.cpp): byte entropy and code sizes of a model's BF16 matrices,
#     the round-trip check of the codes, and the weight sample for the bench;
#   build/lossless_bench (src/lossless_main.cu): the P106 matrix-vector bench of plain BF16 against the codes, with
#     sm_61 machine code and compute_61 PTX for correctness runs on a newer GPU (CUDA_FORCE_PTX_JIT=1); no sm_120, which
#     would double the compile time of its ~320 kernel instances for timings that mean nothing for Pascal.
# Host code is built without AVX (the rig's Celeron has none), the CUDA runtime linked statically.
#
# Usage: bench/mmvq_pascal/build_lossless.sh   (env IMAGE overrides the container image, DOCKER_ARGS adds run arguments,
#                                               ONLY=entropy builds the CPU tool alone)
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build

docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -e ONLY="${ONLY:-}" -v "$PWD:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    g++ -O2 -std=c++17 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c -Wall -o build/lossless_entropy src/lossless_entropy.cpp
    if [ "$ONLY" != entropy ]; then
        nvcc -O3 -std=c++17 \
            -gencode arch=compute_61,code=[sm_61,compute_61] \
            -Wno-deprecated-gpu-targets -cudart static \
            -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" \
            -Xptxas -v -o build/lossless_bench src/lossless_main.cu 2> build/ptxas_lossless.log \
            || { cat build/ptxas_lossless.log; exit 1; }
        cuobjdump -res-usage build/lossless_bench > build/res-usage_lossless.txt
        cuobjdump -sass -arch sm_61 build/lossless_bench > build/sm61_lossless.sass
    fi
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1) and $(g++ --version | head -1)"
        for b in build/lossless_entropy build/lossless_bench; do
            [ -f "$b" ] || continue
            echo "$b: avx (ymm) instructions: $(objdump -d "$b" | grep -c ymm || true), highest glibc symbol: $(objdump -T "$b" | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
        done
    } > build/BUILDINFO_lossless.txt
    cat build/BUILDINFO_lossless.txt
'
