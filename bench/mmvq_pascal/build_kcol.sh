#!/usr/bin/env bash
# Builds build/mmvq_kcol (src/kcol_main.cu: the k-column products of src/kcol_mm.cuh at 1 to 8 columns) inside the
# CUDA 12.9 container, like build.sh: sm_61 machine code for the P106, compute_61 PTX for correctness runs on a newer
# GPU (CUDA_FORCE_PTX_JIT=1), sm_120 for an RTX 50-series card; host code without AVX, the CUDA runtime linked
# statically.
#
# Usage: bench/mmvq_pascal/build_kcol.sh       (env IMAGE overrides the container image, DOCKER_ARGS adds run arguments)
# Env IQ_TREE=<llama.cpp tree>: also the grid types and IQ1_M (Q713). Their grids (ggml-common.h of that tree, MIT, the
#     ggml authors) are copied into build/iq_grids.h for the kernels' grids and the host reference.
# Env FMT=1: also Q8_0, Q6_K, Q5_K, Q5_0 and Q4_0 at the dense shapes where the census has none of them (C323, Q901).
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
IQMNT=()
if [ -n "${IQ_TREE:-}" ]; then
    IQMNT=(-v "$(realpath "$IQ_TREE")/ggml/src/ggml-common.h:/iq/ggml-common.h:ro")
fi

docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -e FMT="${FMT:-}" -v "$PWD:/w" "${IQMNT[@]}" -w /w "$IMAGE" bash -euo pipefail -c '
    IQ=""
    if [ -f /iq/ggml-common.h ]; then
        {
            echo "// The grids of the grid types and IQ1_M, copied by build_kcol.sh from ggml-common.h (MIT, the ggml authors)."
            echo "#pragma once"
            echo "#include <cstdint>"
            echo "#define NGRID_IQ1S 2048"
            echo "namespace kcol_grids {"
            awk "/GGML_TABLE_BEGIN\\((uint8_t|uint64_t|uint32_t), (ksigns_iq2xs|iq2xxs_grid|iq2xs_grid|iq2s_grid|iq3xxs_grid|iq3s_grid|iq1s_grid|iq1s_grid_gpu),/ { p = 1 }
                 p { print } /GGML_TABLE_END/ { p = 0 }" /iq/ggml-common.h |
                sed -E "s/GGML_TABLE_BEGIN\\(([a-z0-9_]+), ([a-z0-9_]+), ([A-Z0-9_]+)\\)/static const \\1 \\2[\\3] = {/; s/GGML_TABLE_END\\(\\)/};/"
            echo "} // namespace kcol_grids"
        } > build/iq_grids.h
        [ "$(grep -c "^static const" build/iq_grids.h)" = 8 ] || { echo "iq_grids.h: expected 8 tables"; exit 1; }
        IQ="-DKCOL_BENCH_IQ -I build"
    fi
    [ -n "$FMT" ] && IQ="$IQ -DKCOL_BENCH_FMT"
    nvcc -O3 -std=c++17 -use_fast_math $IQ \
        -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_120,code=sm_120 \
        -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" \
        -Xptxas -v -o build/mmvq_kcol src/kcol_main.cu 2> build/ptxas_kcol.log \
        || { cat build/ptxas_kcol.log; exit 1; }
    cuobjdump -res-usage build/mmvq_kcol > build/res-usage_kcol.txt
    {
        echo "built: $(date -Is) with $(nvcc --version | tail -1)${IQ:+, with:$IQ}"
        echo "avx (ymm) instructions in the binary: $(objdump -d build/mmvq_kcol | grep -c ymm || true)"
        echo "highest glibc symbol: $(objdump -T build/mmvq_kcol | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
    } > build/BUILDINFO_kcol.txt
    cat build/BUILDINFO_kcol.txt
'
