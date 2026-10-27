#!/usr/bin/env bash
# Builds test-backend-ops once per tuning variant of the grid types' Pascal kernels, on the build machine, from the
# source copy in $WORK/src. The switches are compile-time defaults in ggml/src/ggml-cuda/mmvq-pascal-impl.cuh; a
# variant name sets them: lut<0|1> (MMVQP_IQ_SIGN_LUT: sign masks from a shared-memory table or computed in registers),
# mb<n> (MMVQP_IQ_MIN_BLOCKS: blocks per SM the register limit is set for), r<n> (MMVQP_IQ_R: rows per lane group), and
# an optional p<n> (MMVQP_IQ_PERSIST: one-token kernels with n blocks per SM looping over the row blocks). Each variant
# edits a copy of the header, so only the two files that include it recompile. Prints each variant's registers and
# stack bytes per kernel for compute 6.1.
#   ssh <build machine> "WORK=<folder> VARIANTS='lut1-mb1-r1 lut0-mb3-r1-p3' bash -s" < run/tune_build.sh
# Output: $WORK/out/tune/<variant>/test-backend-ops and $WORK/out/tune.tar.gz (all variants), md5 printed. FRESH=0
# keeps the variants built earlier.
set -euo pipefail
WORK=${WORK:?folder holding src/}
VARIANTS=${VARIANTS:?variant names}
LABEL=${LABEL:-pascal-quants=1}
[ "${FRESH:-1}" = 1 ] && rm -rf "$WORK/out/tune"; mkdir -p "$WORK/out/tune" "$WORK/out/ccache"
docker run --rm --cpu-shares=256 --label "$LABEL" --name v3iq-tune -e VARIANTS="$VARIANTS" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -v "$WORK/src:/src0:ro" -v "$WORK/out:/out" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git binutils > /dev/null
    rm -rf /out/src-tune && cp -a /src0 /out/src-tune
    H=/out/src-tune/ggml/src/ggml-cuda/mmvq-pascal-impl.cuh
    cmake -S /out/src-tune -B /out/build-tune -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70-real" -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake-tune.log 2>&1 || { tail -20 /out/cmake-tune.log; exit 1; }
    for v in $VARIANTS; do
        lut=$(echo $v | sed -E "s/lut([0-9]+)-.*/\1/"); mb=$(echo $v | sed -E "s/.*-mb([0-9]+)-.*/\1/")
        r=$(echo $v | sed -E "s/.*-r([0-9]+).*/\1/"); ps=$(echo $v | sed -nE "s/.*-p([0-9]+)$/\1/p"); ps=${ps:-0}
        sed -i -E "s/^#define MMVQP_IQ_SIGN_LUT [0-9]+/#define MMVQP_IQ_SIGN_LUT $lut/; s/^#define MMVQP_IQ_MIN_BLOCKS [0-9]+/#define MMVQP_IQ_MIN_BLOCKS $mb/; s/^#define MMVQP_IQ_R [0-9]+/#define MMVQP_IQ_R $r/; s/^#define MMVQP_IQ_PERSIST [0-9]+/#define MMVQP_IQ_PERSIST $ps/" $H
        grep -E "^#define MMVQP_IQ_(SIGN_LUT|MIN_BLOCKS|R|PERSIST) " $H | tr "\n" " "; echo
        t0=$(date +%s)
        cmake --build /out/build-tune -j 8 --target test-backend-ops > /out/build-tune-$v.log 2>&1 \
            || { grep -E "error" /out/build-tune-$v.log | head -20; exit 1; }
        mkdir -p /out/tune/$v && cp /out/build-tune/bin/test-backend-ops /out/tune/$v/
        echo "$v built in $(( $(date +%s) - t0 )) s"
        cuobjdump -res-usage -arch sm_61 /out/tune/$v/test-backend-ops 2>/dev/null \
            | grep -A1 "mul_mat_vec_q_pascalIL9ggml_type2[2]E\|mul_mat_vec_q_pascalIL9ggml_type18E" \
            | paste - - | grep -oE "ggml_type(22|18)E[^v]*EEEv|REG:[0-9]+|STACK:[0-9]+" | paste - - - | sed -E "s/EEEv//" \
            | sort -u | tr "\n" ";"; echo
    done
    cd /out && tar czf tune.tar.gz tune && md5sum tune.tar.gz
    chown -R "$HOST_UID:$HOST_GID" /out/tune /out/tune.tar.gz /out/build-tune /out/src-tune
'
