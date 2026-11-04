#!/usr/bin/env bash
# Builds and runs verify/iq_kcol_check (the decoding of the grid types and IQ1_M in the fork's k-column and multi-token
# Pascal kernels, Q713, against ggml's own dequantization) in the CUDA 12.9 container, on the CPU: no GPU needed. The
# kernel headers and ggml come from the llama.cpp tree given (Miner Llama, branch pascal-p106-v3-b12076-iqkcol or
# later), mounted read-only; ggml's base library is built from it into build/iqk_ggml (CPU only, a minute or two)
# unless GGML_LIB names a libggml-base.a of the same tree.
#
# Usage: bench/mmvq_pascal/verify/iq_kcol_check.sh <llama.cpp tree> [trials, default 100]
# Env: MUTATE=1 checks the check: it compiles a copy of mmvq-pascal-iq.cuh with three deliberate faults in decode32
#      (IQ2_S's grid index without its second high bit, IQ3_S's second index given its partner's ninth bit, IQ1_M's
#      negative-delta table entry for grid value +1 off by 2), which must fail exactly iq2_s, iq3_s and iq1_m and leave
#      the other three types passing; the script exits 0 only then.
#      JOBS (default 4), IMAGE, DOCKER_ARGS as for build.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(realpath "${1:?usage: iq_kcol_check.sh <llama.cpp tree> [trials]}")
TRIALS=${2:-100}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LIBMNT=()
if [ -n "${GGML_LIB:-}" ]; then
    LIBMNT=(-v "$(realpath "$GGML_LIB"):/lib/libggml-base.a:ro")
fi
mkdir -p build
docker run --rm ${DOCKER_ARGS:-} -v "$SRC:/src:ro" -v "$PWD:/w" "${LIBMNT[@]}" -w /w \
    -e TRIALS="$TRIALS" -e MUTATE="${MUTATE:-0}" -e JOBS="${JOBS:-4}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$IMAGE" bash -euo pipefail -c '
    LIB=/lib/libggml-base.a
    if [ ! -f "$LIB" ]; then
        LIB=build/iqk_ggml/ggml/src/libggml-base.a
        if [ ! -f "$LIB" ]; then
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq cmake > /dev/null
            cmake -S /src -B build/iqk_ggml -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
                -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
                -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF > build/iqk_ggml.cmake.log
            cmake --build build/iqk_ggml -j "$JOBS" --target ggml-base > build/iqk_ggml.build.log
        fi
    fi
    H=build/iqk_hdr
    rm -rf $H && mkdir -p $H
    cp /src/ggml/src/ggml-cuda/mmvq-pascal-iq.cuh /src/ggml/src/ggml-cuda/mmvq-pascal-kcol-impl.cuh $H/
    { echo "namespace gpu_tab {"
      awk "/GGML_TABLE_BEGIN\\(uint32_t, iq1s_grid_gpu, NGRID_IQ1S\\)/ { p = 1 } p { print } /GGML_TABLE_END/ { p = 0 }" \
          /src/ggml/src/ggml-common.h | sed -E "s/GGML_TABLE_BEGIN\\(uint32_t, iq1s_grid_gpu, NGRID_IQ1S\\)/static const uint32_t iq1s_grid_gpu[NGRID_IQ1S] = {/; s/GGML_TABLE_END\\(\\)/};/"
      echo "} // namespace gpu_tab"; } > $H/iq1s_grid_gpu.h
    [ "$(grep -c "^static const uint32_t iq1s_grid_gpu" $H/iq1s_grid_gpu.h)" = 1 ] || { echo "iq1s_grid_gpu not found"; exit 3; }
    OUTB=build/iq_kcol_check
    if [ "$MUTATE" = 1 ]; then
        perl -0pi -e "s/\(\(\(qh >> \(2\*l\)\) & 3\) << 8\), g0, g1\);/(((qh >> (2*l)) \& 1) << 8), g0, g1);/ or die 1;
                      s/\(\(qq >> 8\) & 0xff\) \| \(\(\(qh >> \(2\*l \+ 1\)\) & 1\) << 8\)/((qq >> 8) \& 0xff) | (((qh >> (2*l)) \& 1) << 8)/ or die 2;
                      s/t_neg = 0x0007FFF7u/t_neg = 0x0009FFF7u/ or die 3" $H/mmvq-pascal-iq.cuh \
            || { echo "the mutations did not apply"; exit 3; }
        [ "$(diff /src/ggml/src/ggml-cuda/mmvq-pascal-iq.cuh $H/mmvq-pascal-iq.cuh | grep -c "^>")" = 3 ] \
            || { echo "the mutations did not apply"; exit 3; }
        OUTB=build/iq_kcol_check_mut
    fi
    nvcc -std=c++17 -O2 -arch=sm_61 -Wno-deprecated-gpu-targets -I $H -I /src/ggml/include -I /src/ggml/src \
        -o $OUTB verify/iq_kcol_check.cu "$LIB" -Xcompiler -fopenmp -lpthread -lm -ldl
    chown -R "$HOST_UID:$HOST_GID" build
    set +e
    $OUTB "$TRIALS" | tee $OUTB.txt
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$MUTATE" = 1 ]; then
        if grep -q "^iq2_s .*FAIL" $OUTB.txt && grep -q "^iq3_s .*FAIL" $OUTB.txt && grep -q "^iq1_m .*FAIL" $OUTB.txt &&
                grep -q "^iq2_xxs .*PASS" $OUTB.txt && grep -q "^iq2_xs .*PASS" $OUTB.txt && grep -q "^iq3_xxs .*PASS" $OUTB.txt; then
            echo "mutation: fails exactly iq2_s, iq3_s and iq1_m, as it must"
            exit 0
        fi
        echo "mutation: NOT caught as expected"
        exit 4
    fi
    exit $rc
'
