#!/usr/bin/env bash
# Builds and runs verify/iq_unit_check (the decoding of IQ2_XXS, IQ2_XS and IQ3_S in the fork's Pascal kernels against
# ggml's own dequantization) in the CUDA 12.9 container, on the CPU: no GPU needed. The kernel header and ggml come
# from the llama.cpp tree given (Miner Llama, branch pascal-p106-v3-b12076-iq2xs or later), mounted read-only; ggml's
# base library is built from it into build/iq_ggml (CPU only, a minute or two) unless GGML_LIB names a libggml-base.a
# of the same tree (a build folder of scripts/build_llamacpp_docker.sh has one in build/ggml/src/).
#
# Usage: bench/mmvq_pascal/verify/iq_unit_check.sh <llama.cpp tree> [trials, default 200]
# Env: MUTATE=1 checks the check: it compiles a copy of the header with two deliberate faults (IQ2_XXS's scale taken one
#      bit too low, IQ3_S's second grid index given its partner's ninth bit), which must fail exactly iq2_xxs and
#      iq3_s and leave iq2_xs passing; the script exits 0 only then.
#      JOBS (default 4), IMAGE, DOCKER_ARGS as for build.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(realpath "${1:?usage: iq_unit_check.sh <llama.cpp tree> [trials]}")
TRIALS=${2:-200}
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
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq cmake > /dev/null
        cmake -S /src -B build/iq_ggml -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_SERVER=OFF > build/iq_ggml.cmake.log
        cmake --build build/iq_ggml -j "$JOBS" --target ggml-base > build/iq_ggml.build.log
        LIB=build/iq_ggml/ggml/src/libggml-base.a
    fi
    H=build/iq_hdr
    rm -rf $H && mkdir -p $H && cp /src/ggml/src/ggml-cuda/mmvq-pascal-iq.cuh $H/
    OUTB=build/iq_unit_check
    if [ "$MUTATE" = 1 ]; then
        sed -i -e "s/(2\*(int) (aux >> 28) + 1)/(2*(int) ((aux >> 27) \& 0xF) + 1)/" \
               -e "s/(((qh >> (2\*l + 1)) & 1) << 8)/(((qh >> (2*l)) \& 1) << 8)/" $H/mmvq-pascal-iq.cuh
        [ "$(diff /src/ggml/src/ggml-cuda/mmvq-pascal-iq.cuh $H/mmvq-pascal-iq.cuh | grep -c "^>")" = 2 ] \
            || { echo "the mutations did not apply"; exit 3; }
        OUTB=build/iq_unit_check_mut
    fi
    nvcc -std=c++17 -O2 -arch=sm_61 -Wno-deprecated-gpu-targets -I $H -I /src/ggml/include -I /src/ggml/src \
        -o $OUTB verify/iq_unit_check.cu "$LIB" -Xcompiler -fopenmp -lpthread -lm -ldl
    chown -R "$HOST_UID:$HOST_GID" build
    set +e
    $OUTB "$TRIALS" | tee $OUTB.txt
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$MUTATE" = 1 ]; then
        if grep -q "^iq2_xxs .*FAIL" $OUTB.txt && grep -q "^iq3_s .*FAIL" $OUTB.txt && grep -q "^iq2_xs .*PASS" $OUTB.txt; then
            echo "mutation: fails exactly iq2_xxs and iq3_s, as it must"
            exit 0
        fi
        echo "mutation: NOT caught as expected"
        exit 4
    fi
    exit $rc
'
