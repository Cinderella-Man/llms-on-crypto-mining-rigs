#!/usr/bin/env bash
# Builds build/fasp_check (src/fasp_check.cu) inside the CUDA 12.9 container: ggml's CPU backend from a Miner Llama
# source tree (static, no AVX, so the binary also runs on the rig's Celeron) as the reference, and the sparse decode
# attention kernel from the same tree's ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh, as sm_61 machine code plus
# compute_61 PTX (CUDA_FORCE_PTX_JIT=1 runs the Pascal code on a newer GPU). The source is mounted read-only.
#
# Usage: SRC=<llama.cpp tree> bench/fattn_sparse/build.sh [name [include dir]]
#   name         output build/<name> (default fasp_check)
#   include dir  a directory searched before the tree's ggml-cuda/ (a mutated copy of the header, see check.sh)
# Env IMAGE (container image), DOCKER_ARGS (extra docker run arguments), JOBS (default 6).
set -euo pipefail
cd "$(dirname "$0")"
SRC=$(realpath "${SRC:?SRC=<llama.cpp source tree>}")
NAME=${1:-fasp_check}
INC=${2:-}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
docker run --rm ${DOCKER_ARGS:-} -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e JOBS="${JOBS:-6}" -e NAME="$NAME" \
    -e INC="$INC" -v "$SRC:/src:ro" -v "$PWD:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    if [ ! -f build/ggml/ggml/src/libggml-cpu.a ]; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq cmake > /dev/null
        cmake -S /src -B build/ggml -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
            -DGGML_BMI2=OFF -DGGML_AVX512=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF > build/ggml-cmake.log
        cmake --build build/ggml -j "$JOBS" --target ggml-cpu ggml-base > build/ggml-build.log
    fi
    inc=""
    [ -n "$INC" ] && inc="-I$INC"
    nvcc -std=c++17 -O2 -gencode arch=compute_61,code=[sm_61,compute_61] -Wno-deprecated-gpu-targets -cudart static \
        -Xcompiler "-march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" -Xptxas -v \
        $inc -I/src/ggml/src/ggml-cuda -I/src/ggml/include -o "build/$NAME" src/fasp_check.cu \
        build/ggml/ggml/src/libggml-cpu.a build/ggml/ggml/src/libggml-base.a -lpthread -ldl -lm 2> "build/ptxas_$NAME.log" \
        || { cat "build/ptxas_$NAME.log"; exit 1; }
    grep -E "registers|stack frame" "build/ptxas_$NAME.log" | sed -n "1,8p"
    echo "avx (ymm) instructions in build/$NAME: $(objdump -d "build/$NAME" | grep -c ymm || true)"
    chown -R "$HOST_UID:$HOST_GID" build
'
