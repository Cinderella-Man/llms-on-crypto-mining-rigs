#!/usr/bin/env bash
# Builds build/verify_<arch>: the benchmark's baseline kernels linked against a static llama.cpp/ggml CUDA build, to
# check that the copied kernels reproduce llama.cpp's outputs bit for bit. Needs a llama.cpp tree with
# patches/q4_1_64-type.diff applied (the Q4_1_64 type) and a static CUDA build of it made with the same CUDA 12.9 image.
#
# Usage: verify/build_verify.sh <llama.cpp source dir> <its build dir> [sm_120|ptx61]
#   sm_120  native code for an RTX 50-series GPU; the llama.cpp build must include sm_120 (e.g. CMAKE_CUDA_ARCHITECTURES=61;120)
#   ptx61   compute_61 PTX only, for checking the Pascal code path on a newer GPU through the driver's JIT; build
#           llama.cpp with -DCMAKE_CUDA_ARCHITECTURES=61-virtual (the README has the full command)
# A llama.cpp build made with GGML_CUDA_NCCL=ON is linked against the static NCCL from the image.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(realpath "$1")
BLD=$(realpath "$2")
ARCH=${3:-sm_120}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
case "$ARCH" in
    sm_120) GENCODE="-gencode arch=compute_120,code=sm_120" ;;
    ptx61)  GENCODE="-gencode arch=compute_61,code=compute_61" ;;
    *) echo "unknown arch $ARCH"; exit 1 ;;
esac
mkdir -p build
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -v "$SRC:/llama:ro" -v "$BLD:/ggml-build:ro" -w /w "$IMAGE" bash -euo pipefail -c "
    nvcc -O3 -std=c++17 -use_fast_math $GENCODE -Wno-deprecated-gpu-targets -c verify/verify_main.cu -o build/verify_main_$ARCH.o
    g++ -O2 -std=c++17 -I/llama/ggml/include -c verify/verify_ggml.cpp -o build/verify_ggml.o
    L=/ggml-build/ggml/src
    nvcc $GENCODE -Wno-deprecated-gpu-targets -cudart static -o build/verify_$ARCH build/verify_main_$ARCH.o build/verify_ggml.o \
        -Xlinker --start-group \$L/libggml.a \$L/libggml-base.a \$L/libggml-cpu.a \$L/ggml-cuda/libggml-cuda.a -Xlinker --end-group \
        -lcublas_static -lcublasLt_static -lnccl_static -lculibos -L/usr/local/cuda/lib64/stubs -lcuda -lgomp -ldl -lpthread -lrt
    echo built build/verify_$ARCH
"
