#!/usr/bin/env bash
# Builds build/multitoken_bitid: verify/multitoken_bitid.cpp linked against the static ggml libraries of a llama.cpp
# test build made by scripts/build_llamacpp_tests_docker.sh (fork branch pascal-p106-v2-multicol or later). The binary
# uses the CUDA runtime libraries that script copies out of the image.
#
# Usage: verify/build_multitoken_bitid.sh <llama.cpp source dir> <out dir of build_llamacpp_tests_docker.sh>
# The out dir can also be the source dir of a scripts/build_llamacpp_docker.sh run (its build/ holds the same static
# libraries; the binary then links what that rig tarball holds). DOCKER_ARGS: extra `docker run` arguments, e.g. a label
# the build machine's policy asks for.
# Then:
#   LD_LIBRARY_PATH=<out dir>/cuda12-libs GGML_CUDA_MMVQ_PASCAL=2 bench/mmvq_pascal/build/multitoken_bitid
#   LD_LIBRARY_PATH=<out dir>/cuda12-libs CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2 bench/mmvq_pascal/build/multitoken_bitid
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(realpath "$1")
OUT=$(realpath "$2")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -v "$SRC:/llama:ro" -v "$OUT/build:/ggml-build:ro" -w /w "$IMAGE" bash -euo pipefail -c "
    L=/ggml-build/ggml/src
    g++ -O2 -std=c++17 -I/llama/ggml/include verify/multitoken_bitid.cpp -o build/multitoken_bitid \
        -Wl,--start-group \$L/libggml.a \$L/libggml-base.a \$L/libggml-cpu.a \$L/ggml-cuda/libggml-cuda.a -Wl,--end-group \
        -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lcuda -fopenmp -ldl -lpthread -lrt
    echo built build/multitoken_bitid
"
