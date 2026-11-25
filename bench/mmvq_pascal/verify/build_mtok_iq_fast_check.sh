#!/usr/bin/env bash
# Builds build/mtok_iq_fast_check: verify/mtok_iq_fast_check.cpp linked against the static ggml libraries of a llama.cpp
# build tree (scripts/build_llamacpp_docker.sh's $WORK/src/build, or any tree configured the same way), in the CUDA 12.9
# container. CPU only.
#
# Usage: verify/build_mtok_iq_fast_check.sh <llama.cpp source dir> <its build dir> [out dir, default build/]
# DOCKER_ARGS: extra `docker run` arguments, e.g. a label the build machine's policy asks for.
# Then, on a card (scripts/pascal_forced.sh for a newer GPU):
#   env $PASCAL_FORCED GGML_CUDA_MTOK_IQ_FAST=0 build/mtok_iq_fast_check off.bin > off.txt
#   env $PASCAL_FORCED GGML_CUDA_MTOK_IQ_FAST=1 build/mtok_iq_fast_check on.bin > on.txt
#   cmp off.bin on.bin && diff off.txt on.txt
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(realpath "$1")
BLD=$(realpath "$2")
OUT=$(realpath -m "${3:-build}")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w:ro" -v "$SRC:/llama:ro" -v "$BLD:/ggml-build:ro" \
    -v "$OUT:/out" -w /w "$IMAGE" bash -euo pipefail -c "
    L=/ggml-build/ggml/src
    g++ -O2 -std=c++17 -I/llama/ggml/include verify/mtok_iq_fast_check.cpp -o /out/mtok_iq_fast_check \
        -Wl,--start-group \$L/libggml.a \$L/libggml-base.a \$L/libggml-cpu.a \$L/ggml-cuda/libggml-cuda.a -Wl,--end-group \
        -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lcuda \
        -fopenmp -ldl -lpthread -lrt
    echo built /out/mtok_iq_fast_check
"
