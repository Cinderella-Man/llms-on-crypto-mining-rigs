#!/usr/bin/env bash
# Builds and runs verify/kcol_unit_check (the k-column products' decoding against ggml's reference dequantization) in
# the CUDA 12.9 container, on the CPU: no GPU needed. Argument: trials per type and shape (default 40).
# Usage: bench/mmvq_pascal/verify/kcol_unit_check.sh [trials]   (env IMAGE, DOCKER_ARGS as for build.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p build
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w "$IMAGE" bash -euo pipefail -c '
    nvcc -std=c++17 -O2 -arch=sm_61 -Wno-deprecated-gpu-targets -o build/kcol_unit_check verify/kcol_unit_check.cu
    build/kcol_unit_check "$@"
' _ "$@"
