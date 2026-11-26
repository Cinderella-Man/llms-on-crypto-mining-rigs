#!/usr/bin/env bash
# C1179: configure and build test-backend-ops of a Miner Llama tree natively (no Docker) on a box with one CUDA GPU.
# The CPU flags are the rig tarball's (scripts/build_llamacpp_docker.sh), so the CPU reference is the rig build's code.
#
#   SRC=<source tree> BUILD=<build folder> ARCHS="61-virtual;70" bash build.sh
#
# ARCHS: CMAKE_CUDA_ARCHITECTURES. "61-virtual;70" builds a V100's sm_70 machine code plus the P106's compute-6.1 PTX;
#   "61-virtual" builds the PTX only, so a newer card runs the P106's code (JIT-compiled by the driver).
# FORCE_MMQ=1: -DGGML_CUDA_FORCE_MMQ=ON (MMQ for every product size, as a P106 picks it; a V100 would hand the
#   non-expert products of 64 or more tokens to cuBLAS).
# CUDA: the toolkit folder (default /usr/local/cuda). JOBS (default 8). TARGETS (default test-backend-ops).
# Set CCACHE_DIR to keep the compiler cache next to the build.
set -euo pipefail
: "${SRC:?}" "${BUILD:?}" "${ARCHS:?}"
CUDA=${CUDA:-/usr/local/cuda}
export PATH="$CUDA/bin:$PATH"

force_mmq=()
if [ "${FORCE_MMQ:-0}" = 1 ]; then
    force_mmq=(-DGGML_CUDA_FORCE_MMQ=ON)
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
    -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON "${force_mmq[@]}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null

# shellcheck disable=SC2086
nice -n 19 cmake --build "$BUILD" -j "${JOBS:-8}" --target ${TARGETS:-test-backend-ops}
