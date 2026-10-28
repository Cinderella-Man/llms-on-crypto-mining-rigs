#!/usr/bin/env bash
# Builds run/mtp_accept.cpp against an existing llama.cpp CUDA build (its static libraries), in the CUDA image, so the
# tool runs exactly that build's MTP graph and kernels. CPU only, at a low CPU share.
#   LLAMA=<llama.cpp checkout> BUILD=<its CUDA build folder, e.g. build-cuda> OUT=<binary path> bash build_tool.sh
set -euo pipefail
: "${LLAMA:?}" "${BUILD:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
B=$LLAMA/$BUILD
mkdir -p "$(dirname "$OUT")"
docker run --rm --label "$LABEL" --cpu-shares 256 -v "$LLAMA:$LLAMA:ro" -v "$HERE:/tool:ro" \
    -v "$(dirname "$OUT"):$(dirname "$OUT")" "$IMAGE" bash -euo pipefail -c "
    apt-get update -qq > /dev/null && apt-get install -y -qq libssl-dev > /dev/null
    nice -n 19 g++ -O2 -std=c++17 -fopenmp /tool/mtp_accept.cpp -o '$OUT' \
        -I'$LLAMA/include' -I'$LLAMA/common' -I'$LLAMA/src' -I'$LLAMA/ggml/include' \
        '$B/common/libllama-common.a' '$B/common/libllama-common-base.a' '$B/vendor/cpp-httplib/libcpp-httplib.a' \
        '$B/src/libllama.a' '$B/ggml/src/libggml.a' -ldl '$B/ggml/src/libggml-cpu.a' \
        '$B/ggml/src/ggml-cuda/libggml-cuda.a' '$B/ggml/src/libggml-base.a' -L/usr/local/cuda/lib64 \
        -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcublasLt /usr/local/cuda/lib64/libculibos.a -lcuda -lnccl \
        -lssl -lcrypto -lpthread -lm"
ls -l "$OUT"
