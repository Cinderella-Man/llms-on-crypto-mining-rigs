#!/usr/bin/env bash
# CPU-only llama-server of the v2.0 engine (fork commit 37e717ff6, unpacked from `git archive` into $SRC, default
# $WORK/src), used only for its /apply-template and /tokenize endpoints: the same chat-template and tokenizer code the
# rig's server ran. The same script builds the fix branch (pascal-p106-v3-promptpath) for the CPU check:
# SRC=$WORK/src-pp OUT=$WORK/out-cpu-pp. Docker, low CPU share, -j 8, labelled container.
# No llamafile GEMM: its tiling differs from the one-token path, and the check compares warm and cold answers.
# TESTS=1 also builds test-llama-archs (random tiny models of each architecture, Q261's mask check).
# Output: $OUT/build/bin/llama-server (OUT default $WORK/out-cpu).
#   WORK=/path/to/workdir bash build_cpu.sh
set -euo pipefail
: "${WORK:?set WORK}"
SRC=${SRC:-$WORK/src}; OUT=${OUT:-$WORK/out-cpu}
if [ "${TESTS:-0}" = 1 ]; then T="-DLLAMA_BUILD_TESTS=ON"; TARGETS="llama-server test-llama-archs"; else T="-DLLAMA_BUILD_TESTS=OFF"; TARGETS=llama-server; fi
mkdir -p "$OUT"
docker run --rm --cpu-shares=256 --label pascal-quants=1 --name pq-rereads-cpu-build-$(basename "$OUT") \
    -e T="$T" -e TARGETS="$TARGETS" -v "$SRC:/src:ro" -v "$OUT:/out" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF $T \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target $TARGETS > /out/build.log 2>&1 \
        || { grep -E "error|Error" /out/build.log | head -40; tail -20 /out/build.log; exit 1; }
    echo "built: out-cpu/build/bin/llama-server"
'
