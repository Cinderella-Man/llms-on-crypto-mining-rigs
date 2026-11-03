#!/usr/bin/env bash
# Q550 negative control: the test build (build_test.sh) with neg_qwen4exp_conv_rows.diff reversed, so qwen4exp's two
# conv rows (the delta-net conv state and the n-gram conv history) go to group 0 again while llama_pipe_init accepts the
# model. check_identity.sh must then report DIFF: the next evaluation reads the conv rows from the slot's group, which
# the slot never wrote, so the checks can tell a state the groups do not keep. llama-pipespec only, into out-neg/
# (ccache shared with out/).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf src-neg && cp -a src src-neg
(cd src-neg && patch -s -R -p1 < ../neg_qwen4exp_conv_rows.diff)
echo "get_grp in qwen4exp.cpp: $(grep -c get_grp src-neg/src/models/qwen4exp.cpp || true) (0: the conv rows write group 0)"
mkdir -p out-neg
docker run --rm --cpu-shares=256 --name pq-build-negative -v "$PWD/src-neg:/src:ro" \
    -v "$PWD/out-neg:/out" -v "$PWD/out/ccache:/ccache" -e ARCHS="61-virtual;86-real" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target llama-pipespec > /out/build.log 2>&1 \
        || { grep -E "error|Error" /out/build.log | head -40; tail -20 /out/build.log; exit 1; }
    echo "built: out-neg/build/bin"
'
rm -rf bin-neg && cp -a out-neg/build/bin bin-neg
