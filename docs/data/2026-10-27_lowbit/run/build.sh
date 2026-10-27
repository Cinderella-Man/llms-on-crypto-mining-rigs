#!/usr/bin/env bash
# Builds of Miner Llama's fork branch pascal-p106-v3-lowbit (N2P/N3P) on the quality machine, from a local checkout of
# the fork: a git-archive copy of COMMIT goes to QM, then Docker builds there at --cpu-shares 256 with -j 8.
#   KIND=cuda (default): static, no AVX (the rig's Celeron), CUDA 12.9, code for compute 6.1 plus compute_61 PTX and
#     sm_70 (a V100 runs test-backend-ops natively and through the compute 6.1 code); a tarball WORK/out/NAME.tar.gz
#     with llama-*, test-backend-ops, test-quantize-fns and BUILDINFO.txt, the one the rig installs
#   KIND=cpu: shared libraries without CUDA in WORK/build-n-cpu (llama-quantize, test-quantize-fns, libggml-base for
#     scripts/lowbit_proxy.py and run/check_n2p_n3p.py)
#   QM=<ssh alias> WORK=<work dir on QM> FORK=<local fork checkout> COMMIT=<sha> [NAME=...] [KIND=cuda|cpu] bash build.sh
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?work dir on QM}
FORK=${FORK:?local checkout of the fork}
COMMIT=${COMMIT:?commit of pascal-p106-v3-lowbit}
KIND=${KIND:-cuda}
NAME=${NAME:-llama.cpp-$(date +%Y%m%d)-${COMMIT:0:9}-pascal-p106-v3-lowbit}

(cd "$FORK" && git archive "$COMMIT") | ssh "$QM" "rm -rf $WORK/src && mkdir -p $WORK/src $WORK/out/ccache && tar -x -C $WORK/src && echo $COMMIT > $WORK/src/COMMIT"

ssh "$QM" "docker run -i --rm --cpu-shares=256 --label pascal-quants=1 -e NAME=$NAME -e KIND=$KIND \
    -v $WORK/src:/src:ro -v $WORK:/w nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -s" <<'EOF'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/w/out/ccache
apt-get update -qq && apt-get install -y -qq cmake ccache git binutils > /dev/null
if [ "$KIND" = cpu ]; then
    cmake -S /src -B /w/build-n-cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /w/build-n-cpu.cmake.log 2>&1
    cmake --build /w/build-n-cpu -j 8 --target test-quantize-fns llama-quantize ggml-base ggml-cpu > /w/build-n-cpu.log 2>&1
    exit 0
fi
cmake -S /src -B /w/out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;70-real" -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
    -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    > /w/out/cmake.log 2>&1 || { tail -40 /w/out/cmake.log; exit 1; }
cmake --build /w/out/build -j 8 --target test-backend-ops test-quantize-fns llama-quantize llama-imatrix llama-server \
    llama-bench llama-cli llama-perplexity > /w/out/build.log 2>&1 \
    || { grep -E "error" /w/out/build.log | head -40; tail -20 /w/out/build.log; exit 1; }
out=/w/out/$NAME
rm -rf "$out" && mkdir -p "$out/bin"
cp /w/out/build/bin/llama-* /w/out/build/bin/test-backend-ops /w/out/build/bin/test-quantize-fns "$out/bin/"
{
    echo "name: $NAME"
    echo "ref: pascal-p106-v3-lowbit (source copy)"
    echo "commit: $(cat /src/COMMIT)"
    echo "image: nvidia/cuda:12.9.2-devel-ubuntu24.04"
    echo "cuda architectures: 61;70-real"
    echo "built: $(date -Is)"
    echo "avx instructions in llama-bench: $(objdump -d /w/out/build/bin/llama-bench | grep -c ymm)"
} > "$out/BUILDINFO.txt"
tar czf "/w/out/$NAME.tar.gz" -C /w/out "$NAME"
md5sum "/w/out/$NAME.tar.gz"
cat "$out/BUILDINFO.txt"
EOF
