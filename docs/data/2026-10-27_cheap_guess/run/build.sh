#!/usr/bin/env bash
# Rig build of Miner Llama's branch pascal-p106-v3-cheapguess (a cheaper guess of Flint's MTP head) on a build machine
# with Docker, from a source copy of a local fork checkout (git archive of the commit, nothing pushed): compute
# capability 6.1 only, the rig build's flags (scripts/build_llamacpp_docker.sh: static, no AVX, CUDA 12.9 on Ubuntu
# 24.04), CPU only at a low share, -j 8. Prints the tarball's name and md5.
#   SRC=<fork checkout> BUILD_HOST=<ssh alias> BUILD_WORK=<folder there> [COMMIT=<ref>] [DOCKER_ARGS=<e.g. a label>] bash build.sh
set -euo pipefail
SRC=${SRC:?local fork checkout}
C=${COMMIT:-pascal-p106-v3-cheapguess}
H=${BUILD_HOST:?ssh alias of the build machine}
D=${BUILD_WORK:?folder on the build machine}
SHA=$(git -C "$SRC" rev-parse --short=9 "$C")
NAME=llama.cpp-$(date +%Y%m%d)-$SHA-pascal-p106-v3-cheapguess
ssh "$H" "rm -rf $D/src && mkdir -p $D/src $D/out/ccache"
git -C "$SRC" archive --format=tar "$C" | ssh "$H" "tar xf - -C $D/src && echo $(git -C "$SRC" rev-parse "$C") > $D/src/COMMIT"
ssh "$H" "NAME=$NAME D=$D DOCKER_ARGS='${DOCKER_ARGS:-}' bash -s" <<'EOF'
set -euo pipefail
docker run --rm --cpu-shares=256 $DOCKER_ARGS --name cheapguess-build -e NAME="$NAME" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -v "$D/src:/src:ro" -v "$D/out:/out" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/out/ccache
    apt-get update -qq && apt-get install -y -qq cmake ccache git binutils > /dev/null
    cmake -S /src -B /out/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61" -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
        -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        > /out/cmake.log 2>&1 || { tail -40 /out/cmake.log; exit 1; }
    cmake --build /out/build -j 8 --target llama-server llama-pipespec llama-pipe-drafter-check > /out/build.log 2>&1 \
        || { grep -E "error" /out/build.log | head -40; tail -20 /out/build.log; exit 1; }
    out=/out/$NAME
    rm -rf "$out" && mkdir -p "$out/bin"
    cp /out/build/bin/llama-server /out/build/bin/llama-pipespec /out/build/bin/llama-pipe-drafter-check "$out/bin/"
    {
        echo "name: $NAME"
        echo "ref: pascal-p106-v3-cheapguess (source copy)"
        echo "commit: $(cat /src/COMMIT)"
        echo "image: nvidia/cuda:12.9.2-devel-ubuntu24.04"
        echo "cuda architectures: 61"
        echo "built: $(date -Is)"
        echo "avx instructions in llama-server: $(objdump -d /out/build/bin/llama-server | grep -c ymm)"
        echo "highest glibc symbol: $(objdump -T /out/build/bin/llama-server | grep -o "GLIBC_[0-9.]*" | sort -V | tail -1)"
    } > "$out/BUILDINFO.txt"
    tar czf "/out/$NAME.tar.gz" -C /out "$NAME"
    md5sum "/out/$NAME.tar.gz"
    chown -R "$HOST_UID:$HOST_GID" /out
    cat "$out/BUILDINFO.txt"
'
EOF
