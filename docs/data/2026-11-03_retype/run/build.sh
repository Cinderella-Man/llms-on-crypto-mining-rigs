#!/usr/bin/env bash
# llama-quantize and libggml-base.so for scripts/gguf_retype_test.py's comparison with llama.cpp's own quantizer, from
# a llama.cpp checkout at the Miner Llama v3 base commit (which knows Flash-Next, `qwen4exp`): CPU only, shared
# libraries, built in Docker. WORK is mounted at its own path, so the binaries' RPATH also holds outside the container.
#   WORK=<build folder> SRC=<llama.cpp checkout> [JOBS=8] [DOCKER_ARGS="--label ... --cpu-shares 256"] bash build.sh
# Output: $WORK/build/bin/{llama-quantize,libggml-base.so,...} and $WORK/build/bin/BUILDINFO.txt.
set -euo pipefail
: "${WORK:?}" "${SRC:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK/ccache"
docker run --rm ${DOCKER_ARGS:-} -e JOBS="${JOBS:-8}" -e WORK="$WORK" -v "$WORK:$WORK" -v "$SRC:/src:ro" "$IMAGE" \
    bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=$WORK/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    git config --global --add safe.directory /src
    cmake -S /src -B $WORK/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
    cmake --build $WORK/build -j "$JOBS" --target llama-quantize > /dev/null
    { echo "commit: $(git -C /src rev-parse HEAD) ($(git -C /src log -1 --format=%ci))"
      echo "CPU only, BUILD_SHARED_LIBS=ON, GGML_NATIVE=ON"; echo "image: '"$IMAGE"'"; echo "built: $(date -Is)"; } \
        > $WORK/build/bin/BUILDINFO.txt
    cat $WORK/build/bin/BUILDINFO.txt
'
