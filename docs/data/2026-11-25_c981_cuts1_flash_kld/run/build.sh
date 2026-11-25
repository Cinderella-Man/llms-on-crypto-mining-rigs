#!/usr/bin/env bash
# Builds run/kld_cuts.cpp into kld-cuts against a Miner Llama rig build's own build tree: the static libraries and the
# compile and link lines of its llama-perplexity, in the image that built it. The tool then runs exactly the engine
# code of that build's tarball (check: the tree's build/bin/llama-perplexity has the tarball's md5).
# The tree is the work folder scripts/build_llamacpp_docker.sh used (WORK): it holds src/ (the checkout) and
# src/build/ (CMake's tree), with paths under /work, so it is mounted read-only at /work.
# Env: TREE (that work folder), OUT (folder for the binary and the build log), IMAGE (default the build script's),
#      DOCKER_ARGS (extra docker run arguments: labels, --cpu-shares), LOCK (optional flock file that serializes
#      builds on a shared machine)
set -euo pipefail
: "${TREE:?}" "${OUT:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cp "$HERE/kld_cuts.cpp" "$OUT/kld_cuts.cpp"
run() {
    # shellcheck disable=SC2086
    docker run --rm ${DOCKER_ARGS:-} --network none -v "$TREE:/work:ro" -v "$OUT:$OUT" "$IMAGE" bash -euo pipefail -c '
        B=/work/src/build
        L=/usr/local/cuda-12.9/targets/x86_64-linux/lib
        cd "'"$OUT"'"
        c++ -O3 -DNDEBUG -fopenmp -Wall -Wextra -DGGML_USE_CPU -DGGML_USE_CUDA \
            -I/work/src/common -I/work/src/vendor -I/work/src/include -I/work/src/ggml/include \
            -c kld_cuts.cpp -o kld_cuts.o
        c++ -O3 -DNDEBUG -fopenmp kld_cuts.o -o kld-cuts -Wl,-rpath,$L: \
            $B/common/libllama-common.a $B/common/libllama-common-base.a $B/vendor/cpp-httplib/libcpp-httplib.a \
            $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
            $B/ggml/src/libggml-base.a /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a \
            -lm -ldl $L/libcudart.so $L/libcublas.so $L/libculibos.a $L/libcublasLt.so \
            /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
        md5sum kld_cuts.cpp kld-cuts "$B/bin/llama-perplexity" "$B/src/libllama.a"
        git -C /work/src rev-parse HEAD 2>/dev/null || true'
}
if [ -n "${LOCK:-}" ]; then
    exec 9> "$LOCK"
    flock -w "${LOCK_WAIT:-3600}" 9 || { echo "the build lock $LOCK stayed taken"; exit 1; }
fi
run 2>&1 | tee "$OUT/build.log"
