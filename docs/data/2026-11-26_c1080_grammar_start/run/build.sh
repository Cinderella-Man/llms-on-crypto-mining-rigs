#!/usr/bin/env bash
# Builds run/grammar_steps.cpp into grammar-steps against a Miner Llama rig build's own build tree: its static
# libraries, in the image that built it (as docs/data/2026-11-25_c981_cuts1_flash_kld/run/build.sh does), so the tool
# runs that build's grammar code exactly.
# The tree is the work folder scripts/build_llamacpp_docker.sh used (WORK): src/ (the checkout) and src/build/
# (CMake's tree), with paths under /work, so it is mounted read-only at /work.
# Env: TREE (that work folder), OUT (folder for the binary and the build log), IMAGE (default the build script's),
#      DOCKER_ARGS (extra docker run arguments: labels, --cpu-shares), LOCK (optional flock file that serializes
#      builds on a shared machine), LOCK_WAIT (seconds, default 3600)
set -euo pipefail
: "${TREE:?}" "${OUT:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cp "$HERE/grammar_steps.cpp" "$OUT/"
run() {
    # shellcheck disable=SC2086
    docker run --rm ${DOCKER_ARGS:-} --network none -v "$TREE:/work:ro" -v "$OUT:$OUT" "$IMAGE" bash -euo pipefail -c '
        B=/work/src/build
        L=/usr/local/cuda-12.9/targets/x86_64-linux/lib
        cd "'"$OUT"'"
        c++ -O2 -DNDEBUG -fopenmp -Wall -Wextra -I/work/src/include -I/work/src/ggml/include \
            -c grammar_steps.cpp -o grammar_steps.o
        c++ -O2 -fopenmp grammar_steps.o -o grammar-steps -Wl,-rpath,$L: \
            $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
            $B/ggml/src/libggml-base.a /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.a \
            -lm -ldl $L/libcudart.so $L/libcublas.so $L/libculibos.a $L/libcublasLt.so \
            /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
        md5sum grammar_steps.cpp grammar-steps "$B/src/libllama.a" "$B/bin/llama-server"
        git -C /work/src rev-parse HEAD 2>/dev/null || true'
}
if [ -n "${LOCK:-}" ]; then
    exec 9> "$LOCK"
    flock -w "${LOCK_WAIT:-3600}" 9 || { echo "the build lock $LOCK stayed taken"; exit 1; }
fi
run 2>&1 | tee "$OUT/build.log"
