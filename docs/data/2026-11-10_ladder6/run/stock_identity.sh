#!/usr/bin/env bash
# C662 (Q1150): the merged branch against stock upstream on a CPU. test-llama-archs builds a random tiny model of every
# architecture and evaluates it; the fork's version of the test can write each model's logits to a file and compare a
# later run against it (LLAMA_ARCHS_LOGITS). The fork's test file uses no API that stock llama.cpp lacks, so it is
# built against the unmodified upstream release the branch merged: stock's build writes the logits, the branch's CPU
# build compares. With every switch at its default the branch must give stock's logits bit for bit on every
# architecture; a difference would be a fork behaviour that changes what a CPU computes.
#   REVERT='<upstream commit> ...' builds the branch's CPU test with those upstream commits reverted (a test tree only)
#   and compares it against BASE_BIN's logits: which differences from an older build a single upstream change explains.
# Usage: WORK=<dir> SRC=<llama.cpp clone holding both commits> UPSTREAM=<upstream tag> BRANCH_REF=<the branch's commit>
#   BRANCH_BIN=<bin/ of the branch's static native CPU build> [REVERT='<commit> ...' BASE_BIN=<bin/ of the older CPU
#   build>] [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] bash stock_identity.sh
# Output: $WORK/stock/bin/test-llama-archs, $WORK/{stock,merged}.log, $WORK/differ.txt, $WORK/verdict.txt (and
#   $WORK/revert/... with REVERT); done when $WORK/done exists
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${UPSTREAM:?UPSTREAM=<upstream tag>}" "${BRANCH_REF:?BRANCH_REF=<branch commit>}" "${BRANCH_BIN:?BRANCH_BIN=<branch CPU bin/>}"
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/ccache"
rm -f "$WORK/done"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
# <name>: a static native CPU build of test-llama-archs from $WORK/<name>/src into $WORK/<name>/bin
build() {
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e N="$1" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S /work/$N/src -B /work/$N/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/$N/cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/$N/build -j "$JOBS" --target test-llama-archs > /work/$N/build.log 2>&1 || rc=$?
        [ $rc = 0 ] && mkdir -p /work/$N/bin && cp /work/$N/build/bin/test-llama-archs /work/$N/bin/
        chown -R "$HOST_UID:$HOST_GID" /work/$N /work/ccache
        exit $rc'
}
# <dir> <base bin dir> <compared bin dir>: the base writes every model's logits, the other build compares
compare() {
    local k=$1 bb=$2 mb=$3
    mkdir -p "$k"
    # shellcheck disable=SC2086
    docker run --rm $D -v "$WORK:$WORK" -v "$bb:$bb:ro" -v "$mb:$mb:ro" -v "$k:/k" "$IMAGE" bash -c '
        run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        run base '"$bb"'
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run merged '"$mb"'
        rm -f /k/*.f32
        chown -R '"$(id -u):$(id -g)"' /k' > "$k/rc.txt" 2>&1
    grep 'logits against' "$k/merged.log" | grep -v ', 0 of ' > "$k/differ.txt"
    echo "base wrote $(cat "$k/n_base.txt") logits files; $(grep -c 'logits against' "$k/merged.log") compared, $(wc -l < "$k/differ.txt") differ: $(cut -d: -f1 "$k/differ.txt" | tr '\n' ' '); written by the compared build only: $(grep -c 'logits_dump_or_compare: wrote' "$k/merged.log"); base: $(grep -m 1 '^Summary' "$k/base.log"); compared: $(grep -m 1 '^Summary' "$k/merged.log"); $(tr '\n' ' ' < "$k/rc.txt")" | tee "$k/verdict.txt"
}

if [ -z "${REVERT:-}" ]; then
    # stock upstream with the fork's test file
    rm -rf "$WORK/stock/src" && mkdir -p "$WORK/stock/src"
    git -C "$SRC" archive "$UPSTREAM" | tar -x -C "$WORK/stock/src"
    git -C "$SRC" show "$BRANCH_REF:tests/test-llama-archs.cpp" > "$WORK/stock/src/tests/test-llama-archs.cpp"
    build stock || { echo "stock build failed: $WORK/stock/build.log" | tee "$WORK/stock/verdict.txt"; date -Is > "$WORK/done"; exit 1; }
    compare "$WORK/stock/cmp" "$WORK/stock/bin" "$BRANCH_BIN"
else
    # the branch with single upstream commits reverted, against an older build's logits
    : "${BASE_BIN:?BASE_BIN=<older CPU bin/>}"
    T=$WORK/revert
    rm -rf "$T/src" && mkdir -p "$T/src"
    git -C "$SRC" archive "$BRANCH_REF" | tar -x -C "$T/src"
    : > "$T/reverted.txt"
    for c in $REVERT; do
        if git -C "$SRC" diff "$c" "$c^" | (cd "$T/src" && git apply -) >> "$T/reverted.txt" 2>&1; then
            echo "reverted $c" >> "$T/reverted.txt"
        else
            echo "FAILED to revert $c" | tee -a "$T/reverted.txt" "$T/verdict.txt"; date -Is > "$WORK/done"; exit 1
        fi
    done
    build revert || { echo "revert build failed: $T/build.log" | tee "$T/verdict.txt"; date -Is > "$WORK/done"; exit 1; }
    compare "$T/cmp" "$BASE_BIN" "$T/bin"
fi
date -Is > "$WORK/done"
