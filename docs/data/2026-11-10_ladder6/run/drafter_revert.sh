#!/usr/bin/env bash
# C662 (Q1150): which upstream change moved the staircase drafter's numbers. llama-pipe-drafter-check runs the drafter
# on a CPU against the reference logits of a case; its step lines differ between the release build and the merged
# branch (both pass). Three native CPU builds of the tool: the release (BASE), the branch (BRANCH_REF), and the branch
# with the upstream commits of REVERT reverted (a test tree). If the last one gives the release's lines, those commits
# are the whole difference.
# Usage: WORK=<dir> SRC=<llama.cpp clone holding the commits> BASE=<release commit> BRANCH_REF=<branch commit>
#   REVERT='<upstream commit> ...' CASE=<folder with drafter.gguf and case.bin> [SPLIT=6,6,5,5,6,5,5,6,5] [JOBS=8]
#   [LOCK=<flock file>] [DOCKER_ARGS=...] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] bash drafter_revert.sh
# Output: $WORK/{base,branch,revert}.steps (the tool's step lines), $WORK/verdict.txt; done when $WORK/done exists
set -uo pipefail
WORK=$(realpath -m "${WORK:?WORK=<dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${BASE:?BASE=<release commit>}" "${BRANCH_REF:?BRANCH_REF=<branch commit>}" "${REVERT:?REVERT=<upstream commits>}"
CASE=$(realpath "${CASE:?CASE=<drafter case folder>}")
SPLIT=${SPLIT:-6,6,5,5,6,5,5,6,5}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/ccache"
rm -f "$WORK/done"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
tree() { # <name> <commit>
    rm -rf "$WORK/$1/src" && mkdir -p "$WORK/$1/src"
    git -C "$SRC" archive "$2" | tar -x -C "$WORK/$1/src"
}
tree base "$BASE"
tree branch "$BRANCH_REF"
tree revert "$BRANCH_REF"
for c in $REVERT; do
    git -C "$SRC" diff "$c" "$c^" | (cd "$WORK/revert/src" && git apply -) || { echo "FAILED to revert $c" | tee "$WORK/verdict.txt"; date -Is > "$WORK/done"; exit 1; }
done
for n in base branch revert; do
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e N="$n" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S /work/$N/src -B /work/$N/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/$N/cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/$N/build -j "$JOBS" --target llama-pipe-drafter-check > /work/$N/build.log 2>&1 || rc=$?
        chown -R "$HOST_UID:$HOST_GID" /work/$N /work/ccache
        exit $rc' || { echo "build of $n failed: $WORK/$n/build.log" | tee "$WORK/verdict.txt"; date -Is > "$WORK/done"; exit 1; }
    # shellcheck disable=SC2086
    docker run --rm $D -v "$WORK:$WORK" -v "$CASE:$CASE:ro" "$IMAGE" nice -n 19 "$WORK/$n/build/bin/llama-pipe-drafter-check" \
        "$CASE/drafter.gguf" "$CASE/case.bin" --split "$SPLIT" --cpu --host-table > "$WORK/$n.log" 2>&1
    grep -a '^step \|^DRAFTER_CHECK' "$WORK/$n.log" > "$WORK/$n.steps"
done
{
    echo "base:   $(grep -a '^DRAFTER_CHECK' "$WORK/base.steps")"
    echo "branch: $(grep -a '^DRAFTER_CHECK' "$WORK/branch.steps")"
    echo "revert: $(grep -a '^DRAFTER_CHECK' "$WORK/revert.steps")"
    echo "branch = base: $(cmp -s "$WORK/base.steps" "$WORK/branch.steps" && echo yes || echo no); branch with $REVERT reverted = base: $([ -s "$WORK/revert.steps" ] && cmp -s "$WORK/base.steps" "$WORK/revert.steps" && echo yes || echo no)"
} | tee "$WORK/verdict.txt"
date -Is > "$WORK/done"
