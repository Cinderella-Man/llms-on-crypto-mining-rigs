#!/usr/bin/env bash
# C267-C269 (Q823-Q825) on a build machine, CPU only: fork branch pascal-p106-v3-b12076-kshape as the rig runs it, and
# the bench with the same launch shapes.
#   rig    the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, no AVX)
#   extra  test-llama-archs, test-llama-context, test-draft-mem and test-hop-cast from the same build tree (a second
#          tarball, as for -ladder3)
#   bench  bench/mmvq_pascal's mmvq_kcol (build_kcol.sh with the grid types from the branch's ggml-common.h) and its
#          header check: src/kcol_mm.cuh must be the branch's mmvq-pascal-kcol-impl.cuh after its 3-line banner
# Usage: WORK=<build folder; its llama.cpp clone in WORK/src has the fork as remote "fork" (https fetch URL)> \
#   REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-kshape] [JOBS=8] [LOCK=<flock file shared by the machine's
#   builds>] [DOCKER_ARGS='--cpu-shares=256 --label ...'] [STEPS='rig extra bench'] bash build.sh
# Output: WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz} (the extras also in its bin/), WORK/extra-<sha>.tar.gz,
#   REPO/bench/mmvq_pascal/build/mmvq_kcol, WORK/md5.txt, WORK/logs/, WORK/DONE (with each step's exit status)
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<build folder>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-kshape}
JOBS=${JOBS:-8}
STEPS=${STEPS:-rig extra bench}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D=${DOCKER_ARGS:-}
mkdir -p "$WORK/logs"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
tree() { ls -td "$WORK"/llama.cpp-*-"$BRANCH" 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; } # the newest
status=""

if step rig; then
    echo "=== rig $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/rig.log" 2>&1
    rc=$?
    status="$status rig=$rc"
    echo "=== rig ended $(date -Is) rc=$rc"
    if [ $rc = 0 ]; then
        T=$(tree)
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        grep -E '^(name|commit|avx|highest)' "$T/BUILDINFO.txt"
    else
        grep -a -E "error|Error" "$WORK/logs/rig.log" | head -40
    fi
fi

if step extra; then
    echo "=== extra $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs test-llama-context test-draft-mem test-hop-cast \
            > /work/logs/extra-build.log 2>&1 || { tail -40 /work/logs/extra-build.log; exit 1; }
        rm -rf /work/extra && mkdir -p /work/extra
        cp build/bin/test-llama-archs build/bin/test-llama-context build/bin/test-draft-mem build/bin/test-hop-cast /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs' > "$WORK/logs/extra.log" 2>&1
    rc=$?
    status="$status extra=$rc"
    if [ $rc = 0 ]; then
        sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
        tar czf "$WORK/extra-$sha.tar.gz" -C "$WORK" extra
        T=$(tree)
        [ -n "$T" ] && cp "$WORK"/extra/* "$T/bin/"
        (cd "$WORK" && md5sum "extra-$sha.tar.gz" && stat -c '%n %s bytes' "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
    fi
    echo "=== extra ended $(date -Is) rc=$rc"
fi

if step bench; then
    echo "=== bench $(date -Is)"
    B=$REPO/bench/mmvq_pascal
    if diff <(tail -n +4 "$B/src/kcol_mm.cuh") "$WORK/src/ggml/src/ggml-cuda/mmvq-pascal-kcol-impl.cuh" > "$WORK/logs/header.diff"; then
        echo "header: kcol_mm.cuh = the branch's mmvq-pascal-kcol-impl.cuh"
        hdr=0
    else
        echo "header: kcol_mm.cuh DIFFERS from the branch ($(grep -c '^[<>]' "$WORK/logs/header.diff") lines)"
        hdr=1
    fi
    IQ_TREE=$WORK/src DOCKER_ARGS="$D" locked nice -n 19 bash "$B/build_kcol.sh" > "$WORK/logs/bench.log" 2>&1
    rc=$?
    status="$status header=$hdr bench=$rc"
    [ $rc = 0 ] && (cd "$B/build" && md5sum mmvq_kcol | tee -a "$WORK/md5.txt")
    grep -a -E "spill|error" "$B/build/ptxas_kcol.log" | sort | uniq -c | head -5
    echo "=== bench ended $(date -Is) rc=$rc"
fi

echo "$(date -Is)$status" > "$WORK/DONE"
cat "$WORK/DONE"
