#!/usr/bin/env bash
# The rig tarball of fork branch pascal-p106-v3-b12076-fap with test-backend-ops (sm_61 + compute_61 PTX, no AVX), on
# a build machine with Docker, CPU only; then the register census of the decode attention kernel in three source trees,
# and the host check of its transpose-reduce:
#   before:  the -ladder commit 98bbd6a2f (v2.0's kernel)
#   selects: before + run/selects_only.diff (both halves read before the selects, the loops as they were)
#   after:   the branch's tip
# WORK: the work folder of scripts/build_llamacpp_docker.sh, whose llama.cpp clone ($WORK/src) has the fork as remote
# "fork" with an https fetch URL; REPO: a checkout of this repository; LOCK: a lock file other builds on the machine
# share (optional); STEPS: any of "tarball census check" (default all three).
# Usage: WORK=<dir> REPO=<checkout> [LOCK=<file>] [JOBS=6] [STEPS=...] [DOCKER_ARGS=...] \
#        docs/data/2026-11-04_fap_spills/run/build.sh
set -euo pipefail
WORK=${WORK:?WORK=<work folder>}
REPO=${REPO:?REPO=<this repository>}
JOBS=${JOBS:-6}
STEPS=${STEPS:-tarball census check}
DA="--cpu-shares 256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data/2026-11-04_fap_spills/run
run() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }

if [[ " $STEPS " == *" tarball "* ]]; then
    run nice -n 19 env TESTS=1 JOBS="$JOBS" WORK="$WORK" DOCKER_ARGS="$DA" \
        "$REPO/scripts/build_llamacpp_docker.sh" fork/pascal-p106-v3-b12076-fap
    t=$(ls -t "$WORK"/llama.cpp-*-pascal-p106-v3-b12076-fap.tar.gz | head -1)
    md5sum "$t"
fi

if [[ " $STEPS " == *" census "* ]]; then
    # one source tree per arm, from the clone
    git -C "$WORK/src" fetch -q fork
    for arm in before:98bbd6a2f selects:98bbd6a2f after:fork/pascal-p106-v3-b12076-fap; do
        rm -rf "$WORK/${arm%%:*}" && mkdir -p "$WORK/${arm%%:*}/src"
        git -C "$WORK/src" archive "${arm#*:}" | tar -x -C "$WORK/${arm%%:*}/src"
    done
    patch -s -d "$WORK/selects/src" -p1 < "$D/selects_only.diff"
    # the prompt kernel's file (untouched) as a control, and for its own 40-byte spill
    run nice -n 19 env WORK="$WORK" ARMS="before selects after" FILES="fattn-pascal.cu fattn-tile-pascal.cu" \
        DOCKER_ARGS="$DA" "$D/census.sh"
fi

if [[ " $STEPS " == *" check "* ]]; then
    # the transpose-reduce on an emulated warp, v2.0's form against the new one
    mkdir -p "$WORK/check" && cp "$D/reduce_check.cpp" "$WORK/check/"
    docker run --rm $DA -v "$WORK/check:/w" -w /w nvidia/cuda:12.9.2-devel-ubuntu24.04 \
        bash -c "g++ -O2 -std=c++17 -o reduce_check reduce_check.cpp && ./reduce_check > reduce_check.txt" > /dev/null 2>&1 || true
    cat "$WORK/check/reduce_check.txt"
fi
