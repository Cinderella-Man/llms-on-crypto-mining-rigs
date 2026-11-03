#!/usr/bin/env bash
# Q230/Q441's builds and CPU checks on a build machine (CPU only, in Docker): the rig tarball of Miner Llama branch
# pascal-p106-v3-b12076-qsa with test-backend-ops (sm_61 + compute_61 PTX, no AVX), bench/fattn_sparse's CPU check
# (check.sh: the kernel's host model against FP64 and ggml's CPU kernel, then the fault codes and source mutations) on
# the same tree, and the CPU end-to-end check (cpu_e2e.sh) on the branch and on its base -ladder.
# WORK: the work folder of scripts/build_llamacpp_docker.sh, whose llama.cpp clone ($WORK/src) has the fork as remote
# "fork" with an https fetch URL; REPO: a checkout of this repository; LOCK: a lock file other builds on the machine
# share (optional).
# Usage: WORK=<dir> REPO=<checkout> [LOCK=<file>] [JOBS=6] [DOCKER_ARGS=...] docs/data/2026-11-04_qsa_sparse_attn/run/build.sh
set -euo pipefail
WORK=$(realpath "${WORK:?WORK=<work folder>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
JOBS=${JOBS:-6}
DA="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
run() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }

run nice -n 19 env TESTS=1 JOBS="$JOBS" WORK="$WORK" DOCKER_ARGS="$DA" \
    "$REPO/scripts/build_llamacpp_docker.sh" fork/pascal-p106-v3-b12076-qsa
t=$(ls -t "$WORK"/llama.cpp-*-pascal-p106-v3-b12076-qsa.tar.gz | head -1)
md5sum "$t"

# the branch and its base as separate trees of the same clone
for x in qsa:fork/pascal-p106-v3-b12076-qsa ladder:fork/pascal-p106-v3-b12076-ladder; do
    d=$WORK/tree-${x%%:*}
    [ -d "$d" ] || git -C "$WORK/src" worktree add -q --detach "$d" "${x#*:}"
    git -C "$d" checkout -q --detach "${x#*:}"
    git -C "$d" log --oneline -1
done

rm -rf "$REPO/bench/fattn_sparse/build"
run nice -n 19 env SRC="$WORK/tree-qsa" JOBS="$JOBS" DOCKER_ARGS="$DA" "$REPO/bench/fattn_sparse/check.sh"
md5sum "$REPO/bench/fattn_sparse/build/fasp_check"

for x in qsa ladder; do
    run nice -n 19 env SRC="$WORK/tree-$x" OUT="$WORK/e2e/$x" JOBS="$JOBS" DOCKER_ARGS="$DA" "$HERE/cpu_e2e.sh"
done
