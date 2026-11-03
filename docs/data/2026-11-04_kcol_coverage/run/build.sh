#!/usr/bin/env bash
# The builds and CPU checks for Q630, Q611 and Q612 on a build machine (CPU only, in Docker): the rig tarball of fork
# branch pascal-p106-v3-b12076-kcol with test-backend-ops (sm_61 + compute_61 PTX, no AVX), the bench binaries
# bench/mmvq_pascal/build/mmvq_kcol and build/mmvq_pascal, the CPU check of the k-column kernels' decoding
# (verify/kcol_unit_check.sh) and the same check with three planted errors (run/mutation_check.sh). WORK: the work
# folder of scripts/build_llamacpp_docker.sh, whose llama.cpp clone ($WORK/src) has the fork as remote "fork" with an
# https fetch URL; REPO: a checkout of this repository; LOCK: a lock file other builds on the machine share (optional).
# Usage: WORK=<dir> REPO=<checkout> [LOCK=<file>] [JOBS=6] [DOCKER_ARGS=...] docs/data/2026-11-04_kcol_coverage/run/build.sh
set -euo pipefail
WORK=${WORK:?WORK=<work folder>}
REPO=${REPO:?REPO=<this repository>}
JOBS=${JOBS:-6}
DA="--cpu-shares 256 ${DOCKER_ARGS:-}"
run() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
run nice -n 19 env TESTS=1 JOBS="$JOBS" WORK="$WORK" DOCKER_ARGS="$DA" \
    "$REPO/scripts/build_llamacpp_docker.sh" fork/pascal-p106-v3-b12076-kcol
run nice -n 19 env DOCKER_ARGS="$DA" "$REPO/bench/mmvq_pascal/build_kcol.sh"
run nice -n 19 env DOCKER_ARGS="$DA" "$REPO/bench/mmvq_pascal/build.sh"
nice -n 19 env DOCKER_ARGS="$DA" "$REPO/bench/mmvq_pascal/verify/kcol_unit_check.sh" 40
nice -n 19 env DOCKER_ARGS="$DA" REPO="$REPO" WORK="$WORK" "$REPO/docs/data/2026-11-04_kcol_coverage/run/mutation_check.sh"
t=$(ls -t "$WORK"/llama.cpp-*-pascal-p106-v3-b12076-kcol.tar.gz | head -1)
md5sum "$t" "$REPO/bench/mmvq_pascal/build/mmvq_kcol" "$REPO/bench/mmvq_pascal/build/mmvq_pascal"
