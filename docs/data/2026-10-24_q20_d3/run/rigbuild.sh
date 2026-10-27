#!/usr/bin/env bash
# D3 exactness set (2026-10-24/25). Run from a copy in the check dir D (README.md), next to a copy of
# scripts/build_llamacpp_docker.sh of this repository.
# Q20/D3: the rig tarball of pascal-p106-v2-drafter (compute 6.1 only, no AVX, TESTS=1) from the bundle remote q20
# (rigwork/q20.bundle, made with `git bundle create - f1b67d8b8..pascal-p106-v2-drafter`), -j 8.
cd "$(dirname "$0")"
WORK=$PWD/rigwork TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares=256 --label pascal-quants=1 --name pq-q20-d3-rigbuild" \
    ./build_llamacpp_docker.sh q20/pascal-p106-v2-drafter > rigbuild.log 2>&1
echo "exit $?" >> rigbuild.log
