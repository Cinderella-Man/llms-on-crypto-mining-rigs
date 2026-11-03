#!/usr/bin/env bash
# The rig build of the rebased v3 base (sm_61 code + compute_61 PTX, no AVX, test-backend-ops), then test-llama-archs from
# the same build tree. WORK is the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src
# needs the fork as remote "fork"); BUILD_SCRIPT defaults to that script in this repo.
# Usage: WORK=<dir> [REF=fork/pascal-p106-v3-b12076] [JOBS=8] [BUILD_SCRIPT=...] [DOCKER_ARGS=...] build.sh
# Output: $WORK/llama.cpp-<date>-<sha>-pascal-p106-v3-b12076.tar.gz and $WORK/extra/test-llama-archs
set -euo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REF=${REF:-fork/pascal-p106-v3-b12076}
BUILD_SCRIPT=${BUILD_SCRIPT:-$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/build_llamacpp_docker.sh}
mkdir -p "$WORK/logs"
echo "=== build started $(date -Is) ref $REF"
TESTS=1 JOBS=${JOBS:-8} WORK=$WORK DOCKER_ARGS="--cpu-shares 256 ${DOCKER_ARGS:-}" "$BUILD_SCRIPT" "$REF"
echo "=== build ended $(date -Is)"
docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$WORK:/work" -e JOBS="${JOBS:-8}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    git config --global --add safe.directory /work/src
    cd /work/src
    cmake --build build -j "$JOBS" --target test-llama-archs > /work/logs/build-archs.log 2>&1 || { tail -40 /work/logs/build-archs.log; exit 1; }
    mkdir -p /work/extra && cp build/bin/test-llama-archs /work/extra/
    chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs
'
echo "=== test-llama-archs built $(date -Is): $WORK/extra/test-llama-archs"
