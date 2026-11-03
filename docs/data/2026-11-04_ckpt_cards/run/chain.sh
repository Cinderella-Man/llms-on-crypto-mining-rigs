#!/usr/bin/env bash
# C22 / C28 (Q440, Q448, Q548) on a build machine, CPU only: the rig build of the checkpoint branch (sm_61, no AVX,
# test-backend-ops), test-llama-context and test-llama-archs from the same build tree, then checks.sh in a container
# without a GPU (the CUDA toolkit's stub libcuda.so stands in for the driver, so the CUDA backend finds no device).
# WORK is the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs the fork as remote
# "fork", with an https fetch URL).
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-ckpt] [JOBS=6] [THREADS=8] [DOCKER_ARGS=...] \
#   bash chain.sh
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ md5 in $WORK/md5.txt), $WORK/extra/<branch>/,
#   $WORK/checks/ (checks.sh), $WORK/logs/; done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-ckpt}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
RUN=docs/data/2026-11-04_ckpt_cards/run
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/DONE"
echo "=== chain started $(date -Is)"

TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="--cpu-shares 256 ${DOCKER_ARGS:-}" \
    "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" > "$WORK/logs/build.log" 2>&1
rc=$?
echo "=== build ended $(date -Is) rc=$rc"
if [ $rc != 0 ]; then
    grep -E "error|Error" "$WORK/logs/build.log" | head -30
    date -Is > "$WORK/DONE"; exit 1
fi
t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
(cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
grep -E "^(commit|avx|highest)" "$WORK/$(basename "$t" .tar.gz)/BUILDINFO.txt"

docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$WORK:/work" -e JOBS="$JOBS" -e B="$BRANCH" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
    apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
    git config --global --add safe.directory /work/src
    cd /work/src
    cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
    cmake --build build -j "$JOBS" --target test-llama-context test-llama-archs > /work/logs/extra.log 2>&1 \
        || { tail -40 /work/logs/extra.log; exit 1; }
    mkdir -p /work/extra/$B
    cp build/bin/test-llama-context build/bin/test-llama-archs /work/extra/$B/
    chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs
' > "$WORK/logs/extra-docker.log" 2>&1
rc=$?
echo "=== extra targets ended $(date -Is) rc=$rc"
[ $rc = 0 ] || { tail -40 "$WORK/logs/extra-docker.log"; date -Is > "$WORK/DONE"; exit 1; }

B=$WORK/$(basename "$t" .tar.gz)/bin
cp "$WORK/extra/$BRANCH"/test-* "$B/"
docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$B:/x:ro" -v "$REPO/$RUN:/ckrun:ro" -v "$WORK/checks:/k" \
    -e THREADS="$THREADS" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
    mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
    cd /k && LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub BIN=/x RUN=/ckrun OUT=/k THREADS=$THREADS bash /ckrun/checks.sh
    rc=$?
    chown -R '"$(id -u):$(id -g)"' /k
    exit $rc' > "$WORK/logs/checks.log" 2>&1
rc=$?
echo "=== checks ended $(date -Is) rc=$rc"
tail -30 "$WORK/logs/checks.log"
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
