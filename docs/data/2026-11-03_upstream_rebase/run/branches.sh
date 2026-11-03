#!/usr/bin/env bash
# C66b (Q573 part b): the rig builds of the feature branches on the rebased base, one at a time (sm_61 code + compute_61
# PTX, no AVX, test-backend-ops), each packed as a rig tarball with its md5. For the branches in EXTRA also
# test-llama-archs, llama-pipespec and llama-pipe-recurrent-check from the same build tree, into $WORK/extra/<branch>/.
# CPU only: no container gets a GPU. WORK is the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in
# $WORK/src needs the fork as remote "fork", with an https fetch URL).
# Usage: WORK=<dir> BRANCHES='<fork branch> ...' [EXTRA='<fork branch> ...'] [JOBS=4] [BUILD_SCRIPT=...] [DOCKER_ARGS=...] \
#   branches.sh
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ its unpacked tree), $WORK/md5.txt, $WORK/logs/build-<branch>.log
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BRANCHES=${BRANCHES:?BRANCHES=<fork branches>}
JOBS=${JOBS:-4}
BUILD_SCRIPT=${BUILD_SCRIPT:-$(cd "$(dirname "$0")/../../../.." && pwd)/scripts/build_llamacpp_docker.sh}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
mkdir -p "$WORK/logs"
fail=0
for b in $BRANCHES; do
    echo "=== $b: build started $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="--cpu-shares 256 ${DOCKER_ARGS:-}" "$BUILD_SCRIPT" "fork/$b" \
        > "$WORK/logs/build-$b.log" 2>&1
    rc=$?
    echo "=== $b: build ended $(date -Is) rc=$rc"
    if [ $rc != 0 ]; then
        grep -E "error|Error" "$WORK/logs/build-$b.log" | head -20
        fail=1
        continue
    fi
    t=$(ls -t "$WORK"/llama.cpp-*-"$b".tar.gz | head -1)
    (cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
    grep -E "^(commit|avx|highest)" "$WORK/$(basename "$t" .tar.gz)/BUILDINFO.txt"
    if [[ " ${EXTRA:-} " == *" $b "* ]]; then
        docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$WORK:/work" -e JOBS="$JOBS" -e B="$b" \
            -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
            apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
            git config --global --add safe.directory /work/src
            cd /work/src
            cmake --build build -j "$JOBS" --target test-llama-archs llama-pipespec llama-pipe-recurrent-check \
                > /work/logs/extra-$B.log 2>&1 || { tail -40 /work/logs/extra-$B.log; exit 1; }
            mkdir -p /work/extra/$B
            cp build/bin/test-llama-archs build/bin/llama-pipespec build/bin/llama-pipe-recurrent-check /work/extra/$B/
            chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs
        '
        rc=$?
        echo "=== $b: extra targets ended $(date -Is) rc=$rc"
        [ $rc = 0 ] || fail=1
    fi
done
exit $fail
