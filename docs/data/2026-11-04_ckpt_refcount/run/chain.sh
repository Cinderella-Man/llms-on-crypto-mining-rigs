#!/usr/bin/env bash
# C154 / C155 (Q704, Q706) on a build machine, CPU only: the rig build of the branch (sm_61, no AVX, test-backend-ops),
# a native CPU build of the same tree (llama-server, test-llama-context, test-llama-archs, llama-fit-params), then
# checks.sh in a container without a GPU (the CUDA toolkit's stub libcuda.so stands in for the driver, so the rig
# build's CUDA backend finds no device).
# WORK is the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs the fork as remote
# "fork", with an https fetch URL).
# Usage: WORK=<dir> REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-ckptrc] [JOBS=6] [THREADS=8]
#   [OLD_BIN=<bin dir of a build without the change>] [BIG_MODEL=<gguf>] [MAIN=<gguf> DRAFT=<gguf> MTP=<gguf>]
#   [MOUNTS='<dirs the models and OLD_BIN live in>'] [DOCKER_ARGS=...] [STEPS='build extra checks'] [PARTS=<checks.sh's>]
#   bash chain.sh
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ md5 in $WORK/md5.txt), $WORK/extra/<branch>/,
#   $WORK/checks/ (checks.sh), $WORK/logs/; done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-ckptrc}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
STEPS=${STEPS:-build extra checks}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
RUN=docs/data/2026-11-04_ckpt_refcount/run
CKRUN=docs/data/2026-11-04_ckpt_cards/run
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/DONE"
echo "=== chain started $(date -Is)"

if [[ " $STEPS " == *" build "* ]]; then
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
fi
t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
B=$WORK/$(basename "$t" .tar.gz)/bin

if [[ " $STEPS " == *" extra "* ]]; then
    docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$WORK:/work" -e JOBS="$JOBS" -e B="$BRANCH" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /dev/null
        cmake --build build-cpu -j "$JOBS" --target llama-server llama-fit-params test-llama-context test-llama-archs \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        mkdir -p /work/extra/$B
        cp build-cpu/bin/llama-server build-cpu/bin/llama-fit-params build-cpu/bin/test-llama-context \
            build-cpu/bin/test-llama-archs /work/extra/$B/
        git log -1 --format="native CPU build of %H" > /work/extra/$B/COMMIT
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs /work/ccache
    ' > "$WORK/logs/extra-docker.log" 2>&1
    rc=$?
    echo "=== native CPU build ended $(date -Is) rc=$rc"
    [ $rc = 0 ] || { tail -40 "$WORK/logs/extra-docker.log"; date -Is > "$WORK/DONE"; exit 1; }
    cp "$WORK/extra/$BRANCH"/test-llama-context "$WORK/extra/$BRANCH"/test-llama-archs "$B/"
fi

if [[ " $STEPS " == *" checks "* ]]; then
    mounts=""
    for m in ${MOUNTS:-}; do mounts+=" -v $m:$m:ro"; done
    docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} $mounts -v "$B:/x:ro" -v "$WORK/extra/$BRANCH:/n:ro" \
        -v "$REPO/$RUN:/run_:ro" -v "$REPO/$CKRUN:/ckrun:ro" -v "$WORK/checks:/k" \
        -e THREADS="$THREADS" -e OLD_BIN="${OLD_BIN:-}" -e BIG_MODEL="${BIG_MODEL:-}" -e MAIN="${MAIN:-}" \
        -e DRAFT="${DRAFT:-}" -e MTP="${MTP:-}" -e PARTS="${PARTS:-}" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        cd /k && LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub BIN=/x NBIN=/n RUN=/run_ CKRUN=/ckrun OUT=/k \
            bash /run_/checks.sh
        rc=$?
        chown -R '"$(id -u):$(id -g)"' /k
        exit $rc' > "$WORK/logs/checks.log" 2>&1
    rc=$?
    echo "=== checks ended $(date -Is) rc=$rc"
    tail -40 "$WORK/logs/checks.log"
fi
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
