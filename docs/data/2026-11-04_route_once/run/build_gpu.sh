#!/usr/bin/env bash
# Q860's CUDA builds, on any x86-64 Linux PC with Docker (no GPU needed to compile):
#   rig     the rig tarball of the fork branch (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 + compute_61 PTX, no
#           AVX) in $WORK/rig, then test-route-once from the same build tree, packed beside it as
#           extra-<commit>.tar.gz (the CUDA check on a P106: run/rig_timing.sh's unit step)
#   cuda86  llama-perplexity and test-route-once of SRC for Ampere cards (sm_86, native CPU code) in $WORK/cuda86/bin,
#           for the KLD by k (run/kld_route_once.sh) and the CUDA check on such a card
# Env: WORK, REPO (a checkout of this repository, for scripts/build_llamacpp_docker.sh), BRANCH (the fork branch,
#      default pascal-p106-v3-b12076-routeonce; the rig step fetches fork/$BRANCH into $WORK/rig/src, a clone with a
#      remote "fork"), SRC (a checkout of the same commit, for cuda86), JOBS (6), LOCK (a file to flock around each
#      build), STEPS (default "rig cuda86"), IMAGE, DOCKER_ARGS
set -uo pipefail
: "${WORK:?}"
BRANCH=${BRANCH:-pascal-p106-v3-b12076-routeonce}
JOBS=${JOBS:-6}
STEPS=${STEPS:-rig cuda86}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares 256 --label pascal-quants=1 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs"

log()    { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
failed=0

if step rig; then
    : "${REPO:?the rig step needs REPO}"
    R=$WORK/rig
    mkdir -p "$R"
    log "rig: build_llamacpp_docker.sh fork/$BRANCH"
    if TESTS=1 JOBS=$JOBS WORK=$R DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
            "fork/$BRANCH" > "$WORK/logs/rig-build.log" 2>&1; then
        tb=$(ls -t "$R"/llama.cpp-*-"$BRANCH".tar.gz | head -n 1)
        sha=$(git -C "$R/src" rev-parse --short=9 HEAD)
        log "rig: $(basename "$tb"), $(stat -c %s "$tb") B, md5 $(md5sum < "$tb" | cut -d' ' -f1)"
        # test-route-once from the rig build's tree (its cmake cache: sm_61, no AVX), with the tests on
        locked docker run --rm $D -v "$R:/work" -e JOBS="$JOBS" -e SHA="$sha" -e HOST_UID="$(id -u)" \
            -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
            apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
            git config --global --add safe.directory /work/src
            cd /work/src
            cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
            nice -n 19 cmake --build build -j "$JOBS" --target test-route-once > /work/extra.log 2>&1 \
                || { tail -40 /work/extra.log; exit 1; }
            rm -rf "/work/extra-$SHA" && mkdir -p "/work/extra-$SHA"
            cp build/bin/test-route-once "/work/extra-$SHA/"
            tar czf "/work/extra-$SHA.tar.gz" -C /work "extra-$SHA"
            chown -R "$HOST_UID:$HOST_GID" /work' > "$WORK/logs/rig-extra.log" 2>&1 \
            && log "rig extra: extra-$sha.tar.gz, $(stat -c %s "$R/extra-$sha.tar.gz") B, md5 $(md5sum < "$R/extra-$sha.tar.gz" | cut -d' ' -f1)" \
            || { log "FAILED rig extra: $WORK/logs/rig-extra.log"; failed=1; }
    else
        grep -E "error|Error" "$WORK/logs/rig-build.log" | head -40
        log "FAILED rig: $WORK/logs/rig-build.log"; failed=1
    fi
fi

if step cuda86; then
    : "${SRC:?the cuda86 step needs SRC}"
    log "cuda86: $(git -C "$SRC" log -1 --format='%h %s')"
    locked docker run --rm $D -v "$WORK:$WORK" -v "$SRC:$SRC" -e WORK="$WORK" -e SRC="$SRC" -e JOBS="$JOBS" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=$WORK/ccache-cuda86
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory "$SRC"
        cmake -S "$SRC" -B "$WORK/cuda86/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > "$WORK/logs/cmake-cuda86.log" 2>&1
        nice -n 19 cmake --build "$WORK/cuda86/build" -j "$JOBS" --target llama-perplexity test-route-once \
            > "$WORK/logs/cuda86.log" 2>&1 || { tail -40 "$WORK/logs/cuda86.log"; exit 1; }
        mkdir -p "$WORK/cuda86/bin"
        cp "$WORK/cuda86/build/bin/llama-perplexity" "$WORK/cuda86/build/bin/test-route-once" "$WORK/cuda86/bin/"
        git -C "$SRC" log -1 --format="%H %s" > "$WORK/cuda86/bin/COMMIT"
        chown -R "$HOST_UID:$HOST_GID" "$WORK/cuda86" "$WORK/logs" "$WORK/ccache-cuda86"' \
        > "$WORK/logs/cuda86.out" 2>&1 \
        && log "cuda86: $WORK/cuda86/bin ($(head -c 9 "$WORK/cuda86/bin/COMMIT"))" \
        || { tail -20 "$WORK/logs/cuda86.out"; log "FAILED cuda86"; failed=1; }
fi
exit $failed
