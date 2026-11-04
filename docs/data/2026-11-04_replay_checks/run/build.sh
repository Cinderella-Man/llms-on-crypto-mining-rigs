#!/usr/bin/env bash
# C372 (Q931): the replay branch's builds, on any x86-64 Linux PC with Docker (no GPU needed to compile):
#   rig     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, CPU code without
#           AVX): it records on the build machine's CPU (scripts/record_checks.sh) and replays on the rig
#           (scripts/replay_checks.sh); on a newer card it replays through the compute_61 PTX
#   cuda86  test-backend-ops for Ampere cards (sm_86 code) with the rig build's CPU flags, so that its CPU backend runs
#           the same code as the rig's: replays and live checks on the build machine's own cards
# Env: WORK (rig: a folder whose src/ is a llama.cpp clone with the fork as remote "fork", https fetch URL), REPO (this
#   repository), BRANCH (default pascal-p106-v3-b12076-replay), SRC (cuda86: a checkout of the same commit), JOBS (8),
#   STEPS (default "rig cuda86"), LOCK (a file to flock around each build), IMAGE (a CUDA 12.9 devel image with
#   cmake, gcc and ccache), DOCKER_ARGS
# Output: WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz}, WORK/cuda86/bin/test-backend-ops (+ COMMIT), WORK/md5.txt,
#   WORK/logs/, WORK/DONE (each step's exit status)
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<build folder>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-replay}
JOBS=${JOBS:-8}
STEPS=${STEPS:-rig cuda86}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; } # one build at a time on a shared machine
status=""

if step rig; then
    REPO=$(realpath "${REPO:?the rig step needs REPO}")
    echo "=== rig $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK IMAGE=$IMAGE DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$WORK/logs/rig.log" 2>&1
    rc=$?
    status="$status rig=$rc"
    if [ $rc = 0 ]; then
        T=$(ls -td "$WORK"/llama.cpp-*-"$BRANCH" | grep -v '\.tar\.gz$' | head -n 1)
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        grep -E '^(name|commit|avx|highest)' "$T/BUILDINFO.txt"
    else
        grep -a -E "error|Error" "$WORK/logs/rig.log" | head -40
    fi
fi

if step cuda86; then
    SRC=$(realpath "${SRC:?the cuda86 step needs SRC}")
    echo "=== cuda86 $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:$WORK" -v "$SRC:$SRC" -e WORK="$WORK" -e SRC="$SRC" -e JOBS="$JOBS" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=$WORK/ccache-cuda86
        command -v cmake > /dev/null || { apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null; }
        git config --global --add safe.directory "$SRC"
        cmake -S "$SRC" -B "$WORK/cuda86/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF \
            -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > "$WORK/logs/cmake-cuda86.log" 2>&1
        nice -n 19 cmake --build "$WORK/cuda86/build" -j "$JOBS" --target test-backend-ops > "$WORK/logs/cuda86.log" 2>&1 \
            || { tail -40 "$WORK/logs/cuda86.log"; exit 1; }
        mkdir -p "$WORK/cuda86/bin"
        cp "$WORK/cuda86/build/bin/test-backend-ops" "$WORK/cuda86/bin/"
        git -C "$SRC" log -1 --format="%H %s" > "$WORK/cuda86/bin/COMMIT"
        chown -R "$HOST_UID:$HOST_GID" "$WORK/cuda86" "$WORK/logs" "$WORK/ccache-cuda86"' > "$WORK/logs/cuda86.out" 2>&1
    rc=$?
    status="$status cuda86=$rc"
    if [ $rc = 0 ]; then
        (cd "$WORK/cuda86/bin" && md5sum test-backend-ops | sed 's|test-backend-ops|cuda86/bin/test-backend-ops|') | tee -a "$WORK/md5.txt"
    else
        tail -n 20 "$WORK/logs/cuda86.out"
    fi
fi

echo "$(date -Is)$status" > "$WORK/DONE"
