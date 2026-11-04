#!/usr/bin/env bash
# Q897's builds of the -spancount branch (llama-imatrix --span-mask / --span-dump), on any x86-64 Linux PC with Docker
# (no GPU needed to compile):
#   cpu     llama-imatrix and test-llama-archs for the CPU (native code) in $WORK/cpu/bin: the checks (cpu_check.sh),
#           the token dumps
#   cuda86  llama-imatrix for Ampere cards (sm_86, native CPU code), packed as
#           $WORK/llama.cpp-<date>-<commit>-pascal-p106-v3-b12076-spancount-sm86.tar.gz (bin/, BUILDINFO.txt) for the
#           GPU pass (gpu_pass.sh); its md5 is printed
# Env: WORK, SRC (a checkout of the branch), JOBS (8), STEPS (default "cpu cuda86"), IMAGE, DOCKER_ARGS, LOCK (a file
#      to flock around each build)
set -uo pipefail
: "${WORK:?}" "${SRC:?}"
JOBS=${JOBS:-8}
STEPS=${STEPS:-cpu cuda86}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--rm --cpu-shares 256 --label pascal-quants=1 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs"

log()    { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
failed=0

# build <name> <cmake args> <targets>: a Release build of SRC in $WORK/<name>/build, the targets copied to <name>/bin
build() {
    local name=$1 args=$2 targets=$3
    log "$name: $(git -C "$SRC" log -1 --format='%h %s')"
    locked docker run $D -v "$WORK:$WORK" -v "$SRC:$SRC" -e WORK="$WORK" -e SRC="$SRC" -e JOBS="$JOBS" -e NAME="$name" \
        -e ARGS="$args" -e TARGETS="$targets" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=$WORK/ccache-$NAME
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory "$SRC"
        cmake -S "$SRC" -B "$WORK/$NAME/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            $ARGS > "$WORK/logs/cmake-$NAME.log" 2>&1
        rc=0
        nice -n 19 cmake --build "$WORK/$NAME/build" -j "$JOBS" --target $TARGETS > "$WORK/logs/build-$NAME.log" 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            mkdir -p "$WORK/$NAME/bin"
            for t in $TARGETS; do cp "$WORK/$NAME/build/bin/$t" "$WORK/$NAME/bin/"; done
            git -C "$SRC" log -1 --format="%H %s" > "$WORK/$NAME/bin/COMMIT"
            git -C "$SRC" status --short >> "$WORK/$NAME/bin/COMMIT"
        fi
        chown -R "$HOST_UID:$HOST_GID" "$WORK/$NAME" "$WORK/logs" "$WORK/ccache-$NAME"
        exit $rc' > "$WORK/logs/$name.out" 2>&1
}

if step cpu; then
    if build cpu "-DGGML_CUDA=OFF" "llama-imatrix test-llama-archs"; then
        log "cpu: $WORK/cpu/bin ($(head -c 9 "$WORK/cpu/bin/COMMIT"))"
    else
        grep -E "error|Error" "$WORK/logs/build-cpu.log" | head -40
        log "FAILED cpu: $WORK/logs/build-cpu.log"; failed=1
    fi
fi

if step cuda86; then
    if build cuda86 "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" \
            "llama-imatrix"; then
        sha=$(git -C "$SRC" rev-parse --short=9 HEAD)
        name=llama.cpp-$(date +%Y%m%d)-$sha-pascal-p106-v3-b12076-spancount-sm86
        rm -rf "${WORK:?}/$name" && mkdir -p "$WORK/$name/bin"
        cp "$WORK/cuda86/bin/llama-imatrix" "$WORK/$name/bin/"
        { echo "branch pascal-p106-v3-b12076-spancount"; cat "$WORK/cuda86/bin/COMMIT"
          echo "image $IMAGE; CUDA sm_86, GGML_NATIVE=ON, static; run it in that image with the GPUs"; } > "$WORK/$name/BUILDINFO.txt"
        tar czf "$WORK/$name.tar.gz" -C "$WORK" "$name"
        log "cuda86: $name.tar.gz, $(stat -c %s "$WORK/$name.tar.gz") B, md5 $(md5sum < "$WORK/$name.tar.gz" | cut -d' ' -f1)"
    else
        grep -E "error|Error" "$WORK/logs/build-cuda86.log" | head -40
        log "FAILED cuda86: $WORK/logs/build-cuda86.log"; failed=1
    fi
fi
exit $failed
