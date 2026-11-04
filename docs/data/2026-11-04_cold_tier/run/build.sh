#!/usr/bin/env bash
# Q238 (C368's sibling, the cold tier) on a build machine, CPU only: three builds of the fork branch
# pascal-p106-v3-b12076-coldtier, one after the other.
#   cpu    a CPU build (test-llama-archs, test-moe-ids, llama-perplexity, llama-quantize, llama-gguf) for run/cpu_checks.sh and the
#          tool's llama-quantize step
#   sm86   a CUDA build for compute 8.6 (RTX 30x0) with llama-perplexity, llama-server, llama-quantize, llama-gguf,
#          test-backend-ops, test-llama-archs and test-moe-ids, packed as <name>-sm86.tar.gz: the KLD runs of the tiered files
#   rig    the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, no AVX)
# Usage: WORK=<build folder; WORK/src is a llama.cpp clone with the fork as remote "fork" (https fetch URL)> \
#   REPO=<this repository> [BRANCH=pascal-p106-v3-b12076-coldtier] [JOBS=8] [LOCK=<flock file shared by the machine's
#   builds>] [DOCKER_ARGS='--cpu-shares=256 --label ...'] [STEPS='cpu sm86 rig'] bash build.sh
# Output: WORK/cpu/bin, WORK/sm86/bin and WORK/llama.cpp-<date>-<sha>-<branch>-sm86{,.tar.gz},
#   WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz} (the rig), WORK/md5.txt, WORK/logs/, WORK/DONE (each step's status)
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<build folder>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-coldtier}
JOBS=${JOBS:-8}
STEPS=${STEPS:-cpu sm86 rig}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D=${DOCKER_ARGS:-}
mkdir -p "$WORK/logs"
rm -f "$WORK/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
status=""

git -C "$WORK/src" fetch -q fork "$BRANCH" && git -C "$WORK/src" checkout -q --force FETCH_HEAD || exit 1
sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
echo "=== $BRANCH at $sha $(date -Is)"

# cmake_build <dir> <targets> <cmake flags...>: a Release build of WORK/src in WORK/<dir>/build, binaries to WORK/<dir>/bin
cmake_build() {
    local dir=$1 targets=$2
    shift 2
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e DIR="$dir" -e TARGETS="$targets" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cmake -S /work/src -B "/work/$DIR/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache "$@" > "/work/logs/cmake-$DIR.log" 2>&1
        rc=0
        nice -n 19 cmake --build "/work/$DIR/build" -j "$JOBS" --target $TARGETS > "/work/logs/build-$DIR.log" 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf "/work/$DIR/bin" && mkdir -p "/work/$DIR/bin"
            for t in $TARGETS; do cp "/work/$DIR/build/bin/$t" "/work/$DIR/bin/"; done
            git -C /work/src log -1 --format="%H %s" > "/work/$DIR/bin/COMMIT"
        fi
        chown -R "$HOST_UID:$HOST_GID" "/work/$DIR" /work/logs /work/ccache
        exit $rc' bash "$@" > "$WORK/logs/$dir.out" 2>&1
}

if step cpu; then
    echo "=== cpu $(date -Is)"
    cmake_build cpu "test-llama-archs test-moe-ids llama-perplexity llama-quantize llama-gguf" -DGGML_CUDA=OFF -DGGML_NATIVE=ON
    rc=$?
    status="$status cpu=$rc"
    [ $rc = 0 ] || grep -a -E "error|Error" "$WORK/logs/build-cpu.log" | head -40
    echo "=== cpu ended $(date -Is) rc=$rc"
fi

if step sm86; then
    echo "=== sm86 $(date -Is)"
    cmake_build sm86 "llama-perplexity llama-server llama-quantize llama-gguf test-backend-ops test-llama-archs test-moe-ids" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF
    rc=$?
    status="$status sm86=$rc"
    if [ $rc = 0 ]; then
        name=llama.cpp-$(date +%Y%m%d)-$sha-$BRANCH-sm86
        rm -rf "${WORK:?}/$name" && mkdir -p "$WORK/$name" && cp -a "$WORK/sm86/bin" "$WORK/$name/"
        { echo "name: $name"; echo "commit: $(git -C "$WORK/src" rev-parse HEAD)"; echo "cuda architectures: 86";
          echo "image: $IMAGE"; echo "built: $(date -Is)"; } > "$WORK/$name/BUILDINFO.txt"
        tar czf "$WORK/$name.tar.gz" -C "$WORK" "$name"
        (cd "$WORK" && md5sum "$name.tar.gz" && stat -c '%n %s bytes' "$name.tar.gz") | tee -a "$WORK/md5.txt"
    else
        grep -a -E "error|Error" "$WORK/logs/build-sm86.log" | head -40
    fi
    echo "=== sm86 ended $(date -Is) rc=$rc"
fi

if step rig; then
    echo "=== rig $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/rig.log" 2>&1
    rc=$?
    status="$status rig=$rc"
    if [ $rc = 0 ]; then
        T=$(ls -td "$WORK"/llama.cpp-*-"$BRANCH" 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1)
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        grep -E '^(name|commit|avx|highest)' "$T/BUILDINFO.txt"
    else
        grep -a -E "error|Error" "$WORK/logs/rig.log" | head -40
    fi
    echo "=== rig ended $(date -Is) rc=$rc"
fi

echo "$(date -Is)$status" > "$WORK/DONE"
echo "=== done:$status"
