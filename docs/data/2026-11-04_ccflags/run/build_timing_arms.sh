#!/usr/bin/env bash
# Q671 / C131: test-backend-ops once per arm of timing_arms.tsv, from the fork branch pascal-p106-v3-b12076-ccflags, for
# the P106 timing (p106_timing.sh). Each arm is the rig build's cmake configuration (scripts/build_llamacpp_docker.sh) plus
# -DGGML_CUDA_PASCAL_FLAGS=<the arm's flags>, which the branch applies to the Pascal kernel files only, so the arms share
# every other object through ccache. CPU only: nvcc in the CUDA 12.9 container, no GPU.
#
# Usage: SRC=<tree of the branch> WORK=<dir> ARMS='base iqmc2 ...' [JOBS=6] [LOCK=<file>] [CCACHE_SEED=<ccache dir>]
#        [DOCKER_ARGS=...] build_timing_arms.sh
#   SRC          the branch's tree (a checkout or `git archive`; mounted read-only at /work/src)
#   CCACHE_SEED  a ccache folder from an earlier build of the same tree family (copied once to $WORK/ccache)
# Output: $WORK/timing/<arm>/bin/test-backend-ops, $WORK/timing/<arm>/BUILDINFO.txt, $WORK/timing/<arm>.log
set -uo pipefail
SRC=$(realpath "${SRC:?SRC=<tree of the branch>}")
WORK=$(realpath -m "${WORK:?WORK=<output dir>}")
ARMS=${ARMS:?ARMS=<arm names of timing_arms.tsv>}
JOBS=${JOBS:-6}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(dirname "$(realpath "$0")")
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/timing" "$WORK/ccache"
[ -n "${CCACHE_SEED:-}" ] && [ -z "$(ls -A "$WORK/ccache")" ] && cp -a "$CCACHE_SEED"/. "$WORK/ccache/"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
for arm in $ARMS; do
    flags=$(awk -F'\t' -v a="$arm" '$1==a {print $2}' "$HERE/timing_arms.tsv")
    [ -n "$flags" ] || { echo "arm $arm: not in timing_arms.tsv" >&2; exit 1; }
    [ "$flags" = - ] && flags=""
    echo "=== $arm: GGML_CUDA_PASCAL_FLAGS='$flags' $(date -Is)"
    locked docker run --rm $D -e ARM="$arm" -e FLAGS="$flags" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -v "$SRC:/work/src:ro" -v "$WORK:/w" "$IMAGE" bash -c '
        set -eo pipefail
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/w/ccache
        apt-get update -qq && apt-get install -y -qq cmake build-essential ccache > /dev/null
        b=/w/timing/$ARM.build
        cmake -S /work/src -B $b -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
            -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DGGML_CUDA_PASCAL_FLAGS="$FLAGS" \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build $b -j "$JOBS" --target test-backend-ops > /dev/null
        mkdir -p /w/timing/$ARM/bin
        cp $b/bin/test-backend-ops /w/timing/$ARM/bin/
        { echo "arm: $ARM"; echo "GGML_CUDA_PASCAL_FLAGS: $FLAGS"; echo "built: $(date -Is)";
          nvcc --version | tail -n 1; ccache -s | grep -iE "hit|miss" | head -n 4; } > /w/timing/$ARM/BUILDINFO.txt
        rm -rf $b
        chown -R "$HOST_UID:$HOST_GID" /w/timing /w/ccache
    ' > "$WORK/timing/$arm.log" 2>&1 || { echo "arm $arm FAILED, see $WORK/timing/$arm.log"; tail -n 5 "$WORK/timing/$arm.log"; }
    ls -la "$WORK/timing/$arm/bin/test-backend-ops" 2>&1 | tail -n 1
done
echo "=== done $(date -Is)"
