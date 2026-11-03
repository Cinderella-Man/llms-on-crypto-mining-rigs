#!/usr/bin/env bash
# C18-C20 (Q436-Q438): the rig build of the recurrent-layer fusions branch, test-llama-archs from the same tree, then
# a CPU-only check that the fused graph gives the unfused graph's logits on a random qwen4exp (Flash-Next's
# architecture) model, in a container with NO GPU (the CUDA toolkit's stub libcuda.so as libcuda.so.1).
#
# The CPU check runs test-llama-archs -a qwen4exp -b CPU twice per setting: LLAMA_GDN_FUSE=0 writes the reference
# logits (one batch of 128 tokens, then the same tokens in steps of 1 to 8), LLAMA_GDN_FUSE=1 compares with them.
# Settings: no rollback snapshots, and LLAMA_ARCHS_N_RS_SEQ=3 (four slots, as a 4-token verify keeps).
#
# Usage: WORK=<dir> [BRANCH=pascal-p106-v3-b12076-gdnfuse] [JOBS=6] [LOCK=<file>] [DOCKER_ARGS=...] build_and_cpu_check.sh
#   WORK   the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src needs the fork as
#          remote "fork" with an https fetch URL; the public repo (this script's) in $WORK/repo
#   LOCK   a file to flock around the two builds, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree), $WORK/md5.txt, $WORK/extra/test-llama-archs,
#         $WORK/logs/*.log, $WORK/checks/cpu/*.log; done when $WORK/logs/chain.log ends with "=== chain ended"
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-gdnfuse}
JOBS=${JOBS:-6}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks/cpu"
echo "=== chain started $(date -Is)"

locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$WORK/repo/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
    t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
    (cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
    grep -E "^(commit|avx|highest)" "$WORK/$(basename "$t" .tar.gz)/BUILDINFO.txt"
    # test-llama-archs from the same build tree (the rig build leaves it in $WORK/src/build)
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs > /work/logs/extra.log 2>&1 \
            || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-llama-archs /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
export -f build
export WORK BRANCH JOBS IMAGE D

echo "=== build started $(date -Is)"
locked bash -c build
rc=$?
echo "=== build ended $(date -Is) rc=$rc"
[ $rc = 0 ] || { echo "=== chain ended $(date -Is) (build failed)"; exit 1; }

echo "=== CPU checks started $(date -Is)"
docker run --rm $D -v "$WORK/extra:/x:ro" -v "$WORK/checks/cpu:/k" "$IMAGE" bash -c '
    mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub
    rm -f /k/*.f32
    for rs in 0 3; do
        for fuse in 0 1; do
            echo "--- n_rs_seq=$rs LLAMA_GDN_FUSE=$fuse"
            LLAMA_ARCHS_N_RS_SEQ=$rs LLAMA_GDN_FUSE=$fuse LLAMA_ARCHS_LOGITS=/k/rs$rs- \
                nice -n 19 /x/test-llama-archs -a "^qwen4exp$" -b CPU -s 1 > /k/archs-rs$rs-fuse$fuse.log 2>&1
            echo "rc=$?"
            grep -E "logits against|one batch against|wrote|FAIL|OK" /k/archs-rs$rs-fuse$fuse.log | sed "s/\x1b\[[0-9;]*m//g"
        done
    done
    chown -R '"$(id -u):$(id -g)"' /k'
echo "=== CPU checks ended $(date -Is) rc=$?"
cat "$WORK/md5.txt"
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
