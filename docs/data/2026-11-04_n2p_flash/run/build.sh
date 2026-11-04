#!/usr/bin/env bash
# N2P/N3P for Qwen3.8-Flash-Next (Q928/Q220) on a build machine, CPU only: fork branch pascal-p106-v3-b12076-n2p as
# the rig runs it, plus the tools the file and the CPU checks need.
#   rig    the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, no AVX)
#   extra  test-quantize-fns, llama-quantize, llama-gguf-split and llama-imatrix from the same build tree (a second
#          tarball), and the CPU check: test-quantize-fns over every type, N2P-128 and N3P-128 included
#   kld    llama-perplexity and test-backend-ops for compute 8.6 (the build machine's own cards: KL divergence, the native
#          MMQ check), in WORK/kld-<sha>/bin
# Usage: WORK=<build folder; its llama.cpp clone in WORK/src has the branch's remote> REPO=<this repository>
#   [REF=fork/pascal-p106-v3-b12076-n2p] [JOBS=8] [LOCK=<flock file shared by the machine's builds>]
#   [DOCKER_ARGS='--cpu-shares=256 --label ...'] [STEPS='rig extra kld'] bash build.sh
# Output: WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz} (the extras also in its bin/), WORK/extra-<sha>.tar.gz,
#   WORK/md5.txt, WORK/logs/ (logs/quantize-fns.txt: the CPU check), WORK/DONE (with each step's exit status)
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<build folder>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
REF=${REF:-fork/pascal-p106-v3-b12076-n2p}
BRANCH=${REF##*/}
JOBS=${JOBS:-8}
STEPS=${STEPS:-rig extra kld}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D=${DOCKER_ARGS:-}
mkdir -p "$WORK/logs"
rm -f "$WORK/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
tree() { ls -td "$WORK"/llama.cpp-*-"$BRANCH" 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; } # the newest
status=""

if step rig; then
    echo "=== rig $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "$REF" \
        > "$WORK/logs/rig.log" 2>&1
    rc=$?
    status="$status rig=$rc"
    echo "=== rig ended $(date -Is) rc=$rc"
    if [ $rc = 0 ]; then
        T=$(tree)
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        grep -E '^(name|commit|avx|highest)' "$T/BUILDINFO.txt"
    else
        grep -a -E "error|Error" "$WORK/logs/rig.log" | sort | uniq -c | head -40
    fi
fi

if step extra; then
    echo "=== extra $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-quantize-fns llama-quantize llama-gguf-split llama-imatrix \
            > /work/logs/extra-build.log 2>&1 || { grep -a -E "error" /work/logs/extra-build.log | head -40; exit 1; }
        rm -rf /work/extra && mkdir -p /work/extra
        cp build/bin/test-quantize-fns build/bin/llama-quantize build/bin/llama-gguf-split build/bin/llama-imatrix /work/extra/
        # the CPU check: every type round trip, reference against the fast quantizer, dot products against the
        # reference; the rig CPU has no AVX, so this build has none either
        nice -n 19 build/bin/test-quantize-fns -v > /work/logs/quantize-fns.txt 2>&1; echo "test-quantize-fns rc=$?" >> /work/logs/quantize-fns.txt
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs' > "$WORK/logs/extra.log" 2>&1
    rc=$?
    status="$status extra=$rc"
    if [ $rc = 0 ]; then
        sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
        tar czf "$WORK/extra-$sha.tar.gz" -C "$WORK" extra
        T=$(tree)
        [ -n "$T" ] && cp "$WORK"/extra/* "$T/bin/"
        (cd "$WORK" && md5sum "extra-$sha.tar.gz" && stat -c '%n %s bytes' "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
        grep -a -E "FAIL|failed|rc=" "$WORK/logs/quantize-fns.txt" | head -20
        grep -a -E -i "n[23]p" "$WORK/logs/quantize-fns.txt" | head -20
    else
        tail -n 30 "$WORK/logs/extra.log"
    fi
    echo "=== extra ended $(date -Is) rc=$rc"
fi

if step kld; then
    echo "=== kld $(date -Is)"
    # the quant machine's own cards (RTX 3080/3090, compute 8.6): llama-perplexity for the KL divergence runs, which
    # need the branch's types, and test-backend-ops for the native MMQ check (the rig build's compute 6.1 code can't
    # run MMQ correctly on these cards); a separate build folder in the same tree
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        sha=$(git rev-parse --short=9 HEAD)
        cmake -B build-86 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
            -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > /dev/null
        nice -n 19 cmake --build build-86 -j "$JOBS" --target llama-perplexity test-backend-ops > /work/logs/kld-build.log 2>&1 ||
            { grep -a -E "error" /work/logs/kld-build.log | sort | uniq -c | head -40; exit 1; }
        out=/work/kld-$sha && rm -rf "$out" && mkdir -p "$out/bin"
        cp build-86/bin/llama-perplexity build-86/bin/test-backend-ops "$out/bin/"
        git log -1 --format="commit: %H (%ci)" > "$out/BUILDINFO.txt"
        chown -R "$HOST_UID:$HOST_GID" "$out" /work/logs' > "$WORK/logs/kld.log" 2>&1
    rc=$?
    status="$status kld=$rc"
    [ $rc = 0 ] && (cd "$WORK" && md5sum kld-*/bin/* | tee -a "$WORK/md5.txt") || tail -n 30 "$WORK/logs/kld.log"
    echo "=== kld ended $(date -Is) rc=$rc"
fi

echo "$(date -Is)$status" > "$WORK/DONE"
cat "$WORK/DONE"
