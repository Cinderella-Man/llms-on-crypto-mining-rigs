#!/usr/bin/env bash
# Upstream fixes: build the stages of the upstream-fixes branch (upstream llama.cpp master plus one commit per fix) in
# Docker. One build directory per kind, so each stage after the first rebuilds only what its commit changed:
#   cpu   llama-server, CPU backend, static: for the speculative-decoding statistical test (stat.sh)
#   cuda  test-backend-ops with CUDA for "61-virtual;86-real", static: sm_86 machine code for an RTX 3090 and compute_61
#         PTX, which the driver compiles and runs instead under CUDA_FORCE_PTX_JIT=1
# Usage: SRC=<git tree of the branch> WORK=<dir> STAGES='<kind>:<tag>=<commit> ...' [JOBS=8] [LOCK=<flock file>]
#   [DOCKER_ARGS=...] [CPU_IMAGE=<ubuntu 24.04 with git cmake build-essential>] [CUDA_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04]
#   bash build.sh
# Output: WORK/bin-<kind>-<tag>/ (the binaries), WORK/logs/<kind>-<tag>.{cmake,build}.log, WORK/builds.tsv (kind, tag,
#   commit, tree hash, rc)
set -uo pipefail
SRC=$(realpath "${SRC:?SRC=<git tree>}")
WORK=${WORK:?WORK=<dir>}
STAGES=${STAGES:?STAGES='<kind>:<tag>=<commit> ...'}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-ubuntu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK/logs"
WORK=$(realpath "$WORK")
D="--label pascal-quants=1 --cpu-shares 256 ${DOCKER_ARGS:-}"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
[ -f "$WORK/builds.tsv" ] || printf 'kind\ttag\tcommit\ttree\trc\n' > "$WORK/builds.tsv"
for st in $STAGES; do
    kind=${st%%:*}; rest=${st#*:}; tag=${rest%%=*}; commit=${rest#*=}
    git -C "$SRC" checkout -q --detach "$commit" || { echo "checkout $commit failed"; exit 1; }
    tree=$(git -C "$SRC" rev-parse "HEAD^{tree}")
    B=$WORK/build-$kind
    case $kind in
        cpu)  image=$CPU_IMAGE; targets=llama-server
              cfg="-DGGML_CUDA=OFF -DGGML_NATIVE=ON" ;;
        cuda) image=$CUDA_IMAGE; targets=test-backend-ops
              cfg="-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=61-virtual;86-real" ;;
        *) echo "unknown kind $kind"; exit 1 ;;
    esac
    echo "$(date -Is) build $kind $tag $commit"
    # the values go in as variables: CMAKE_CUDA_ARCHITECTURES holds a ";"
    locked docker run --rm $D -v "$SRC:$SRC:ro" -v "$WORK:$WORK" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -e S="$SRC" -e B="$B" -e CFG="$cfg" -e JOBS="$JOBS" -e TARGETS="$targets" -e L="$WORK/logs/$kind-$tag" \
        "$image" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        command -v cmake > /dev/null || { apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null; }
        cmake -S "$S" -B "$B" -DCMAKE_BUILD_TYPE=Release $CFG -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=ON > "$L.cmake.log" 2>&1 &&
        nice -n 19 cmake --build "$B" -j "$JOBS" --target $TARGETS > "$L.build.log" 2>&1
        rc=$?
        chown -R "$HOST_UID:$HOST_GID" "$B"
        exit $rc'
    rc=$?
    if [ $rc = 0 ]; then
        rm -rf "$WORK/bin-$kind-$tag"; mkdir -p "$WORK/bin-$kind-$tag"
        for t in $targets; do cp "$B/bin/$t" "$WORK/bin-$kind-$tag/"; done
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$tag" "$commit" "$tree" "$rc" >> "$WORK/builds.tsv"
    echo "$(date -Is) build $kind $tag rc $rc"
    [ $rc = 0 ] || exit $rc
done
