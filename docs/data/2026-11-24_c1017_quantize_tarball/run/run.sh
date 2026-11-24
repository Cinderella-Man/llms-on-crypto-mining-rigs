#!/usr/bin/env bash
# C1017 (Q1750) and C1012 (Q1745), 2026-11-24: llama-quantize and the fork's other hand-run tests in every TESTS=1
# tarball of scripts/build_llamacpp_docker.sh, so the GPU pipe check (bench/pipespec/gpu_pipe_check.sh) needs nothing
# outside the tarball. Run on a build machine with three newer NVIDIA cards (RTX 3090). Correctness only: nothing here
# is timed.
# Steps (STEPS, in this order):
#   src      a local clone of SEED (a llama.cpp clone that holds HEAD_SHA), the fork as remote "fork", REF fetched and
#            checked against HEAD_SHA; the compiler cache hard-linked from RIG_CCACHE (the cache that built REF_BIN)
#   rig      the rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1, no EXTRA_TARGETS, the web page pinned
#            (HF_UI_VERSION, as REF_BIN was built), under LOCK; its md5 and bytes, its binaries' md5s and bytes, and every
#            binary REF_BIN also has compared by md5
#   pipe     gpu_pipe_check.sh with QUANT unset, the new tarball's bin/ as base and as head, its default arms and checks;
#            the PTX JIT cache copied from JIT_SEED
#   compare  the quantized copies' md5s (models.md5) against REF_MODELS, the same seed's copies made by another
#            llama-quantize
# Usage: W=<work dir> SEED=<llama.cpp clone holding HEAD_SHA> REPO=<a copy of this repository> REF_BIN=<an earlier rig
#   build's bin/ of HEAD_SHA> [RIG_CCACHE=<dir>] [REF_MODELS=<a models.md5 of gpu_pipe_check.sh, same MODEL_SEED>]
#   [JIT_SEED=<a PTX JIT cache to copy>] [LOCK=<flock file>] [PREFLIGHT=<cuda_preflight.sh>] [GPUS=0,1,2]
#   [STEPS=...] [JOBS=8] [NAME=c1017] bash run.sh
# Every container: labels pascal-quants=1 and $NAME=1, --cpu-shares=256; the build with --memory=16g.
# Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone holding HEAD_SHA>}
REPO=${REPO:?REPO=<a copy of this repository>}
REF_BIN=${REF_BIN:?REF_BIN=<an earlier rig build bin/ of HEAD_SHA>}
REF=${REF:-fork/pascal-p106-v5}
HEAD_SHA=${HEAD_SHA:-af00bb13b0fd59612bcab545964b61978f04b8d6}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12483}
STEPS=${STEPS:-src rig pipe compare}
JOBS=${JOBS:-8}
GPUS=${GPUS:-0,1,2}
NAME=${NAME:-c1017}
PREFLIGHT=${PREFLIGHT:-}
LBL="--label pascal-quants=1 --label $NAME=1"
mkdir -p "$W/rig" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter "label=$NAME=1" | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# <new md5 list> <reference md5 list>: per name of the reference, same, differs or missing in the new list
same() { awk 'NR == FNR { h[$2] = $1; next } { n = ($2 in h) ? h[$2] : "missing"
    print $2 "\t" (n == $1 ? "same" : (n == "missing" ? "missing" : "differs")) "\t" $1 "\t" n }' "$1" "$2"; }
build_dir() { ls -d "$1"/llama.cpp-*-"${2:0:9}"* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
export LC_ALL=C
echo "=== start $(date -Is): $REF ($HEAD_SHA); steps: $STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
        git -C "$S" fetch -q "$SEED" 'refs/remotes/*:refs/remotes/*' 'refs/tags/*:refs/tags/*' || finish 1
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    t0=$(date +%s)
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS= \
        DOCKER_ARGS="$LBL --cpu-shares=256 --memory=16g --memory-swap=16g -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    rc=$?
    echo "rig build rc=$rc in $(($(date +%s) - t0)) s (the lock wait included)" | tee "$W/rig/RC"
    T=$(build_dir "$W/rig" "$HEAD_SHA")
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    echo "$T" > "$W/rig/build.txt"
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    (cd "$T/bin" && md5sum *) > "$W/checks/bin-md5-new.txt"
    (cd "$REF_BIN" && md5sum *) > "$W/checks/bin-md5-ref.txt"
    (cd "$T/bin" && stat -c '%n %s' * | awk '{ print $1 "\t" $2 }') > "$W/checks/bin-bytes-new.tsv"
    (cd "$REF_BIN" && stat -c '%n %s' * | awk '{ print $1 "\t" $2 }') > "$W/checks/bin-bytes-ref.tsv"
    # every binary of REF_BIN: the same md5 in the new build, another one, or missing
    same "$W/checks/bin-md5-new.txt" "$W/checks/bin-md5-ref.txt" > "$W/checks/bin-compare.tsv"
    echo "binaries of the earlier build: $(cut -f2 "$W/checks/bin-compare.tsv" | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
    echo "new binaries only in this build: $(comm -13 <(awk '{print $2}' "$W/checks/bin-md5-ref.txt" | sort) <(awk '{print $2}' "$W/checks/bin-md5-new.txt" | sort) | paste -sd' ')"
    grep -aE '^(name|commit|web page|check tools):' "$T/BUILDINFO.txt"
    echo "=== rig done $(date -Is)"
fi

if step pipe; then
    T=$(cat "$W/rig/build.txt" 2>/dev/null) || { echo "no rig build"; finish 1; }
    P=$W/pipe
    mkdir -p "$P"
    [ -d "$P/jitcache" ] || { [ -n "${JIT_SEED:-}" ] && cp -a "$JIT_SEED" "$P/jitcache"; }
    [ -z "$PREFLIGHT" ] || "$PREFLIGHT" -g "$GPUS" > "$P/preflight.txt" 2>&1 || { cat "$P/preflight.txt"; finish 1; }
    t0=$(date +%s)
    env -u QUANT -u SEED WORK="$P" BUILDS="base=$T/bin head=$T/bin" GPUS="$GPUS" MOUNTS="$W/rig" PREFLIGHT="$PREFLIGHT" \
        DOCKER_ARGS="$LBL --cpu-shares=256" NAME="$NAME-pipe" \
        bash "$REPO/bench/pipespec/gpu_pipe_check.sh" > "$W/logs/pipe.log" 2>&1
    rc=$?
    echo "pipe check rc=$rc in $(($(date +%s) - t0)) s"
    for t in base head; do echo "$t: $(grep -a llama-quantize "$P/$t/bin.md5")"; done
    echo "verdicts: $(tail -n +2 "$P/summary.tsv" | cut -f6 | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
    echo "=== pipe done $(date -Is)"
fi

if step compare; then
    P=$W/pipe
    for t in base head; do grep -a '\.gguf$' "$P/$t/models.md5" | sort -k2 > "$W/checks/models-$t.md5"; done
    if [ -n "${REF_MODELS:-}" ] && [ -f "$REF_MODELS" ]; then
        grep -a '\.gguf$' "$REF_MODELS" | sort -k2 > "$W/checks/models-ref.md5"
        same "$W/checks/models-head.md5" "$W/checks/models-ref.md5" > "$W/checks/models-compare.tsv"
        echo "copies against REF_MODELS: $(cut -f2 "$W/checks/models-compare.tsv" | sort | uniq -c | awk '{printf "%s %s; ", $1, $2}')"
    fi
    cmp -s "$W/checks/models-base.md5" "$W/checks/models-head.md5" && echo "base and head: the same files" || echo "base and head: files differ"
fi
finish 0
