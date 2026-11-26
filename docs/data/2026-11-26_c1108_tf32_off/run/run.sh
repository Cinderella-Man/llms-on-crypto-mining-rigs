#!/usr/bin/env bash
# C1108 (Q1841): cuBLAS as on a P106 in the GPU checks that run the rig build on newer NVIDIA cards: TF32 off
# (NVIDIA_TF32_OVERRIDE=0) and F16 and quantized sources computed in F32 (GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32), together
# scripts/pascal_forced.sh's PASCAL_CUBLAS. One rig build on one card, three steps (STEPS, default all three):
#   archs   run/archs.sh: test-llama-archs -b CUDA0 -s 1 under five settings (stock; TF32 off alone; the forced set
#           before C1108; that set with TF32 off alone; the forced set now), each twice
#   replay  the card suite's MUL_MAT share replayed from C1023's kept record (REC/mul_mat) by C1023's driver
#           (docs/data/2026-11-24_c1023_suite_record/run/run.sh), once per tag of TAGS: "before" with the driver's
#           settings before C1108 (the fork's matrix-vector kernels at a P106's shapes), "tf32" with those and
#           NVIDIA_TF32_OVERRIDE=0, "after" with its default now (the kernels and PASCAL_CUBLAS); no cpu-check (the
#           record is not in question here)
#   pipe    bench/pipespec/gpu_pipe_check.sh on the build, this one card (layouts "one virt"), its default arms and
#           checks, once per tag into WORK/pipe-<tag>: "before" with CARD_ENV= (cuBLAS as the card runs it), "tf32"
#           with CARD_ENV=NVIDIA_TF32_OVERRIDE=0, "after" with its default (PASCAL_CUBLAS's two settings)
# Correctness only: nothing is timed but each step's wall time. All three share one PTX JIT cache, WORK/jitcache.
#
#   WORK=<out folder> BIN=<the build's bin folder> GPU=<NVML index of one card> REPO=<a copy of this repository>
#   REC=<the kept record's folder, holding mul_mat/> [JIT_SEED=<a PTX JIT cache of the same build to copy first>]
#   [PREFLIGHT=<cuda_preflight.sh>] [LABEL=c1108=1] [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [STEPS='archs replay pipe']
#   [TAGS='before tf32 after'] [MOUNTS='<folders the containers read>'] bash run.sh
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/run.log 2>&1 < /dev/null &
# Output: WORK/archs/ (archs.sh's), WORK/bops/ and WORK/settings-<tag>.txt (C1023's driver), WORK/pipe-<tag>/
# (gpu_pipe_check.sh's), WORK/steps.tsv (step, exit status, seconds), WORK/run-md5.txt, WORK/run.done at the end (its
# exit status).
set -u
: "${WORK:?}" "${BIN:?}" "${GPU:?}" "${REPO:?}" "${REC:?}"
LABEL=${LABEL:-c1108=1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MYSTEPS=${STEPS:-archs replay pipe}
TAGS=${TAGS:-before tf32 after}
PRE=${PREFLIGHT:-}
SEED=${JIT_SEED:-}
# the drivers below read some of the same names from the environment: they get only what this script passes them
unset STEPS PREFLIGHT JIT_SEED RUNS
HERE=$(cd "$(dirname "$0")" && pwd)
DARGS="--label pascal-quants=1 --label $LABEL --cpu-shares=256"
BUILD=$(dirname "$BIN")
mkdir -p "$WORK/jitcache"
rm -f "$WORK/run.done"
say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $MYSTEPS " == *" $1 "* ]]; }
step() { # <name> <timeout seconds> <command ...>: one line of steps.tsv
    local name=$1 lim=$2 t0 rc
    shift 2
    t0=$(date +%s)
    say "$name: start"
    timeout "$lim" "$@"
    rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$(($(date +%s) - t0))" >> "$WORK/steps.tsv"
    say "$name: rc=$rc (124: timed out) in $(($(date +%s) - t0)) s"
}
[ -f "$WORK/steps.tsv" ] || printf 'step\trc\tseconds\n' > "$WORK/steps.tsv"
[ -n "$SEED" ] && [ -d "$SEED" ] && [ -z "$(ls -A "$WORK/jitcache")" ] && cp -a "$SEED/." "$WORK/jitcache/"

# shellcheck source=../../../../scripts/pascal_forced.sh
. "$REPO/scripts/pascal_forced.sh" || exit 1
md5sum "$REPO/scripts/pascal_forced.sh" "$HERE"/* "$REPO/docs/data/2026-11-24_c1023_suite_record/run/run.sh" \
    "$REPO/bench/pipespec/gpu_pipe_check.sh" "$BIN/test-llama-archs" "$BIN/test-backend-ops" > "$WORK/run-md5.txt"

if [ -n "$PRE" ]; then
    PREFLIGHT_DOCKER_ARGS="$DARGS" IMAGE="$IMAGE" "$PRE" -g "$GPU" > "$WORK/preflight.txt" 2>&1
    rc=$?
    say "preflight rc=$rc"
    [ $rc = 0 ] || { say "preflight failed: nothing run"; echo 3 > "$WORK/run.done"; exit 3; }
fi

if has archs; then
    step archs 1800 env WORK="$WORK/archs" BIN="$BIN" GPU="$GPU" PF="$REPO/scripts/pascal_forced.sh" \
        JITCACHE="$WORK/jitcache" IMAGE="$IMAGE" LABEL="$LABEL" bash "$HERE/archs.sh"
fi

if has replay; then
    # C1023's settings before C1108: today's PASCAL_FORCED_MMV without PASCAL_CUBLAS (the same six switches)
    before=""
    for kv in $PASCAL_FORCED_MMV; do [[ " $PASCAL_CUBLAS " == *" $kv "* ]] || before+="$kv "; done
    R=$REPO/docs/data/2026-11-24_c1023_suite_record/run/run.sh
    for tag in $TAGS; do
        on=()
        [ $tag = before ] && on=(ON="${before% }")
        [ $tag = tf32 ] && on=(ON="NVIDIA_TF32_OVERRIDE=0 ${before% }")
        step "replay-$tag" 1500 env "${on[@]}" WORK="$WORK" NEW="$BIN" REC="$REC" BUILDS="$tag=$BIN" MOUNTS="$BUILD" \
            DOCKER_ARGS="$DARGS" GPU="$GPU" IMG="$IMAGE" STEPS=replay SHARES=mul_mat NAME=c1108 ALLOW_MISSING=1 \
            CPU_CHECK=0 bash "$R"
        cp "$WORK/settings.txt" "$WORK/settings-$tag.txt"
    done
fi

if has pipe; then
    for tag in $TAGS; do
        card=()
        [ $tag = before ] && card=(CARD_ENV=)
        [ $tag = tf32 ] && card=(CARD_ENV=NVIDIA_TF32_OVERRIDE=0)
        step "pipe-$tag" 1800 env "${card[@]}" WORK="$WORK/pipe-$tag" BUILDS="land=$BIN" GPUS="$GPU" MOUNTS="$BUILD" \
            DOCKER_ARGS="$DARGS" JITCACHE="$WORK/jitcache" IMG="$IMAGE" NAME="c1108-pipe-$tag" \
            bash "$REPO/bench/pipespec/gpu_pipe_check.sh"
    done
fi

say "done: our containers left: $(docker ps -q --filter "label=$LABEL" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
echo 0 > "$WORK/run.done"
