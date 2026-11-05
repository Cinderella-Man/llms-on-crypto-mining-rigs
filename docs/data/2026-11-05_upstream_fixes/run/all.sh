#!/usr/bin/env bash
# Upstream fixes: every build and check of this folder in order, as one detached job on a build machine.
#   1. build.sh: llama-server (CPU) at C_MASTER and C_DRAWS
#   2. stat.sh in the background (CPU): the draws reused after a checkpoint restore, master against the fix
#   3. build.sh: test-backend-ops (CUDA, 61-virtual;86-real) at C_TEST (master plus the new test cases) and C_GUARD
#      (plus the row-guard fix)
#   4. guard_reps.sh on one GPU: the new cases REPS times per build on the sm_86 code, then REPS_JIT times through the
#      compute_61 PTX (CUDA_FORCE_PTX_JIT=1)
#   5. the MUL_MAT_ID batch limit through the compute_61 PTX: MUL_MAT_ID_SHARED once per build under CUDA_FORCE_PTX_JIT=1
#   6. stat_check.py on the stat arms
# Usage: SRC=<git tree of the branch> WORK=<dir> REPO=<this repository> MODELS=<dir with st/ and sd/, each holding
#   qwen4exp-moe.gguf: 2026-11-04_block_verify's stat target and drafter> [GPU_ARGS='<docker GPU request for one card>']
#   C_MASTER=<commit> C_TEST=<commit> C_GUARD=<commit> C_DRAWS=<commit> [REPS=30] [REPS_JIT=10] [N_STAT_G=1500]
#   [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [CPU_IMAGE=...] [CUDA_IMAGE=...] [STEPS="cpu stat cuda guard"] bash all.sh
#   STEPS runs a subset (the builds of a step it skips must already be in WORK)
# Output: WORK/{builds.tsv, guard/, stat/, jit_batch.tsv, summary.txt}; WORK/DONE at the end
set -uo pipefail
WORK=${WORK:?}; REPO=$(realpath "${REPO:?}"); MODELS=$(realpath "${MODELS:?}")
mkdir -p "$WORK"; WORK=$(realpath "$WORK"); rm -f "$WORK/DONE"
RUN=$REPO/docs/data/2026-11-05_upstream_fixes/run
BV=$REPO/docs/data/2026-11-04_block_verify/run
CPU_IMAGE=${CPU_IMAGE:-ubuntu:24.04}
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--label pascal-quants=1 --cpu-shares 256 ${DOCKER_ARGS:-}"
export SRC WORK JOBS LOCK DOCKER_ARGS CPU_IMAGE CUDA_IMAGE
echo "=== all started $(date -Is)"
has() { [[ " ${STEPS:-cpu stat cuda guard} " == *" $1 "* ]]; }

has cpu && { STAGES="cpu:master=${C_MASTER:?} cpu:fix=${C_DRAWS:?}" bash "$RUN/build.sh" || { echo "cpu build failed"; touch "$WORK/DONE"; exit 1; } }

STAT_PID=
if has stat; then
echo "=== stat started $(date -Is)"
mkdir -p "$WORK/stat"
docker run --rm $D -v "$WORK:/c" -v "$BV:/r:ro" -v "$RUN:/u:ro" -v "$MODELS:/m:ro" \
    -e B_MASTER=/c/bin-cpu-master/llama-server -e B_FIX=/c/bin-cpu-fix/llama-server \
    -e ST=/m/st/qwen4exp-moe.gguf -e SD=/m/sd/qwen4exp-moe.gguf -e N_STAT_G="${N_STAT_G:-1500}" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$CPU_IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
    bash /u/stat.sh; python3 -I /r/stat_check.py /c/stat > /c/stat/check.txt 2>&1
    chown -R "$HOST_UID:$HOST_GID" /c/stat' > "$WORK/stat-console.txt" 2>&1 &
STAT_PID=$!
fi

has cuda && { STAGES="cuda:test=${C_TEST:?} cuda:fix=${C_GUARD:?}" bash "$RUN/build.sh" || { echo "cuda build failed"; wait $STAT_PID; touch "$WORK/DONE"; exit 1; } }

if has guard; then
# without GPU_ARGS the job waits here for WORK/gpu_args (one line: the docker GPU request), written when a card is free
if [ -z "${GPU_ARGS:-}" ]; then
    echo "=== waiting for $WORK/gpu_args $(date -Is)"
    until [ -s "$WORK/gpu_args" ]; do sleep 60; done
    GPU_ARGS=$(cat "$WORK/gpu_args")
fi
echo "=== guard started $(date -Is)"
gpu() { docker run --rm $D $GPU_ARGS -v "$WORK:$WORK" -v "$RUN:$RUN:ro" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$@"; }
gpu -e BINS="master=$WORK/bin-cuda-test fix=$WORK/bin-cuda-fix" -e WORK="$WORK" -e OUT="$WORK/guard" -e REPS="${REPS:-30}" \
    -e REPS_JIT="${REPS_JIT:-10}" "$CUDA_IMAGE" bash -c '
    nvidia-smi --query-gpu=name,pci.bus_id,driver_version --format=csv,noheader
    LABEL=sass JIT=0 bash '"$RUN"'/guard_reps.sh
    LABEL=jit JIT=1 REPS=$REPS_JIT bash '"$RUN"'/guard_reps.sh
    # the MUL_MAT_ID batch limit through the compute_61 PTX, once per build
    for b in master fix; do
        LABEL=jit-shared-$b JIT=1 REPS=1 OPS=MUL_MAT_ID_SHARED CASES=. BINS="$b=$WORK/bin-cuda-$( [ $b = master ] && echo test || echo fix )" \
            bash '"$RUN"'/guard_reps.sh
    done
    chown -R "$HOST_UID:$HOST_GID" "$OUT"' > "$WORK/guard-console.txt" 2>&1
echo "=== guard done $(date -Is) rc $?"
fi

[ -n "$STAT_PID" ] && wait $STAT_PID
echo "=== stat done $(date -Is)"
{
    echo "== builds"; cat "$WORK/builds.tsv"
    echo "== guard: failures per build and case (sass, jit)"
    for l in sass jit; do
        awk -F'\t' -v l="$l" 'NR > 1 { n[$2 "\t" $4]++; if ($5 == "FAIL") f[$2 "\t" $4]++ }
            END { for (k in n) print l "\t" k "\t" f[k] + 0 "/" n[k] }' "$WORK/guard/$l.tsv" | sort
    done
    echo "== MUL_MAT_ID_SHARED through the compute_61 PTX"
    cat "$WORK"/guard/jit-shared-*.tsv | grep -v '^label' | awk -F'\t' '$5 != "OK"'
    echo "== stat"; grep -E "gram-|^gram|verdict" "$WORK/stat-console.txt" "$WORK/stat/check.txt"
} > "$WORK/summary.txt" 2>&1
touch "$WORK/DONE"
echo "=== all done $(date -Is)"
