#!/usr/bin/env bash
# C1024's rig half (Q1757): the deep lane (bench/deep_lane) on one P106 rig with one build, two arms that differ only in
# GGML_CUDA_MMQ_J64_I64: unset (upstream b12674's 128-row tile for 64-column prompt products, the control) and 1 (the
# 64-row tile). Each depth's decode pass reads the prompt's 64-token tail in one micro-batch, the case #31352 halves the
# blocks of; the primes read 4,032 and 4,096 tokens in the preset's micro-batches (Flint 128, Flash-Next 512), which
# take the same 64-column tile. Run from the repository root on a machine that only runs ssh, once per model:
#   RIG=<ssh alias> BIN=<llama-server on the rig> MODEL=<model file on the rig> [MODEL_DRAFT=<draft on the rig>] \
#     PRESET=release/<flint|flash-next>-v5/settings.env TEXT_LOCAL=<recorded agent text here> OUT=<folder here> \
#     WORK=<folder on the rig> STEP=start|wait|collect bash docs/data/2026-11-24_c1024_mmq_j64_tile/run/rig_lane.sh
# STEP=start runs a dry run first and starts only when both arms' lines are the preset's, the second with the switch.
# Loads: warm-up (unset), then unset, j64, unset, j64. Depths 0, 4,096 and 8,192 (DEPTHS), 256 tokens a pass, the
# sampler pass, the RAM watchdog at 60 MiB (a 4 GB host).
set -uo pipefail
: "${RIG:?}" "${WORK:?}"
STEP=${STEP:-start}
L=bench/deep_lane/remote.sh
case $STEP in
start)
    : "${BIN:?}" "${MODEL:?}" "${PRESET:?}" "${TEXT_LOCAL:?}"
    lane=(RIG="$RIG" WORK="$WORK" BIN="$BIN" MODEL="$MODEL" PRESET="$PRESET" TEXT_LOCAL="$TEXT_LOCAL"
        TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 N_PREDICT=256 WARM=1 REPS=2
        DEPTHS="${DEPTHS:-0 4096 8192}" ARMS="unset j64" ARM_ENV_j64=GGML_CUDA_MMQ_J64_I64=1)
    [ -n "${MODEL_DRAFT:-}" ] && lane+=(MODEL_DRAFT="$MODEL_DRAFT")
    env "${lane[@]}" DRY=1 bash "$L" start > "${OUT:-.}/c1024-dry.console" 2>&1
    cat "${OUT:-.}/c1024-dry.console"
    grep -q 'DRY=1: nothing started' "${OUT:-.}/c1024-dry.console" || { echo "the dry run failed"; exit 1; }
    # two server lines, the switch in exactly one of them
    [ "$(grep -cE ' env .* -m ' "${OUT:-.}/c1024-dry.console")" = 2 ] \
        && [ "$(grep -cE ' env .*GGML_CUDA_MMQ_J64_I64=1 .* -m ' "${OUT:-.}/c1024-dry.console")" = 1 ] \
        || { echo "the arms' lines are not the preset's with the switch in the j64 arm only"; exit 1; }
    env "${lane[@]}" bash "$L" start
    ;;
wait)
    RIG="$RIG" WORK="$WORK" bash "$L" wait
    ;;
collect)
    : "${OUT:?}"
    RIG="$RIG" WORK="$WORK" OUT="$OUT" bash "$L" collect
    ;;
esac
