#!/usr/bin/env bash
# C997 (Q1730), the step half: a preset's decode step with GGML_CUDA_TOP_K_SORT unset and set (C985's switch: the
# card's radix top-k hands the sampler its candidates sorted, so one seed gives one reply). One detached deep-lane run
# (bench/deep_lane: prompts of exactly DEPTHS tokens cut from recorded agent text, each depth primed, then the whole
# prompt twice at temperature 0, 256 tokens; R176's lane, docs/data/2026-11-17_r176_flint_step_map/run/r176.sh) on the
# preset exactly, the switch the only change. Arms: unset, sort (ARM_ENV_sort=GGML_CUDA_TOP_K_SORT=1), REPS loads each
# (default 2), alternated unset1 sort1 unset2 sort2; no warm-up load, PAUSE=4 as R176.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (llama-server on the rig), MODEL (the
# preset's file on the rig), PRESET (default release/flint-v5/settings.env), TEXT_LOCAL (the recorded agent text here:
# the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192), REPS (default 2).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c997}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_sort=GGML_CUDA_TOP_K_SORT=1
  PRESET=${PRESET:-release/flint-v5/settings.env} \
  BIN=$BIN MODEL=$MODEL DRAFT=off TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 WARM=0 REPS=${REPS:-2} \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="unset sort" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
