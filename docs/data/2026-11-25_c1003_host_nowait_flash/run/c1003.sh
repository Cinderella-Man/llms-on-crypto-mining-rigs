#!/usr/bin/env bash
# C1003 (Q1736): the host profile without waiting (GGML_CUDA_PROFILE_HOST_NOWAIT=1, C877) on a P106, on Flash-Next's
# v5 preset. R121's deep-lane driver (docs/data/2026-11-17_r121_flash_step_map/run/r121.sh) with its profile way, cut
# to the three loads this row asks for plus one: prompts of exactly DEPTHS tokens cut from recorded agent text, each
# depth primed, then the whole prompt twice at temperature 0, 256 tokens, and once with the preset's sampler, on
# release/flash-next-v5/settings.env exactly with its MTP draft, with only the profilers' own variables added per arm:
#   u    the preset, no profiler (the warm-up load, the first timed load and the last: the step unprofiled, twice,
#        so that the profiled loads sit between two unprofiled ones)
#   hw   the preset + GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE: the host profile the default way, which waits at each
#        entry for the card's earlier evaluations before it writes them (R121's way)
#   hn   the same + GGML_CUDA_PROFILE_HOST_NOWAIT=1: the profile writes only the evaluations whose end event has
#        passed and never waits
# Loads: warm:u (depth 0 only, untimed), then u1 hw1 hn1 u2. PAUSE=4 leaves four idle seconds before every request,
# so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the engine's llama-server on the rig),
# MODEL (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL (the recorded agent
# text here: the v3.0 release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c1003}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="u hw hn"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_hw="$(host hw)"
  export ARM_ENV_hn="$(host hn) GGML_CUDA_PROFILE_HOST_NOWAIT=1"
  PRESET=release/flash-next-v5/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="warm:u u1:u hw1:hw hn1:hn u2:u" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
