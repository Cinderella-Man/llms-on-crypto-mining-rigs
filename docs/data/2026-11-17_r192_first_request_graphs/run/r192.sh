#!/usr/bin/env bash
# R192 (Q1561): which passes of a first request at a new depth run their CUDA graph kernel by kernel, are captured or
# are replayed, against its repeat. One detached deep-lane run (bench/deep_lane) of R121's line
# (docs/data/2026-11-17_r121_flash_step_map/run/r121.sh) at depth 4,096 only: the prime, the first request and its
# repeat at temperature 0, 256 tokens, on the v4 preset exactly (release/flash-next-v4/settings.env), with only logging
# added in the one timed load:
#   GGML_CUDA_GRAPH_DEBUG=1  the CUDA backend's graph decisions (a graph reused, its warm-up complete or reset), which
#                            it logs at debug level, so the server line gets -lv 5 to print them
#   GGML_CUDA_PROFILE + GGML_CUDA_PROFILE_MODE=graph  one line per graph evaluation and card with how it ran ("eager":
#                            kernel by kernel, "capture", "replay"), its graph's number and its GPU time; an event pair
#                            around each evaluation, nothing synchronized
#   GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE  R121's host profile and step trace, so the passes are cut and timed as
#                            in R121
# Loads: warm:w4u (the preset, depth 0, untimed, as in R121), then g1. PAUSE=4 leaves four idle seconds before every
# request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4.0's llama-server on the rig), MODEL
# (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL (the recorded agent text
# here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r192}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_g="GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_PROFILE=$O/gp-g1.csv GGML_CUDA_PROFILE_MODE=graph GGML_CUDA_PROFILE_HOST=$O/hp-g1.csv LLAMA_STEP_TRACE=$O/st-g1.jsonl"
  export ARM_ARGS_g="-lv 5"
  PRESET=release/flash-next-v4/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 \
    DEPTHS=4096 N_PREDICT=256 \
    ARMS="w4u g" \
    LOADS="warm:w4u g1:g" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
