#!/usr/bin/env bash
# R191 (Q1562): R192's pair (docs/data/2026-11-17_r192_first_request_graphs/run/r192.sh) with one change in the timed
# load: GGML_CUDA_KEEPALIVE_EAGER=0 in place of the v4 preset's =1. At 0 the keep-alive switch (C811) only counts the
# evaluations that run kernel by kernel and stops no block (keepalive-policy.h, eager_begin and outside), so it tells
# whether the first request's slow kernel-by-kernel passes at a new depth are the cost of the switch's stops, and
# whether, without the stops, any first launch waits about 1 s for a running block (FN85).
# Everything else is R192's: the deep lane (bench/deep_lane) at depth 4,096 only, the prime, the first request and its
# repeat at temperature 0, 256 tokens, the v4 preset exactly (release/flash-next-v4/settings.env), the warm load on the
# preset as it is, and in the timed load the same logging:
#   GGML_CUDA_GRAPH_DEBUG=1 (with -lv 5), GGML_CUDA_PROFILE + GGML_CUDA_PROFILE_MODE=graph (each evaluation's "eager",
#   "capture" or "replay" and its GPU time on its card), GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE (R121's host profile
#   and step trace)
# The arm's environment goes after the preset's, and env keeps the last value of a name, so the server runs at 0; the
# keep-alive line the server prints after each reply says "EAGER=0".
# Loads: warm:w4u (the preset, depth 0, untimed), then k1. PAUSE=4 leaves four idle seconds before every request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4.0's llama-server on the rig), MODEL
# (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL (the recorded agent text
# here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r191}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_k="GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_PROFILE=$O/gp-k1.csv GGML_CUDA_PROFILE_MODE=graph GGML_CUDA_PROFILE_HOST=$O/hp-k1.csv LLAMA_STEP_TRACE=$O/st-k1.jsonl GGML_CUDA_KEEPALIVE_EAGER=0"
  export ARM_ARGS_k="-lv 5"
  PRESET=release/flash-next-v4/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 \
    DEPTHS=4096 N_PREDICT=256 \
    ARMS="w4u k" \
    LOADS="warm:w4u k1:k" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
