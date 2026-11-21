#!/usr/bin/env bash
# C878's rig load (C926): R195's deep-lane load (docs/data/2026-11-17_r195_first_request_fix/run/r195.sh) on the C878
# build (fork branch pascal-p106-v5-keepalive-why-new) with GGML_CUDA_KEEPALIVE_EAGER=3 and GGML_CUDA_KEEPALIVE_TRACE=1:
# one timed load, the first request at a new depth and its repeat at 4,096 then 6,144 tokens (DEPTHS), each with its
# prime, temperature 0, 256 tokens, Flash-Next's v4 preset exactly (release/flash-next-v4/settings.env) as R195 ran it,
# and R195's logging (GGML_CUDA_GRAPH_DEBUG=1 with -lv 5, GGML_CUDA_PROFILE in graph mode, GGML_CUDA_PROFILE_HOST,
# LLAMA_STEP_TRACE). The server log then holds one "ggml_cuda_keepalive_trace" line for each key new to a card: its
# nearest earlier key on that card, how many nodes differ, the first one and the field that differs there.
# Loads: warm:w4u (the preset, depth 0, untimed), then k1 (=3 with the trace). PAUSE=4 leaves four idle seconds before
# every request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), DEPTHS (default "4096 6144"), WORK (the run folder on the rig), BIN (the C878
# llama-server on the rig), MODEL (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig),
# TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
DEPTHS=${DEPTHS:-4096 6144}
export RIG WORK=${WORK:-/tmp/c878}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
log() { echo "GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=graph GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_k="$(log k1) GGML_CUDA_KEEPALIVE_EAGER=3 GGML_CUDA_KEEPALIVE_TRACE=1"
  export ARM_ARGS_k="-lv 5"
  PRESET=release/flash-next-v4/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 \
    DEPTHS="$DEPTHS" N_PREDICT=256 \
    ARMS="w4u k" \
    LOADS="warm:w4u k1:k" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
