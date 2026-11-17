#!/usr/bin/env bash
# R121 (Q1393, with Q1484 merged in, and Q1133's first-request trace): Flash-Next's decode step on Miner Llama v4.0
# taken apart at depth 0, 4,096 and 8,192 tokens, for verify passes of 1, 4 and 5 tokens. One detached deep-lane run
# (bench/deep_lane: prompts of exactly DEPTHS tokens cut from recorded agent text, each depth primed, then the whole
# prompt twice at temperature 0, 256 tokens, and once with the preset's sampler) on the v4 preset exactly
# (release/flash-next-v4/settings.env), with only the profilers' own variables added per arm:
#   w4u        the preset, no profiler (the warm-up load, and the last load: the step unprofiled)
#   w4h1, w4h2 the preset + GGML_CUDA_PROFILE_HOST (each card's host time and GPU time per graph evaluation, nothing
#              synchronized) + LLAMA_STEP_TRACE (one line per server step with the host's marks): the step split
#   w4g1, w4g2 the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group (every launch group timed, the stream
#              synchronized after every evaluation): the kernels
#   w5*        the same with --spec-draft-n-max 4: verify passes of 5 tokens
#   w1*        the same without the MTP draft (the deep lane's DRAFT off drops -md and every --spec-* flag): passes of
#              1 token
# Each profiled load is its own arm, so that every load writes its own profile files into the results folder.
# Loads: warm:w4u (depth 0 only, untimed), then w4h1 w4g1 w5h1 w5g1 w1h1 w1g1, the same six again, then w4u1.
# PAUSE=4 leaves four idle seconds before every request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4.0's llama-server on the rig), MODEL
# (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL (the recorded agent text
# here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 4096 8192).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r121}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="w4u w4h1 w4h2 w4g1 w4g2 w5h1 w5h2 w5g1 w5g2 w1h1 w1h2 w1g1 w1g2"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      w4u) ;;
      *h?) export "ARM_ENV_$a=$(host "$a")" ;;
      *g?) export "ARM_ENV_$a=$(group "$a")" ;;
    esac
    case $a in
      w5*) export "ARM_ARGS_$a=--spec-draft-n-max 4" ;;
      w1*) export "ARM_DRAFT_$a=off" ;;
    esac
  done
  PRESET=release/flash-next-v4/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 4096 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="warm:w4u w4h1:w4h1 w4g1:w4g1 w5h1:w5h1 w5g1:w5g1 w1h1:w1h1 w1g1:w1g1 w4h2:w4h2 w4g2:w4g2 w5h2:w5h2 w5g2:w5g2 w1h2:w1h2 w1g2:w1g2 w4u1:w4u" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
