#!/usr/bin/env bash
# R176 (Q1510, Ornith's half): Ornith-1.5-122B Pascal N2P GGUF's decode step on Miner Llama v4.0 taken apart at depth
# 0 and 8,192 tokens, for passes of 1, 2, 3 and 4 tokens. R121's method
# (docs/data/2026-11-17_r121_flash_step_map/run/r121.sh): one detached deep-lane run (bench/deep_lane: prompts of
# exactly DEPTHS tokens cut from recorded agent text, each depth primed, then the whole prompt twice at temperature 0,
# 256 tokens) on the v4 preset (release/ornith-v4/settings.env), with only the profilers' own variables added per arm:
#   w1h1, w1h2 the preset + GGML_CUDA_PROFILE_HOST (each card's host time and GPU time per graph evaluation, nothing
#              synchronized) + LLAMA_STEP_TRACE (one line per server step with the host's marks): the step split
#   w1g1, w1g2 the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group (every launch group timed, the stream
#              synchronized after every evaluation): the kernels
# The preset runs without a draft, so it decodes one token a pass. Passes of 2, 3 and 4 tokens come from Ornith's own
# MTP layer (the file's last block) drafting 1, 2 or 3 tokens a step (--spec-type draft-mtp --spec-draft-n-max K; the
# server's draft probability floor is 0, so every step drafts K tokens and verifies K + 1):
#   w2h1, w2g1 one drafted token: verify passes of 2 tokens
#   w3h1, w3g1 two: passes of 3 tokens
#   w4h1, w4g1 three: passes of 4 tokens
# The MTP arms run on the split that fitted Ornith's MTP line on v3.0 (-ts 6,6,6,6,6,5,5,5,5 -ub 256: the last card
# also holds the MTP layer's draft context; docs/data/2026-11-06_ornith_mtp_rig2, "The fit"), and with a context of
# 16,384 tokens (the deepest prompt here is 8,192 + 256), which leaves room for v4's checkpoints on the cards. A pass
# reads only the cache's used cells, so the context size does not change a step at these depths.
# Loads in this order: the one-token map first (w1h1 w1g1 w1h2 w1g2), then w2h1 w2g1 w3h1 w3g1 w4h1 w4g1. No warm-up
# load (each load's client sends its own untimed warm-up request). PAUSE=4 leaves four idle seconds before every
# request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4.0's llama-server on the rig), MODEL
# (Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig), TEXT_LOCAL (the recorded agent text here: the release
# check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192), LOADS (default above).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r176}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="w1h1 w1h2 w1g1 w1g2 w2h1 w2g1 w3h1 w3g1 w4h1 w4g1"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      *h?) export "ARM_ENV_$a=$(host "$a")" ;;
      *g?) export "ARM_ENV_$a=$(group "$a")" ;;
    esac
    case $a in
      w[234]*)
        export "ARM_ARGS_$a=--spec-type draft-mtp --spec-draft-n-max $((${a:1:1} - 1)) -ts 6,6,6,6,6,5,5,5,5"
        export "ARM_UB_$a=256" "ARM_CTX_$a=16384"
        ;;
    esac
  done
  PRESET=release/ornith-v4/settings.env \
    BIN=$BIN MODEL=$MODEL DRAFT=off TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    SAMPLER_PASS=0 PAUSE=4 WARM=0 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-w1h1:w1h1 w1g1:w1g1 w1h2:w1h2 w1g2:w1g2 w2h1:w2h1 w2g1:w2g1 w3h1:w3h1 w3g1:w3g1 w4h1:w4h1 w4g1:w4g1}" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
