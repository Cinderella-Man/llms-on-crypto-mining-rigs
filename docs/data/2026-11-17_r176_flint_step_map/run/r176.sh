#!/usr/bin/env bash
# R176, Flint's half (Q1510, with Q1543's passes): Flint Pascal Q8 GGUF's decode step on Miner Llama v4.0 taken apart at
# depth 0 and 8,192 tokens, R121's method (docs/data/2026-11-17_r121_flash_step_map/run/r121.sh): one detached
# deep-lane run (bench/deep_lane: prompts of exactly DEPTHS tokens cut from recorded agent text, each depth primed,
# then the whole prompt twice at temperature 0, 256 tokens) on Flint's v4 preset exactly (release/flint-v4/settings.env),
# with only the profilers' variables and, for the passes of several tokens, the speculative flags added per arm.
#
# How Flint gets a pass of several tokens: its preset's pipelined guesses (--pipe-depth 4) are one-token evaluations
# in pipe slots, so the preset never runs one. A pass of w tokens is a batched verify: the server's speculative path
# with free n-gram drafts (ngram-simple, a one-token key, w - 1 drafted tokens) and synthetic acceptance of every draft
# (--spec-synth-len w, the fork's benchmarking switch; Q19's method in bench/eagle3_flint/verify_cost.sh). A slot that
# can speculate never pipelines (pipe_eligible), so these arms verify w tokens a step through the path a drafter's
# verify takes: logits for every row, the sampling chain, the rollback. Their replies are not valid text; only the
# times count. The width is a server flag (per-request draft limits are compiled out in v4.0), so each width is a load.
#
# Arms, each its own load (each writes its own profile files):
#   h1, h2   the preset + GGML_CUDA_PROFILE_HOST (each card's host and GPU time per evaluation, nothing synchronized)
#            + LLAMA_STEP_TRACE (the server's host marks per step): the one-token step as served, pipelined
#   g1, g2   the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group (every launch group timed, the stream
#            synchronized after every evaluation): the kernels by part
#   s8, s4, s2  the preset + passes of 8, 4 and 2 tokens (above) + the host profile and step trace
# Order: h1 g1 s8 s4 s2 h2 g2 (no warm-up load: a Flint load takes about ten minutes). PAUSE=4 leaves four idle seconds
# before every request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4.0's llama-server on the rig), MODEL
# (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (the recorded agent text here: the release check's
# proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192), LOADS (default all seven, in order).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r176}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="h1 g1 s8 s4 s2 h2 g2"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
synth() { echo "--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m $(($1 - 1)) --spec-synth-len $1"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      h?|s?) export "ARM_ENV_$a=$(host "$a")" ;;
      g?) export "ARM_ENV_$a=$(group "$a")" ;;
    esac
    case $a in
      s?) export "ARM_ARGS_$a=$(synth "${a#s}")" ;;
    esac
  done
  PRESET=release/flint-v4/settings.env \
  BIN=$BIN MODEL=$MODEL DRAFT=off TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-h1:h1 g1:g1 s8:s8 s4:s4 s2:s2 h2:h2 g2:g2}" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
