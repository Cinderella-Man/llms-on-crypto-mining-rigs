#!/usr/bin/env bash
# C1139 (Q1872): why gmix-q2 reads prompts about a third slower than pmix-q2-fast, by matrix product and weight type.
# R125's lane (../../2026-11-25_r125_gmix_step/run/r125.sh) on Flash-Next's v5 preset (release/flash-next-v5/
# settings.env) with the prompts in focus: the deep lane (bench/deep_lane) primes each depth with the prompt minus its
# tail, so at depths 0 8192 9216 one load reads two prompts of its own:
#   - the 8,192 prime: 8,128 tokens from an empty cache (R125's long prime), two calls of up to 4,096 tokens in
#     micro-batches of 512 (-b 4096 -ub 512);
#   - the 9,216 prime: 1,023 new tokens on top of 8,128 cached ones, one call under the preset's taper
#     (--ubatch-taper short=1400:64), so in micro-batches of 64, the size that loses the most on the agent check.
# Each depth then runs R125's generation requests (the prompt twice at temperature 0, 256 tokens, once with the
# preset's sampler), so the replies can be compared with R125's and C1146's.
# One load per file and profiler (p = pmix-q2-fast, g = gmix-q2 with GGML_CUDA_MTOK_IQ_FAST=2, the PTX grid kernels of
# the four-token verify, as in C1173):
#   pg, gg   the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group: every launch group timed, the stream
#            synchronized after every evaluation (the GPU time by part, product and weight type)
#   ph, gh   the preset + GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE: each card's host and GPU time per evaluation,
#            nothing synchronized (where the host waits inside an evaluation)
#   pu, gu   the preset, no profiler (the prompts' own speed)
# Loads: warm:pu (untimed), pg gg ph gh pu1 gu1.
# The model file: the deep lane's MODEL is pmix-q2-fast's first shard; every g arm's ARM_ARGS end in
# "-m <gmix-q2's first shard>", which llama-server takes as the last -m given (as in R125); each server log's model
# loader line names the file it read.
# PAUSE=4 leaves four idle seconds before every request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (llama-server on the rig), MODEL
# (pmix-q2-fast's first shard on the rig), GMIX (gmix-q2's first shard on the rig), MODEL_DRAFT (the MTP draft on the
# rig), TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect),
# DEPTHS (default 0 8192 9216), LOADS (to run a subset).
# STEP=dry|start|wait|status|collect|stop; dry prints every arm's server line and checks that each g arm's line is
# its p arm's plus GGML_CUDA_MTOK_IQ_FAST=2 (before the binary) and "-m GMIX" (at the end).
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c1139}
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="pu gu pg gg ph gh"
FAST=GGML_CUDA_MTOK_IQ_FAST=2
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
lane() {
  : "${BIN:?}" "${MODEL:?}" "${GMIX:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      ?u) env_a= ;;
      ?g) env_a=$(group "$a") ;;
      ?h) env_a=$(host "$a") ;;
    esac
    case $a in g?) env_a="${env_a:+$env_a }$FAST"; export "ARM_ARGS_$a=-m $GMIX" ;; esac
    [ -n "$env_a" ] && export "ARM_ENV_$a=$env_a"
  done
  PRESET=release/flash-next-v5/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 8192 9216}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-warm:pu pg:pg gg:gg ph:ph gh:gh pu1:pu gu1:gu}" \
    bash $L start
}
case ${STEP:-start} in
dry)
  out=$(DRY=1 lane 2>&1)
  echo "$out"
  grep -q 'DRY=1: nothing started' <<< "$out" || { echo "C1139: the dry run failed"; exit 1; }
  for x in u g h; do
    p=$(grep -A1 " arm p$x:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    g=$(grep -A1 " arm g$x:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    want=${p//-p$x./-g$x.}  # the profile files carry the arm's name
    want="${want/ $BIN / $FAST $BIN } -m $GMIX"
    [ -n "$p" ] && [ "$want" = "$g" ] && echo "C1139: g$x is p$x plus $FAST and -m GMIX" \
      || { echo "C1139: g$x differs from p$x in more than $FAST and -m GMIX"; exit 1; }
  done
  ;;
start) lane ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "usage: STEP=dry|start|wait|status|collect|stop c1139.sh"; exit 1 ;;
esac
