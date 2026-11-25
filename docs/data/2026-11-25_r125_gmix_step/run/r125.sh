#!/usr/bin/env bash
# R125 (Q1409): where gmix-q2's extra time a step goes. R121's per-part profile of the four-token verify pass
# (../../2026-11-17_r121_flash_step_map/run/r121.sh: GGML_CUDA_PROFILE in group mode for the launch groups and R41's
# parts, GGML_CUDA_PROFILE_HOST with LLAMA_STEP_TRACE for the step split) on Flash-Next's v5 preset
# (release/flash-next-v5/settings.env), one load of each kind with pmix-q2-fast and one with gmix-q2; then R115's step
# pair (../../2026-11-12_r115_gmix_step/run/r115_lane.sh) again on both files, with and without
# GGML_CUDA_MMVQ_PASCAL_MT=1 (the 2-4-token read-once kernels for the products the multi-token expert and k-column
# kernels decline). One detached deep-lane run (bench/deep_lane: prompts of exactly DEPTHS tokens cut from recorded
# agent text, each depth primed, then the prompt twice at temperature 0, 256 tokens, and once with the preset's
# sampler).
# The model file: the deep lane's MODEL is pmix-q2-fast's first shard; every gmix arm's ARM_ARGS end in
# "-m <gmix-q2's first shard>", which llama-server takes as the last -m given (as in R115); each server log's model
# loader line names the file it read.
# Arms (p = pmix-q2-fast, g = gmix-q2):
#   pu, gu   the preset, no profiler
#   pg, gg   the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group: every launch group timed, the stream
#            synchronized after every evaluation (the parts and the products)
#   ph, gh   the preset + GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE: each card's host and GPU time per evaluation,
#            nothing synchronized (the step split)
#   pm, gm   the preset + GGML_CUDA_MMVQ_PASCAL_MT=1, no profiler
# Loads: warm:pu (untimed), pg gg ph gh, then the step pair pu1 gu1 pm1 gm1.
# PAUSE=4 leaves four idle seconds before every request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (llama-server on the rig), MODEL
# (pmix-q2-fast's first shard on the rig), GMIX (gmix-q2's first shard on the rig), MODEL_DRAFT (the MTP draft on the
# rig), TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect),
# DEPTHS (default 0 8192, R115's), LOADS (to run a subset).
# STEP=dry|start|wait|status|collect|stop; dry prints every arm's server line and checks that each gmix arm's line is
# its pmix arm's plus "-m GMIX".
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r125}
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="pu gu pg gg ph gh pm gm"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
lane() {
  : "${BIN:?}" "${MODEL:?}" "${GMIX:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      ?u) env_a= ;;
      ?g) env_a=$(group "$a") ;;
      ?h) env_a=$(host "$a") ;;
      ?m) env_a=GGML_CUDA_MMVQ_PASCAL_MT=1 ;;
    esac
    [ -n "$env_a" ] && export "ARM_ENV_$a=$env_a"
    case $a in g?) export "ARM_ARGS_$a=-m $GMIX" ;; esac
  done
  PRESET=release/flash-next-v5/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-warm:pu pg:pg gg:gg ph:ph gh:gh pu1:pu gu1:gu pm1:pm gm1:gm}" \
    bash $L start
}
case ${STEP:-start} in
dry)
  out=$(DRY=1 lane 2>&1)
  echo "$out"
  grep -q 'DRY=1: nothing started' <<< "$out" || { echo "R125: the dry run failed"; exit 1; }
  for x in u g h m; do
    p=$(grep -A1 " arm p$x:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    g=$(grep -A1 " arm g$x:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    want=${p//-p$x./-g$x.}  # the profile files carry the arm's name
    [ -n "$p" ] && [ "$want -m $GMIX" = "$g" ] && echo "R125: g$x is p$x plus -m GMIX" || { echo "R125: g$x differs from p$x in more than -m"; exit 1; }
  done
  ;;
start) lane ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "usage: STEP=dry|start|wait|status|collect|stop r125.sh"; exit 1 ;;
esac
