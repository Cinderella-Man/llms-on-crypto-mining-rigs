#!/usr/bin/env bash
# C1146 (Q1871): the IQ2_XXS, IQ2_S, IQ2_XS and IQ1_M multi-token routed gate/up kernels on a P106, the old kernels
# against their rewrites (GGML_CUDA_MTOK_IQ_FAST unset: the old kernels; 1: the fast kernels in CUDA C; 2: the same
# kernels in hand-written PTX; the switch moves the four types at once). gmix-q2 on Flash-Next's v5 preset
# (release/flash-next-v5/settings.env), one build: R125's group profile (../../2026-11-25_r125_gmix_step/run/r125.sh,
# its gg load: GGML_CUDA_PROFILE in group mode, every launch group of the four-token verify pass timed) once at each
# setting, then R125's step pair (its gu1 load: no profiler) at each setting, the unset one twice, first and last.
# One detached deep-lane run (bench/deep_lane: prompts of exactly DEPTHS tokens cut from recorded agent text, each
# depth primed, then the prompt twice at temperature 0, 256 tokens, and once with the preset's sampler).
# Arms:
#   g0, g1, g2   the preset + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group; the switch unset, =1, =2
#   u0, u1, u2   the preset, no profiler; the switch unset, =1, =2
#   p0           the preset with the release-side file pmix-q2-fast (PMIX: its first shard, given as the last -m,
#                which llama-server takes, as in R125), no profiler, the switch unset (it moves no Q2_0 product)
# Loads: warm:u0 (untimed), g0 g1 g2, then u0 u1 u2 u0b (u0b is u0 again, for the spread of two loads). The second
# lane (LOADS="u2b:u2 p0:p0", WORK another folder, PMIX set) adds a second load at 2 and pmix-q2-fast on the same build.
# PAUSE=4 leaves four idle seconds before every request, so each trace cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (llama-server on the rig, the build with
# GGML_CUDA_MTOK_IQ_FAST), MODEL (gmix-q2's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig),
# TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect),
# DEPTHS (default 0 8192, R125's), LOADS (to run a subset), PMIX (pmix-q2-fast's first shard on the rig; adds arm p0).
# STEP=dry|start|wait|status|collect|stop; dry prints every arm's server line and checks that each arm's line is u0's
# with only its own variables added (p0's: u0's plus "-m PMIX").
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c1146}
L=bench/deep_lane/remote.sh
O=$WORK/out
ARMS="u0 g0 g1 g2 u1 u2${PMIX:+ p0}"
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
fast() { case $1 in ?0) ;; ?1) echo GGML_CUDA_MTOK_IQ_FAST=1 ;; ?2) echo GGML_CUDA_MTOK_IQ_FAST=2 ;; esac; }
lane() {
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  for a in $ARMS; do
    case $a in
      u?) env_a=$(fast "$a") ;;
      g?) env_a="$(group "$a") $(fast "$a")"; env_a=${env_a% } ;;
      p0) env_a=; export "ARM_ARGS_p0=-m $PMIX" ;;
    esac
    [ -n "$env_a" ] && export "ARM_ENV_$a=$env_a"
  done
  PRESET=release/flash-next-v5/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 PAUSE=4 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-warm:u0 g0:g0 g1:g1 g2:g2 u0:u0 u1:u1 u2:u2 u0b:u0}" \
    bash $L start
}
case ${STEP:-start} in
dry)
  out=$(DRY=1 lane 2>&1)
  echo "$out"
  grep -q 'DRY=1: nothing started' <<< "$out" || { echo "C1146: the dry run failed"; exit 1; }
  base=$(grep -A1 " arm u0:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
  [ -n "$base" ] || { echo "C1146: no line for u0"; exit 1; }
  for a in g0 g1 g2 u1 u2; do
    line=$(grep -A1 " arm $a:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    add=$(case $a in u?) fast "$a" ;; g?) e="$(group "$a") $(fast "$a")"; echo "${e% }" ;; esac)
    # the arm's variables come right after the preset's, before the binary
    want=${base/ $BIN / $add $BIN }
    [ "$line" = "$want" ] && echo "C1146: $a is u0 plus $add" || { echo "C1146: $a differs from u0 in more than $add"; exit 1; }
  done
  if [ -n "${PMIX:-}" ]; then
    line=$(grep -A1 " arm p0:" <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    [ "$line" = "$base -m $PMIX" ] && echo "C1146: p0 is u0 plus -m PMIX" || { echo "C1146: p0 differs from u0 in more than -m PMIX"; exit 1; }
  fi
  ;;
start) lane ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "usage: STEP=dry|start|wait|status|collect|stop c1146.sh"; exit 1 ;;
esac
