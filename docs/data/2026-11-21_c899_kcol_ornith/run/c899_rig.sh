#!/usr/bin/env bash
# C899's rig check (Q1625): Ornith-1.5-122B Pascal N2P GGUF's passes of 2, 3 and 4 tokens on the C899 build
# (pascal-p106-v5-kcol-ornith), arm "base" (Ornith's v5 preset, release/ornith-v5/settings.env, GGML_CUDA_MMVQ_PASCAL_MT
# unset) against the rule arm (the preset plus GGML_CUDA_MMVQ_PASCAL_MT=2: the read-once kernel only for the dense Q5_0
# and Q8_0 products a P106 ran faster on it in R197). R197's run (docs/data/2026-11-19_r197_ornith_passes/run/r197.sh)
# with one change, the second arm's switches; its loads keep R197's names (w<k>r<n> is the rule arm here), so R197's
# run/tables.sh and run/compare.py read this run unchanged (their "ro" column is the rule arm). The arms, each one load:
#   w1h1        the preset (one token a pass) + GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE: the one-token step and reply
#               on this boot (the release file's identity: R197's and R176's replies at one token), and the warm-up
#   w<k>b1, w<k>b2   arm base at width k (2, 3, 4) + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group
#   w<k>r1, w<k>r2   the rule arm at width k, the same profiler
# Order (LOADS): w1h1, then w2b1 w2r1 w3b1 w3r1 w4b1 w4r1, then w2r2 w2b2 w3r2 w3b2 w4r2 w4b2 (R197's).
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the C899 build's llama-server on the rig),
# MODEL (Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig), TEXT_LOCAL (the recorded agent text here: the
# release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192), LOADS (default above).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c899}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
RO="GGML_CUDA_MMVQ_PASCAL_MT=2"
ARMS="w1h1 w2b1 w2r1 w3b1 w3r1 w4b1 w4r1 w2r2 w2b2 w3r2 w3b2 w4r2 w4b2"
host() { echo "GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
group() { echo "GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=group"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  loads=''
  for a in $ARMS; do
    case $a in
      w1h?) export "ARM_ENV_$a=$(host "$a")" ;;
      w?b?) export "ARM_ENV_$a=$(group "$a")" ;;
      w?r?) export "ARM_ENV_$a=$(group "$a") $RO" ;;
    esac
    case $a in
      w[234]*)
        export "ARM_ARGS_$a=--spec-type draft-mtp --spec-draft-n-max $((${a:1:1} - 1)) -ts 6,6,6,6,6,5,5,5,5"
        export "ARM_UB_$a=256" "ARM_CTX_$a=16384"
        ;;
    esac
    loads="$loads $a:$a"
  done
  PRESET=release/ornith-v5/settings.env \
    BIN=$BIN MODEL=$MODEL DRAFT=off TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    SAMPLER_PASS=0 PAUSE=4 WARM=0 \
    DEPTHS="${DEPTHS:-0 8192}" N_PREDICT=256 \
    ARMS="$ARMS" \
    LOADS="${LOADS:-${loads# }}" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
