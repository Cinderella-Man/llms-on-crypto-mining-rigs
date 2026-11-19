#!/usr/bin/env bash
# R197 (Q1580): Ornith-1.5-122B Pascal N2P GGUF's passes of 2, 3 and 4 tokens on the v5 engine with Ornith's v5 preset
# (release/ornith-v5/settings.env), arm "base" (the preset) against arm "ro" (the preset plus the read-once switches
# that reach Ornith's Q5_0 and Q8_0 dense products: GGML_CUDA_MMVQ_PASCAL_MT=1, GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0,
# GGML_CUDA_MMVQ_PASCAL_GLU_MT=1). R176's method (docs/data/2026-11-17_r176_ornith_step_map/run/r176.sh): one
# detached deep-lane run (bench/deep_lane), depths 0 and 8,192, each depth primed, then the whole prompt twice at
# temperature 0, 256 tokens. Passes of k tokens come from Ornith's own MTP layer drafting k - 1 tokens a step
# (--spec-type draft-mtp --spec-draft-n-max k-1; the server's draft probability floor is 0, so every step drafts k - 1
# and verifies k), on the split that fitted Ornith's MTP line on v3.0 (-ts 6,6,6,6,6,5,5,5,5 -ub 256) with a context of
# 16,384 tokens, as R176. The arms, each one load:
#   w1h1        the preset (one token a pass) + GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE (nothing synchronized): the
#               one-token step on the rig's current boot and clocks; it runs first, so it is also the warm-up of the
#               alternated loads after it
#   w<k>b1, w<k>b2   arm base at width k (2, 3, 4) + GGML_CUDA_PROFILE with GGML_CUDA_PROFILE_MODE=group (every launch
#               group timed, the stream synchronized after every evaluation): the kernels
#   w<k>r1, w<k>r2   arm ro at width k, the same profiler
# Order (LOADS): w1h1, then w2b1 w2r1 w3b1 w3r1 w4b1 w4r1, then the second round the other way round within each width:
# w2r2 w2b2 w3r2 w3b2 w4r2 w4b2, so each arm's two loads of a width are an hour apart and drift shows in their spread.
# No deep-lane warm-up load (each load's client sends its own untimed warm-up request). PAUSE=4 leaves four idle seconds
# before every request, so each profile cuts into one piece per request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the v5 engine's llama-server on the rig),
# MODEL (Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig), TEXT_LOCAL (the recorded agent text here: the
# release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 8192), LOADS (default above).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r197}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
RO="GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1"
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
