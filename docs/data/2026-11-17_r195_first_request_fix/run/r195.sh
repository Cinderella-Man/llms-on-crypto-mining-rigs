#!/usr/bin/env bash
# R195 (Q1562), C866's rig check on the v5 build: R192's deep-lane pair (the first request at a new depth and its repeat)
# at 4,096, 6,144 and 8,192 tokens, in two timed loads on the same build: GGML_CUDA_KEEPALIVE_EAGER=1 (Flash-Next's v4
# preset as released) and =3 (C866: the keep-alive's new-shape key without the cache's length, keeping what the length
# picks). Everything else is C866's runner (docs/data/2026-11-17_c866_keepalive_key/run/c866.sh), which is R191's and
# R192's: the deep lane (bench/deep_lane), per depth the prime, the first request and its repeat at temperature 0, 256
# tokens, the v4 preset exactly (release/flash-next-v4/settings.env), the warm load on the preset as it is, and in each
# timed load the same logging:
#   GGML_CUDA_GRAPH_DEBUG=1 (with -lv 5), GGML_CUDA_PROFILE + GGML_CUDA_PROFILE_MODE=graph (each evaluation's "eager",
#   "capture" or "replay" and its GPU time on its card), GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE (R121's host profile
#   and step trace)
# The arm's environment goes after the preset's, and env keeps the last value of a name. The =1 arm sets the preset's own
# value again, so the two timed loads differ in that one word.
# Why 6,144: on Flash-Next's file two picks change right at 4,096 and 8,192 (the indexer's top-k for 4 and 5 rows from
# 1,024 pools, about 4,100 cells; the sparse attention for 4 tokens from 4 x 2,051 = 8,204 cells), so at =3 the first
# request at those depths is new on its cards and stops as at =1 (once a server, by design). At 6,144, after the 4,096
# pair, no pick changes.
# Loads: warm:w4u (the preset, depth 0, untimed), then e1 (=1), then k1 (=3). PAUSE=4 leaves four idle seconds before
# every request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), DEPTHS (default "4096 6144 8192"), WORK (the run folder on the rig), BIN (the v5
# llama-server on the rig), MODEL (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig),
# TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
DEPTHS=${DEPTHS:-4096 6144 8192}
export RIG WORK=${WORK:-/tmp/r195}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
log() { echo "GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_PROFILE=$O/gp-$1.csv GGML_CUDA_PROFILE_MODE=graph GGML_CUDA_PROFILE_HOST=$O/hp-$1.csv LLAMA_STEP_TRACE=$O/st-$1.jsonl"; }
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_e="$(log e1) GGML_CUDA_KEEPALIVE_EAGER=1"
  export ARM_ENV_k="$(log k1) GGML_CUDA_KEEPALIVE_EAGER=3"
  export ARM_ARGS_e="-lv 5" ARM_ARGS_k="-lv 5"
  PRESET=release/flash-next-v4/settings.env \
  BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=0 PAUSE=4 \
    DEPTHS="$DEPTHS" N_PREDICT=256 \
    ARMS="w4u e k" \
    LOADS="warm:w4u e1:e k1:k" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
