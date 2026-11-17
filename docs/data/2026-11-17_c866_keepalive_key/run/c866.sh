#!/usr/bin/env bash
# C866 (Q1562), the rig check: R192's pair (docs/data/2026-11-17_r192_first_request_graphs/run/r192.sh) on C866's build
# with GGML_CUDA_KEEPALIVE_EAGER=3 in place of the v4 preset's =1, at three depths in one load (4,096, 6,144, 8,192).
# At 3 the keep-alive's key for an evaluation run kernel by kernel leaves the cache's length out and keeps what the
# length picks (which attention and top-k kernels run), so a pass at a new depth should keep its card's clock-holding
# block, as the repeat's do, while the MTP draft's first one-token pass on the sparse attention still stops it (R191:
# without that stop it waited about 1 s on card 8).
# Why three depths in one load: the key changes where a pick changes, and on Flash-Next's file two picks change right at
# 4,096 and 8,192 (the README: the indexer's top-k for 4 and 5 rows from 1,024 pools, about 4,100 cells; the sparse
# attention for 4 tokens from 4 x 2,051 = 8,204 cells). The first request at each of those depths is therefore new on
# its cards and stops as at 1 (once a server, by design). At 6,144, after the 4,096 pair, no pick changes: the first
# request there must keep every block (0 stops) and run as fast as its repeat.
# Everything else is R191's (docs/data/2026-11-17_r191_keepalive_stop/run/r191.sh): the deep lane (bench/deep_lane), per
# depth the prime, the first request and its repeat at temperature 0, 256 tokens, the v4 preset exactly
# (release/flash-next-v4/settings.env), the warm load on the preset as it is, and in the timed load the same logging:
#   GGML_CUDA_GRAPH_DEBUG=1 (with -lv 5), GGML_CUDA_PROFILE + GGML_CUDA_PROFILE_MODE=graph (each evaluation's "eager",
#   "capture" or "replay" and its GPU time on its card), GGML_CUDA_PROFILE_HOST + LLAMA_STEP_TRACE (R121's host profile
#   and step trace)
# The arm's environment goes after the preset's, and env keeps the last value of a name, so the server runs at 3; it
# logs "GGML_CUDA_KEEPALIVE_EAGER=3: ..." at its start and "EAGER=3" in the keep-alive line after each reply.
# Loads: warm:w4u (the preset, depth 0, untimed), then k1. PAUSE=4 leaves four idle seconds before every request.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), DEPTHS (default "4096 6144 8192"), WORK (the run folder on the rig), BIN (C866's llama-server
# on the rig), MODEL (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL (the
# recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every arm's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
DEPTHS=${DEPTHS:-4096 6144 8192}
export RIG WORK=${WORK:-/tmp/c866}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
O=$WORK/out
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  export ARM_ENV_k="GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_PROFILE=$O/gp-k1.csv GGML_CUDA_PROFILE_MODE=graph GGML_CUDA_PROFILE_HOST=$O/hp-k1.csv LLAMA_STEP_TRACE=$O/st-k1.jsonl GGML_CUDA_KEEPALIVE_EAGER=3"
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
