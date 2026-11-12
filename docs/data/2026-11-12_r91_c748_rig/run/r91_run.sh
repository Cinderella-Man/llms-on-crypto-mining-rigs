#!/usr/bin/env bash
# R91 (Q1323): C748's KV-only draft pass with C100's k-pool tables on the cards, on a P106 rig, through the deep lane
# (bench/deep_lane). From a machine that reaches the rig over ssh and only runs ssh and tar, at this repository's root.
#   RIG          the rig's ssh alias (non-interactive)
#   WORK         a folder on the rig (default /tmp/deep-lane)
#   BIN          the bin/ folder of the build on the rig (pascal-p106-v4 d097d6d88, or any build with C748's code)
#   MODEL        the first shard of the Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, on the rig
#   MODEL_DRAFT  the MTP draft, on the rig
#   TEXT_LOCAL   the release check's recorded requests here (proxy.jsonl.bodies.gz)
#   OUT          the results folder here (collect)
# STEP=start|wait|collect; DRY=1 with start prints each arm's server line and starts nothing.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/deep-lane}
[ -n "${DRY:-}" ] && export DRY
KV="LLAMA_MTP_KV_ONLY=1"
DEV="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1"
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
    ARMS="sparse kv dev kvdev check" \
    ARM_ENV_kv="$KV" \
    ARM_ENV_dev="$DEV" \
    ARM_ENV_kvdev="$KV $DEV" \
    ARM_ENV_check="$KV $DEV LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
    LOADS="warm:sparse kv1:kv dev1:dev kvdev1:kvdev kvdev2:kvdev dev2:dev kv2:kv check:check" \
    DEPTHS="0 16384 49152" WARM_DEPTHS="0 16384 49152" PAUSE=4 \
    bash bench/deep_lane/remote.sh start
  ;;
wait) bash bench/deep_lane/remote.sh wait ;;
collect) CLEAN=${CLEAN:-0} OUT=${OUT:?} bash bench/deep_lane/remote.sh collect ;;
esac
