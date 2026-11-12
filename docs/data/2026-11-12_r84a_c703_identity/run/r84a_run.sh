#!/usr/bin/env bash
# R84 (a) (Q1304): C703's kept catch-up (LLAMA_MTP_CATCHUP_OVERLAP=1) against the draft's own catch-up on a P106 rig,
# untimed: every verify step's drafted tokens and the draft's probabilities written by LLAMA_SPEC_DUMP, through the
# deep lane (bench/deep_lane), loads draft1, overlap1, draft2 at depths 0 and 32,768. From a machine that reaches the
# rig over ssh and only runs ssh and tar, at this repository's root.
#   RIG          the rig's ssh alias (non-interactive)
#   WORK         a folder on the rig (default /tmp/deep-lane; the dumps are written there)
#   BIN          the bin/ folder of a build with C703's code on the rig (pascal-p106-v4 d097d6d88 or later)
#   MODEL        the first shard of the Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, on the rig
#   MODEL_DRAFT  the MTP draft, on the rig
#   TEXT_LOCAL   the release check's recorded requests here (proxy.jsonl.bodies.gz)
#   OUT          the results folder here (collect)
# STEP=start|wait|collect; DRY=1 with start prints each arm's server line and starts nothing.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/deep-lane}
[ -n "${DRY:-}" ] && export DRY
case ${STEP:-start} in
start)
  : "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 \
    DEPTHS="0 32768" ARMS="draft overlap" \
    ARM_ENV_draft="LLAMA_SPEC_DUMP=$WORK/dump-draft.jsonl" \
    ARM_ENV_overlap="LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_SPEC_DUMP=$WORK/dump-overlap.jsonl" \
    LOADS="draft1:draft overlap1:overlap draft2:draft" \
    bash bench/deep_lane/remote.sh start
  ;;
wait) bash bench/deep_lane/remote.sh wait ;;
collect)  # the deep lane's folder, then the two dumps beside it (remote.sh collect copies WORK/out only)
  : "${OUT:?}"
  CLEAN=0 OUT=$OUT bash bench/deep_lane/remote.sh collect || exit 1
  ssh "$RIG" "tar -C $WORK -cf - dump-draft.jsonl dump-overlap.jsonl" | tar -C "$OUT" -xf - || { echo "dump copy failed"; exit 1; }
  ls -l "$OUT"/dump-*.jsonl
  ;;
esac
