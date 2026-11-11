#!/usr/bin/env bash
# r76_run.sh: R76 (Q1278), C620's KV-only MTP catch-up (LLAMA_MTP_KV_ONLY=1) on a P106 rig, on the v4 night merge
# (it holds C620's change; no other switch it adds is set). From a machine that only runs ssh, at the repository's root;
# every step but `lane` and `prompt` waits in one ssh call or reads files here.
#   STEP=lane     (a) the identity, untimed: R53's line at depths 0 and 32,768, 16 tokens a request, every verify step
#                 dumped per load (LLAMA_SPEC_DUMP). The second control load is its own arm (draftb, the same line as
#                 draft) so that each load writes its own dump.
#   STEP=prompt   (b) R53's prompt table (its depths 0, 8,192, 16,384, 32,768 and 49,152) with the draft, the draft with
#                 the switch and no draft, two loads each after a warm-up load of the draft.
#   STEP=wait PART=lane|prompt              one ssh call that returns when that part has ended
#   STEP=collect PART=lane|prompt OUT=<folder>   that part's results to OUT, then its rig folder removed
# Env: RIG (the rig's ssh alias), WORK (a folder on the rig), BUILDS (the builds folder on the rig), BUILD (the build's
# folder name in it), MODELS (the rig folder with swift_q2_0/ and the MTP draft), RECORDED (the release check's recorded
# agent text here, proxy.jsonl.bodies.gz).
set -uo pipefail
: "${STEP:?}" "${RIG:?}" "${WORK:?}"
export RIG
common() {
  BIN=${BUILDS:?}/${BUILD:?}/bin/llama-server \
    MODEL=${MODELS:?}/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
    MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
    TEXT_LOCAL=${RECORDED:?}/proxy.jsonl.bodies.gz TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 \
    "$@" bash bench/deep_lane/remote.sh start
}
case $STEP in
lane)
  export WORK=$WORK/lane
  common env DEPTHS="0 32768" ARMS="draft kvonly draftb" \
    ARM_ENV_draft="LLAMA_SPEC_DUMP=$WORK/out/dump-draft.jsonl" \
    ARM_ENV_kvonly="LLAMA_MTP_KV_ONLY=1 LLAMA_SPEC_DUMP=$WORK/out/dump-kvonly.jsonl" \
    ARM_ENV_draftb="LLAMA_SPEC_DUMP=$WORK/out/dump-draftb.jsonl" \
    LOADS="draft1:draft kvonly1:kvonly draft2:draftb"
  ;;
prompt)
  export WORK=$WORK/prompt
  common env ARMS="draft kvonly nodraft" ARM_ENV_kvonly=LLAMA_MTP_KV_ONLY=1 ARM_DRAFT_nodraft=off \
    LOADS="warm:draft draft1:draft kvonly1:kvonly nodraft1:nodraft draft2:draft kvonly2:kvonly nodraft2:nodraft"
  ;;
wait) WORK=$WORK/${PART:?} bash bench/deep_lane/remote.sh wait ;;
collect) WORK=$WORK/${PART:?} OUT=${OUT:?} bash bench/deep_lane/remote.sh collect ;;
*) echo "STEP=lane|prompt|wait|collect"; exit 1 ;;
esac
