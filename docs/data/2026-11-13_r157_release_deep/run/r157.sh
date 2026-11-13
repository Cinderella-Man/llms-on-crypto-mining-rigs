#!/usr/bin/env bash
# R157 (Q1475): Flash-Next's release deep lane. Decode and prompt reading at depth for v4's final settings v4r on the
# release candidate v4-rc2 (pascal-p106-v4 eb82916d3), against the v3.0 release (pascal-p106-v3-b12076-ladder5
# 1551f5fe1), on one rig. This is R90's lane step (docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh: bench/deep_lane with
# the sampler pass, each arm on its own build) with the depths of R88 (docs/data/2026-11-13_r88_union_deep) and two
# arms:
#   v3    v3.0's build (ARM_BIN_v3) on the v3 release preset (release/flash-next-v3/settings.env) as shipped
#   v4r   v4-rc2's build (BIN) with v4r's switches and arguments as the agent check runs them
#         (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh: PRESET_ENV + GAINS_ENV + V4C2_ENV + V4R_ENV, and the line
#         MT_V4R): -b 4096, no -otd token_embd=CUDA4 (LLAMA_DRAFT_EMBD_DEV=layer places the draft's token table),
#         --spec-draft-n-ubatch 512, probabilistic draft verification, four checkpoints on the cards and inside a
#         prompt's call, and the micro-batch taper short=600:64
# The lane drops -otd from every line (ARGS_DROP is the same for all arms) and v3's ARM_ARGS puts the preset's
# -otd token_embd=CUDA4 back, so v3's line holds exactly the preset's flags.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4-rc2's llama-server on the rig), REF_BIN
# (v3.0's llama-server on the rig), MODEL (Swift Q2_0's first shard on the rig), MODEL_DRAFT (the MTP draft on the rig),
# TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect),
# DEPTHS (default 0 4096 8192 16384 32768 49152). STEP=start|wait|status|collect; DRY=1 prints both server lines and
# starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r157}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
# r156.sh's GAINS_ENV, V4C2_ENV and V4R_ENV, in that order
GAINS_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1"
V4C2_ENV="LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 LLAMA_SPEC_REJECT_STOP=1 LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1"
V4R_ENV="LLAMA_SCHED_PER_SIZE_RESERVE=2 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_MTP_HROW_LAZY=1 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0"
# MT_V4R's flags that differ from the preset's line (-otd dropped for every arm, below)
V4R_ARGS="-b 4096 --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64"
case ${STEP:-start} in
start)
  : "${BIN:?}" "${REF_BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${TEXT_LOCAL:?}"
  BIN=$BIN ARM_BIN_v3=$REF_BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT DRAFT=on TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 \
    DEPTHS="${DEPTHS:-0 4096 8192 16384 32768 49152}" N_PREDICT=256 \
    ARGS_DROP=-otd \
    ARMS="v3 v4r" \
    ARM_ARGS_v3="-otd token_embd=CUDA4" \
    ARM_ENV_v4r="$GAINS_ENV $V4C2_ENV $V4R_ENV" \
    ARM_ARGS_v4r="$V4R_ARGS" \
    LOADS="warm:v3 v31:v3 v4r1:v4r v32:v3 v4r2:v4r" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
