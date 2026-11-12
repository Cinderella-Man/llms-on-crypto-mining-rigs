#!/usr/bin/env bash
# r123a_loads.sh: R123a (Q1401, first half), the prompt switches of pascal-p106-v4 as one ladder on an agent session's
# own prompts. One detached load of R43's replay (docs/data/2026-11-10_prompt_lane/run/r43_run.sh, LOADS=A, the 62
# recorded prompts, two passes) on the rig per call; each rung adds its switch to the rung before. Run from a machine
# that only runs ssh; the load runs on the rig, detached, and ends with $WORK/r123a_<RUNG>/DONE. Nothing is read from
# the rig after the start: env.txt and the server's switch line are checked once DONE exists.
#   RUNG=base     the line every rig row of 2026-11-12 starts from (docs/data/2026-11-12_r122_host_ladder/run/preset.env):
#                 R90's gains line (docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh: GAINS_ENV, GAINS_ARGS) without
#                 LLAMA_SAMPLING_TOP_K_ROWS=1 (FN84, docs/data/2026-11-12_fn84_graph_realloc/), --ubatch-taper
#                 short=600:64 in place of on (R60, docs/data/2026-11-12_r60_taper_v4/), the draft's micro-batch of
#                 512, plus C720's and C719's k-pool switches and R122's eight host switches (all gains,
#                 docs/data/2026-11-12_r122_host_ladder/)
#   RUNG=graphs   + GGML_CUDA_GRAPHS_FORCE=1     C702: prompt micro-batches captured as CUDA graphs (R82)
#   RUNG=kvonly   + LLAMA_MTP_KV_ONLY=1          C620 with C748's fix: the draft's catch-up writes only its KV (R91)
#   RUNG=overlap  + LLAMA_MTP_CATCHUP_OVERLAP=1  C703: the draft's catch-up runs beside the target's next call (R84 a)
#   RUNG=ngram    + LLAMA_NGRAM_PREFETCH=1       C761: the n-gram rows of a call's later micro-batches read ahead
#   RUNG=base2    base again, for the reference's spread
# Every request carries the line's taper (PASSES_A='short=600:64 short=600:64'); FRESH=0, since the replay's fresh
# prompts run with the taper off and on, not the line's.
# Env: RIG (ssh alias), WORK (the folder on the rig with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv and the
#      v3 preset's settings.env), BUILDS, BUILD (the rig tarball's folder), MODEL (Swift Q2_0's first shard on the
#      rig), DRAFT (the MTP draft on the rig), RUNG, DRY=1 prints the rig command only.
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${RUNG:?}"
# R90's gains line without LLAMA_SAMPLING_TOP_K_ROWS=1, C720's and C719's k-pool switches (R122's preset.env)
LINE_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1"
# R122's eight host switches (rungs a to f)
HOST_ENV="LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1"
BASE_ENV="$LINE_ENV $HOST_ENV"
# R90's GAINS_ARGS with R60's taper; r43_run.sh's line holds --spec-draft-n-ubatch 128, and the 512 after it is the
# value the server keeps (the parser takes the last of a repeated flag and logs a warning)
TAPER=short=600:64
ARGS_ADD="--spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper $TAPER"
G=GGML_CUDA_GRAPHS_FORCE=1
K="$G LLAMA_MTP_KV_ONLY=1"
O="$K LLAMA_MTP_CATCHUP_OVERLAP=1"
N="$O LLAMA_NGRAM_PREFETCH=1"
case $RUNG in
  base|base2) ENV="$BASE_ENV" ;;
  graphs) ENV="$BASE_ENV $G" ;;
  kvonly) ENV="$BASE_ENV $K" ;;
  overlap) ENV="$BASE_ENV $O" ;;
  ngram) ENV="$BASE_ENV $N" ;;
  *) echo "RUNG is base, graphs, kvonly, overlap, ngram or base2"; exit 1 ;;
esac
OUTN=r123a_$RUNG
cmd="cd $WORK && export $ENV && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A='$TAPER $TAPER' FRESH=0 setsid nohup bash $WORK/r43_run.sh > $WORK/$OUTN.out 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
echo "started $OUTN at $(date -Is); wait for $WORK/$OUTN/DONE"
