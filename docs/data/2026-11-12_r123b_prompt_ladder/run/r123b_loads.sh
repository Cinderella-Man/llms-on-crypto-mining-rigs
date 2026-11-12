#!/usr/bin/env bash
# r123b_loads.sh: R123b (Q1401, second half), four more prompt-side settings of pascal-p106-v4 as one ladder on an
# agent session's own prompts, on top of R123a's kept line. R123a's launcher (../../2026-11-12_r123a_prompt_ladder/run/
# r123a_loads.sh) with this ladder's rungs: one detached load of R43's replay (docs/data/2026-11-10_prompt_lane/run/
# r43_run.sh, LOADS=A, the 62 recorded prompts, two passes) on the rig per call; each rung adds its setting to the rung
# before. Run from a machine that only runs ssh; the load runs on the rig, detached, and ends with
# $WORK/r123b_<RUNG>/DONE. Nothing is read from the rig after the start.
#   RUNG=base     R123a's kept line: the 2026-11-12 line (R90's gains line without LLAMA_SAMPLING_TOP_K_ROWS=1,
#                 --ubatch-taper short=600:64, the draft's micro-batch of 512, C720's and C719's k-pool switches), R122's
#                 eight host switches, and R123a's GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1
#                 LLAMA_MTP_CATCHUP_OVERLAP=1 (docs/data/2026-11-12_r123a_prompt_ladder/), without LLAMA_NGRAM_PREFETCH=1
#   RUNG=embd     + LLAMA_DRAFT_EMBD_DEV=layer   C762: the MTP draft's token table on its layer's card (card 8), in place
#                 of the line's -otd token_embd=CUDA4 (R104). r43_run.sh holds that flag in its line, so from this rung
#                 on the load runs r43_run_noemb.sh, made on the rig from r43_run.sh with only that flag removed
#   RUNG=sparse   + GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1   C705: the sparse attention rule for prompt micro-batches
#                 (R73 (1): exact on a P106, docs/data/2026-11-12_r73_qsa_prompt_replay/)
#   RUNG=queues   + CUDA_SCALE_LAUNCH_QUEUES=2x  CUDA's launch queue at twice its size (R62)
#   RUNG=b4096    + -b 4096                      a logical batch of 4,096 (R63); appended to the line, the last -b wins
#   RUNG=base2    base again, for the reference's spread
# Every request carries the line's taper (PASSES_A='short=600:64 short=600:64'); FRESH=0, as R123a.
# Env: RIG (ssh alias), WORK (the folder on the rig with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv and the
#      v3 preset's settings.env), BUILDS, BUILD (the rig tarball's folder), MODEL (Swift Q2_0's first shard on the
#      rig), DRAFT (the MTP draft on the rig), RUNG, DRY=1 prints the rig command only.
# Writes $WORK/r123b_<RUNG>.launch on the rig at the start: the run script's md5, its line, and every LLAMA_, GGML_ and
# CUDA_ variable the load starts with (r43_run.sh's env.txt lists only LLAMA_ and GGML_).
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${RUNG:?}"
# R90's gains line without LLAMA_SAMPLING_TOP_K_ROWS=1, C720's and C719's k-pool switches (R122's preset.env)
LINE_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1"
# R122's eight host switches (rungs a to f)
HOST_ENV="LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1"
# R123a's three kept rungs
KEPT_ENV="GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1"
BASE_ENV="$LINE_ENV $HOST_ENV $KEPT_ENV"
# R90's GAINS_ARGS with R60's taper; r43_run.sh's line holds --spec-draft-n-ubatch 128, and the 512 after it is the
# value the server keeps (the parser takes the last of a repeated flag)
TAPER=short=600:64
ARGS_ADD="--spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper $TAPER"
E=LLAMA_DRAFT_EMBD_DEV=layer
S="$E GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1"
Q="$S CUDA_SCALE_LAUNCH_QUEUES=2x"
RUN=r43_run.sh
case $RUNG in
  base|base2) ENV="$BASE_ENV" ;;
  embd) ENV="$BASE_ENV $E"; RUN=r43_run_noemb.sh ;;
  sparse) ENV="$BASE_ENV $S"; RUN=r43_run_noemb.sh ;;
  queues) ENV="$BASE_ENV $Q"; RUN=r43_run_noemb.sh ;;
  b4096) ENV="$BASE_ENV $Q"; RUN=r43_run_noemb.sh; ARGS_ADD="$ARGS_ADD -b 4096" ;;
  *) echo "RUNG is base, embd, sparse, queues, b4096 or base2"; exit 1 ;;
esac
OUTN=r123b_$RUNG
# r43_run_noemb.sh: r43_run.sh with " -otd token_embd=CUDA4" removed from its line, nothing else; the start stops if
# the flag is still there
MK="sed 's/ -otd token_embd=CUDA4//' r43_run.sh > r43_run_noemb.sh && ! grep -q 'token_embd=CUDA4' r43_run_noemb.sh"
cmd="cd $WORK && $MK && export $ENV && { md5sum $RUN; grep -m1 '^export ARGS=' $RUN; env | grep -E '^(LLAMA_|GGML_|CUDA_)' | sort; } > $WORK/$OUTN.launch && \
BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A='$TAPER $TAPER' FRESH=0 setsid nohup bash $WORK/$RUN > $WORK/$OUTN.out 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
echo "started $OUTN at $(date -Is); wait for $WORK/$OUTN/DONE"
