#!/usr/bin/env bash
# r139_load.sh: R139 (Q1432), C813's LLAMA_MTP_HROW_LAZY=1 on a P106. R123b's launcher
# (../../2026-11-12_r123b_prompt_ladder/run/r123b_loads.sh) with its `b4096` rung's line and the switch: one detached
# load of R43's replay (docs/data/2026-11-10_prompt_lane/run/r43_run.sh without its -otd token_embd=CUDA4, LOADS=A, the 62
# recorded prompts, two passes, the taper on every request, FRESH=0) on the rig per call, with one card's memory sampled
# every 2 s beside it. Run from a machine that only runs ssh; the load runs on the rig, detached, and ends with
# $WORK/r139_<RUNG>/DONE. Nothing is read from the rig after the start.
#   RUNG=lazy    R123b's b4096 rung (R123a's kept line + LLAMA_DRAFT_EMBD_DEV=layer, GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1,
#                CUDA_SCALE_LAUNCH_QUEUES=2x, -b 4096) + LLAMA_MTP_HROW_LAZY=1
#   RUNG=b4096   R123b's b4096 rung on this build, the switch unset (only if a same-build reference is wanted)
# Env: RIG (ssh alias), WORK (the folder on the rig with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv and the
#      v3 preset's settings.env), BUILDS, BUILD (the rig tarball's folder), MODEL (Swift Q2_0's first shard on the
#      rig), DRAFT (the MTP draft on the rig), RUNG, CARD (the card sampled, nvidia-smi's index: 8, the output card),
#      DRY=1 prints the rig command only.
# Writes on the rig, at the start, $WORK/r139_<RUNG>.launch (the run script's md5, its line, and every LLAMA_, GGML_ and
# CUDA_ variable the load starts with) and, until the load's DONE, $WORK/r139_<RUNG>.card.csv (nvidia-smi's timestamp
# and the card's used MiB every 2 s).
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${RUNG:?}"
CARD=${CARD:-8}
# R123b's base line: R90's gains line without LLAMA_SAMPLING_TOP_K_ROWS=1, C720's and C719's k-pool switches, R122's
# eight host switches and R123a's three kept rungs
LINE_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1"
HOST_ENV="LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1"
KEPT_ENV="GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1"
BASE_ENV="$LINE_ENV $HOST_ENV $KEPT_ENV"
TAPER=short=600:64
ARGS_ADD="--spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper $TAPER -b 4096"
# R123b's b4096 rung: its sparse and queues rungs' settings on top of the embd rung
Q="LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 CUDA_SCALE_LAUNCH_QUEUES=2x"
RUN=r43_run_noemb.sh
case $RUNG in
  lazy) ENV="$BASE_ENV $Q LLAMA_MTP_HROW_LAZY=1" ;;
  b4096) ENV="$BASE_ENV $Q" ;;
  *) echo "RUNG is lazy or b4096"; exit 1 ;;
esac
OUTN=r139_$RUNG
# r43_run_noemb.sh: r43_run.sh with " -otd token_embd=CUDA4" removed from its line, nothing else (as R123b); the start
# stops if the flag is still there
MK="sed 's/ -otd token_embd=CUDA4//' r43_run.sh > r43_run_noemb.sh && ! grep -q 'token_embd=CUDA4' r43_run_noemb.sh"
# the card's memory every 2 s until the load's DONE (one nvidia-smi process for the whole load); it starts once the
# load's own list has removed an earlier folder of this name
SAMPLE="sleep 15; nvidia-smi -i $CARD --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 2 > $WORK/$OUTN.card.csv 2>&1 & p=\$!; \
while [ ! -e $WORK/$OUTN/DONE ]; do sleep 10; done; kill \$p"
cmd="cd $WORK && $MK && rm -rf $WORK/$OUTN && export $ENV && { md5sum $RUN; grep -m1 '^export ARGS=' $RUN; env | grep -E '^(LLAMA_|GGML_|CUDA_)' | sort; } > $WORK/$OUTN.launch && \
BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A='$TAPER $TAPER' FRESH=0 setsid nohup bash $WORK/$RUN > $WORK/$OUTN.out 2>&1 < /dev/null & \
setsid nohup bash -c '$SAMPLE' > /dev/null 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
echo "started $OUTN at $(date -Is); wait for $WORK/$OUTN/DONE"
