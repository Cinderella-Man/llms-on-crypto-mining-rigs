#!/usr/bin/env bash
# r60_loads.sh: R60 (Q1235), the micro-batch taper's settings on v4's whole line, one detached load of R43's replay
# (docs/data/2026-11-10_prompt_lane/run/r43_run.sh) on the rig. Run from a machine that only runs ssh; the load runs
# on the rig, detached, and ends with $WORK/<load>/DONE.
#   LOAD=a    R90's gains line (docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh: GAINS_ENV exported, GAINS_ARGS and the
#             draft's micro-batch of 512 appended to R43's line), the taper set per request in five passes:
#             off, on, tail=512:128,short=600:64, short=600:64, off; then four fresh 2,049-token prompts off and on
#   LOAD=ro8  the same line with GGML_CUDA_MMVQ_PASCAL_RO8=1 also exported, passes on, off (timing only: RO8 had no
#             native check on a P106 yet, so its replies are compared with load a's)
# Env: RIG (ssh alias), WORK (the folder on the rig with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv and the
#      v3 preset's settings.env), BUILDS, BUILD (the rig tarball's folder), MODEL (Swift Q2_0's first shard on the
#      rig), DRAFT (the MTP draft on the rig), LOAD (a or ro8), DRY=1 prints the rig command only.
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${LOAD:?}"
# R90's gains line, as r90.sh has it
GAINS_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1"
GAINS_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on"
# r43_run.sh's line holds --spec-draft-n-ubatch 128; the 512 after it is the value the server keeps (the parser takes
# the last of a repeated flag and logs a warning)
ARGS_ADD="--spec-draft-n-ubatch 512 $GAINS_ARGS"
case $LOAD in
  a) ENV="$GAINS_ENV"; PASSES="off on tail=512:128,short=600:64 short=600:64 off"; OUTN=r60a ;;
  ro8) ENV="$GAINS_ENV GGML_CUDA_MMVQ_PASCAL_RO8=1"; PASSES="on off"; OUTN=r60ro8 ;;
  *) echo "LOAD is a or ro8"; exit 1 ;;
esac
cmd="cd $WORK && export $ENV && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A='$PASSES' setsid nohup bash $WORK/r43_run.sh > $WORK/$OUTN.out 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
sleep 5
ssh "$RIG" "cat $WORK/$OUTN/env.txt; tail -n 3 $WORK/$OUTN/run.log 2> /dev/null"
