#!/usr/bin/env bash
# fn84_loads.sh: FN84 (Q1405), what reallocates the target's prompt graph on v4's `gains` line. One detached load of
# R43's replay (docs/data/2026-11-10_prompt_lane/run/r43_run.sh) on the rig, counts only: R60's line
# (docs/data/2026-11-12_r60_taper_v4/run/r60_loads.sh, load a) with one task of the replay, taper off, plus one fresh
# 2,049-token prompt (run with the taper off, then on), the server at -lv 5 (rig_taper.sh's default). DROP removes
# switches or flags from the line, so that the load whose count of `graph reallocs` falls to 0 names the cause
# (count them with docs/data/2026-11-12_r60_taper_v4/run/realloc_counts.py). Run from a machine that only runs ssh;
# the load runs on the rig, detached, and ends with $WORK/fn84_<NAME>/DONE.
# Env: RIG (ssh alias), WORK (the folder on the rig with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv and the
#      v3 preset's settings.env), BUILDS, BUILD (the rig tarball's folder), MODEL (Swift Q2_0's first shard on the
#      rig), DRAFT (the MTP draft on the rig), NAME (the load's name: output in $WORK/fn84_<NAME>),
#      DROP (space-separated, empty for the whole line): a switch's name (e.g. LLAMA_SAMPLING_TOP_K_ROWS), `switches`
#      (every switch of the line's environment), `draft-ubatch-512` (the draft's micro-batch of 512, leaving
#      r43_run.sh's 128), `probabilistic` (--spec-draft-sampling probabilistic --spec-draft-p-min 0), `incall`
#      (--ctx-checkpoints-incall), `taper` (--ubatch-taper on as the server's default), `all` (all of these; the line
#      keeps --ctx-checkpoints-vram 4), TASK (bugfix-javascript-complex-numbers: the replay's first task, so its text
#      is R60's pass 0's), DRY=1 prints the rig command only.
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${NAME:?}"
DROP=" ${DROP:-} "
TASK=${TASK:-bugfix-javascript-complex-numbers}
case $DROP in *" all "*) DROP="$DROP switches draft-ubatch-512 probabilistic incall taper " ;; esac
# R90's gains line, as r60_loads.sh has it (docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh)
GAINS_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1"
ENV=""
for kv in $GAINS_ENV; do
  case $DROP in *" ${kv%%=*} "*|*" switches "*) continue ;; esac
  ENV="${ENV:+$ENV }$kv"
done
# r43_run.sh's line holds --spec-draft-n-ubatch 128; a 512 after it is the value the server keeps
ARGS_ADD=""
case $DROP in *" draft-ubatch-512 "*) ;; *) ARGS_ADD="--spec-draft-n-ubatch 512" ;; esac
case $DROP in *" probabilistic "*) ;; *) ARGS_ADD="$ARGS_ADD --spec-draft-sampling probabilistic --spec-draft-p-min 0" ;; esac
ARGS_ADD="$ARGS_ADD --ctx-checkpoints-vram 4"
case $DROP in *" incall "*) ;; *) ARGS_ADD="$ARGS_ADD --ctx-checkpoints-incall" ;; esac
case $DROP in *" taper "*) ;; *) ARGS_ADD="$ARGS_ADD --ubatch-taper on" ;; esac
ARGS_ADD=${ARGS_ADD# }
OUTN=fn84_$NAME
cmd="cd $WORK && ${ENV:+export $ENV && }BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/$OUTN LOADS=A \
ARGS_ADD='$ARGS_ADD' PASSES_A=off TASKS=$TASK FRESH=1 FRESH_N=2048 setsid nohup bash $WORK/r43_run.sh \
> $WORK/$OUTN.out 2>&1 < /dev/null &"
echo "$cmd"
[ "${DRY:-0}" = 1 ] && exit 0
# an ssh call that starts a detached job can wait for it: the timeout ends the call, not the job
timeout 20 ssh "$RIG" "$cmd" || true
sleep 5
ssh "$RIG" "cat $WORK/$OUTN/env.txt; tail -n 3 $WORK/$OUTN/run.log 2> /dev/null"
