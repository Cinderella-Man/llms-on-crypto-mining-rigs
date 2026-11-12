#!/usr/bin/env bash
# r130_loads.sh: R130 (Q1407, C803), the MTP draft's re-plans on a P106 with LLAMA_SCHED_PER_SIZE_RESERVE off and on.
# FN84's load (docs/data/2026-11-12_fn84_graph_realloc/run/fn84_loads.sh) unchanged but for ADD_ENV, switches added to
# R90's gains line (by default LLAMA_UBATCH_TRACE=1, so the scheduler logs every re-plan with its drain and plan times,
# those after a change of cards too, which -lv 5's count leaves out). One detached load of R43's replay on the rig:
# one task, taper off, plus one fresh 2,049-token prompt (taper off, then on), the server at -lv 5. Run from a machine
# that only runs ssh; the load ends with $WORK/fn84_<NAME>/DONE. Count with run/draft_trace.py parse (the draft's
# re-plans per request and phase, their drains and plans) and FN84's run/fn84_requests.py (replies, checkpoints).
# Env: as fn84_loads.sh (RIG, WORK, BUILDS, BUILD, MODEL, DRAFT, NAME, DROP, TASK, DRY), and ADD_ENV (space-separated
#      VAR=value added to the line's environment; default LLAMA_UBATCH_TRACE=1), e.g.
#      ADD_ENV='LLAMA_UBATCH_TRACE=1 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_SCHED_PER_SIZE_RESERVE=1'
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${NAME:?}"
DROP=" ${DROP:-} "
TASK=${TASK:-bugfix-javascript-complex-numbers}
case $DROP in *" all "*) DROP="$DROP switches draft-ubatch-512 probabilistic incall taper " ;; esac
# R90's gains line, as r60_loads.sh has it (docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh)
GAINS_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1"
ENV=""
for kv in $GAINS_ENV ${ADD_ENV:-LLAMA_UBATCH_TRACE=1}; do
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
