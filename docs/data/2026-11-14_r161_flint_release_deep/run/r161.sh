#!/usr/bin/env bash
# R161 (Q1480): Flint's release deep lane. Decode and prompt reading at depth on Flint Pascal Q8 GGUF (p065) for the
# v4 preset on the release candidate v4-rc2 (pascal-p106-v4 eb82916d3) against the v3.0 release
# (pascal-p106-v3-b12076-ladder5 1551f5fe1), plus a third arm with every untried candidate of release/flint-v4 stacked,
# on one rig. R157's lane (docs/data/2026-11-13_r157_release_deep/run/r157.sh: bench/deep_lane with the sampler pass,
# each arm on its own build), with three arms:
#   v3    v3.0's build (ARM_BIN_v3) on release/flint-v3/settings.env as shipped (no switch)
#   v4    v4-rc2's build (BIN) on release/flint-v4/settings.env: v3.0's line and its three switches
#   v4s   v4 plus the candidate stack: STACK_ENV and STACK_ARGS (below), every candidate of release/flint-v4/README.md
#         ("Candidates") but LLAMA_SCHED_PER_SIZE_RESERVE=2, which acts only when the scheduler re-plans
# v3's and v4's lines are the release files' own: v4's switches are read from release/flint-v4/settings.env, and the
# script stops if that file's PRESET_ARGS is not v3.0's line.
# FIT=1 runs the stack's fit load instead: one load of v4s in its own run folder, no warm-up, one prompt at the deepest
# depth (49,152 tokens) and its 256 tokens, so that the forced graphs and the checkpoints take their memory; beside the
# lane's own monitors it logs every card's memory every 2 s (fitmon_gpu.csv) and the host's MemAvailable and CmaFree
# every second (fitmon_mem.tsv) into the run's results folder.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (v4-rc2's llama-server on the rig), REF_BIN
# (v3.0's llama-server on the rig), MODEL (Flint Pascal Q8 GGUF on the rig), TEXT_LOCAL (the recorded agent text here:
# the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), DEPTHS (default 0 4096 8192 16384 32768 49152),
# STACK_ENV and STACK_ARGS (the stack, default every candidate; a candidate dropped after the fit load is left out
# here). STEP=start|wait|status|collect; DRY=1 prints the server lines and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
V3_ARGS=$(. release/flint-v3/settings.env && echo "$PRESET_ARGS")
V4_ARGS=$(. release/flint-v4/settings.env && echo "$PRESET_ARGS")
V4_ENV=$(. release/flint-v4/settings.env && echo "$PRESET_ENV")
[ "$V3_ARGS" = "$V4_ARGS" ] || { echo "release/flint-v4's line is not release/flint-v3's"; exit 1; }
STACK_ENV=${STACK_ENV-GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1}
STACK_ARGS=${STACK_ARGS---ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64}
if [ "${FIT:-0}" = 1 ]; then
  export WORK=${WORK:-/tmp/r161fit}
  ARMS="v4s" LOADS="fit:v4s" WARM=0 DEPTHS_RUN=49152
else
  export WORK=${WORK:-/tmp/r161}
  ARMS="v3 v4 v4s" LOADS="warm:v3 v31:v3 v41:v4 s1:v4s v32:v3 v42:v4 s2:v4s" WARM=1
  DEPTHS_RUN=${DEPTHS:-0 4096 8192 16384 32768 49152}
fi
# The fit load's monitor, on the rig: every card's memory every 2 s, the host's memory every second, until the run ends
FITMON='o=$WORK/out; mkdir -p $o
for _ in $(seq 60); do [ -s $o/driver.pid ] && break; sleep 1; done
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 2000 > $o/fitmon_gpu.csv 2>&1 &
n=$!
echo "epoch avail_mib cma_free_mib" > $o/fitmon_mem.tsv
until [ -e $o/done ] || [ -e $o/failed ] || ! kill -0 $(cat $o/driver.pid 2> /dev/null) 2> /dev/null; do
  while read -r k v _; do case $k in MemAvailable:) a=$v ;; CmaFree:) c=$v ;; esac; done < /proc/meminfo
  printf "%(%s)T %d %d\n" -1 $((a / 1024)) $((${c:-0} / 1024)) >> $o/fitmon_mem.tsv
  sleep 1
done
kill $n'
case ${STEP:-start} in
start)
  : "${BIN:?}" "${REF_BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  PRESET=$PWD/release/flint-v3/settings.env BIN=$BIN ARM_BIN_v3=$REF_BIN MODEL=$MODEL TEXT_LOCAL=$TEXT_LOCAL \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 DEPTHS="$DEPTHS_RUN" WARM=$WARM \
    ARMS="$ARMS" \
    ARM_ENV_v4="$V4_ENV" \
    ARM_ENV_v4s="$V4_ENV${STACK_ENV:+ $STACK_ENV}" \
    ARM_ARGS_v4s="$STACK_ARGS" \
    LOADS="$LOADS" \
    bash $L start || exit 1
  if [ "${FIT:-0}" = 1 ] && [ -z "${DRY:-}" ]; then
    timeout 30 ssh "$RIG" "WORK=$WORK setsid -f nohup bash -c '$FITMON' > /dev/null 2>&1 < /dev/null"
    echo "fit monitor started"
  fi
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
