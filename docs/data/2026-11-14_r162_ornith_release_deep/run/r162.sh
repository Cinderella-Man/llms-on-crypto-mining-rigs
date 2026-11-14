#!/usr/bin/env bash
# R162 (Q1481): Ornith's release deep lane. Decode and prompt reading at depth for Ornith-1.5-122B Pascal N2P GGUF
# (valid1) on nine P106-090 cards, three arms in one run, each on its own build:
#   v3    v3.0's build (REF_BIN) on release/ornith-v3/settings.env as shipped (the keep-alive)
#   v4    v4-rc2's build (BIN) on release/ornith-v4/settings.env: v3.0's line plus the three v4 switches
#   v4s   the candidate stack: v4 plus every candidate of release/ornith-v4/README.md that fits memory, here all but
#         LLAMA_SCHED_PER_SIZE_RESERVE=2: the staged hop, forced prompt graphs, four checkpoints on the cards and
#         inside a prompt's call, the micro-batch taper and -b 4096 in place of -b 2048
# This is R157's lane (docs/data/2026-11-13_r157_release_deep/run/r157.sh: bench/deep_lane with the sampler pass)
# with the depths 0 to 48,640 (the deepest whose 256 written tokens fit the 49,152-token context).
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the lane runs detached on the
# rig, and the tables are built on the rig too. Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN
# (v4-rc2's llama-server on the rig), REF_BIN (v3.0's llama-server on the rig), MODEL (the Ornith file on the rig),
# TEXT_LOCAL (the recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect).
# STEP=fit|start|wait|status|tables|collect; DRY=1 prints the server lines and starts none.
#   fit      one load of v4s alone, one prompt at 48,640 tokens and 256 tokens written, with run/fitmon.sh logging
#            each card's memory and the host's MemAvailable every second (WORK-fit and WORK-mon on the rig); the
#            stack fits when no card runs out of memory and MemAvailable stays at or above 400 MB. Then
#            STEP=wait WORK=<WORK>-fit, kill fitmon.sh and its nvidia-smi on the rig, and STEP=collect WORK=<WORK>-fit
#   tables   after the run has ended and before collect: deep_table.py and prompt_depth.py on the rig, control v3 and
#            control v4, into WORK/out/tables
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r162}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
V4_ENV="GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1"
STACK_ENV="GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1"
STACK_ARGS="--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 -b 4096"
common() {
  export PRESET=release/ornith-v3/settings.env BIN=${BIN:?} MODEL=${MODEL:?} TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 ARM_ENV_v4="$V4_ENV" ARM_ENV_v4s="$V4_ENV $STACK_ENV" \
    ARM_ARGS_v4s="$STACK_ARGS"
}
case ${STEP:-start} in
fit)
  common
  ssh "$RIG" "mkdir -p $WORK-mon && cat > $WORK-mon/fitmon.sh" < "$(dirname "$0")/fitmon.sh"
  timeout 20 ssh "$RIG" "cd $WORK-mon && D=\$PWD setsid -f nohup bash fitmon.sh > fitmon.out 2>&1 < /dev/null"
  WORK=$WORK-fit ARMS=v4s LOADS=fit:v4s WARM=0 DEPTHS=48640 bash $L start
  ;;
start)
  common
  ARM_BIN_v3=${REF_BIN:?} DEPTHS="0 4096 8192 16384 32768 48640" ARMS="v3 v4 v4s" \
    LOADS="warm:v3 v31:v3 v41:v4 s1:v4s v32:v3 v42:v4 s2:v4s" bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
tables)
  ssh "$RIG" "cat > $WORK/deep_lane/prompt_depth.py" < docs/data/2026-11-10_deep_lane/run/prompt_depth.py
  ssh "$RIG" "cd $WORK && mkdir -p out/tables && T=out/tables &&
    for c in v3 v4; do
      python3 -I deep_lane/deep_table.py out --control \$c --out \$T/table_\$c.md --tsv \$T/classes_\$c.tsv > /dev/null &&
      python3 -I deep_lane/deep_table.py out --control \$c --class-on pass2 --out \$T/table_\${c}_pass2.md \
        --tsv \$T/classes_\${c}_pass2.tsv > /dev/null &&
      python3 -I deep_lane/prompt_depth.py out --control \$c --out \$T/prompt_depth_\$c.md --tsv \$T/prompt_depth_\$c > /dev/null ||
      exit 1
    done && ls \$T"
  ;;
collect) OUT=${OUT:?} bash $L collect ;;
esac
