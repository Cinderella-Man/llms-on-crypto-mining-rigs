#!/usr/bin/env bash
# R240 (Q1910): Qwen3.8-Flash-Next's four-token switch point below 0.5, with the per-token sparse attention kernel in 5
# splits, on the deep lane. The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its
# n-gram table in a third shard, and the MTP draft, on; since C1175 it sets GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 and
# _MIN_RATIO=0.5) on nine P106-090 cards, one build (a pascal-p106-v5 tree).
# The kernel (ggml/src/ggml-cuda/fattn-sparse-pascal*.cu*): with GGML_CUDA_FATTN_SPARSE_PASCAL=1 a decode or verify call
# of n_q query tokens (up to 8) goes to the per-token kernel once the cache holds _MIN_RATIO x 2,051 (the cells a QSA
# layer keeps) x n_q cells; its splits per list are ceil(4 x _NSM / (2 KV heads x n_q)), so _NSM=10 gives a four-token
# verify 5 splits (two full waves on a P106's 5 SMs) and a one-token call 20. Under 2,051 cells the mask keeps every
# cell, so the per-token kernel reads them all. Arms, all with _NSM=10 (the preset's two values are dropped and each
# arm sets both):
#   c      _MIN_RATIO=0.5, the preset as it stands (control): four-token verifies on the sparse kernel from 4,102 cells,
#          one-token calls from 1,026
#   r375   _MIN_RATIO=0.375: four-token verifies from 3,077 cells (3,076.5), one-token calls from 770 (769.1)
#   r25    _MIN_RATIO=0.25: four-token verifies from 2,051 cells (2,051.0), one-token calls from 513 (512.8)
# A request at depth D generates 256 tokens, so its verify passes read D to D + 256 cells plus the cache's padding: at
# depth 0 every arm runs the dense kernels for four-token verifies; at 2,048 only r25 runs the sparse one; at 3,072
# r375 and r25; at 4,096 all three.
# This is C1175's run/c1175.sh (docs/data/2026-11-26_c1175_nsm10_rig2/, R175's deep-lane design) with the switch point
# as the arm and lower depths.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as R69, R175 and C1175), PRESET (default
# release/flash-next-v5/settings.env), OUT (here, for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of c, then two loads an arm with the arms' order
#            reversed in the second round (Q1253): c, r375, r25, r25, r375, c; depths 0, 2,048, 3,072 and 4,096; 256
#            tokens at temperature 0 twice per depth, then once with the preset's sampler (SAMPLER_PASS=1), every reply
#            kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control c, R175's run/steps.py and
# R69's run/replies.py (the commands in this folder's README).
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r240}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
NSM=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 2048 3072 4096" \
    ENV_DROP="$NSM $RATIO" ARMS="c r375 r25" \
    ARM_ENV_c="$NSM=10 $RATIO=0.5" ARM_ENV_r375="$NSM=10 $RATIO=0.375" ARM_ENV_r25="$NSM=10 $RATIO=0.25" \
    LOADS="warm:c ca:c r375a:r375 r25a:r25 r25b:r25 r375b:r375 cb:c"
}
case ${STEP:-start} in
dry) common; DRY=1 bash $L start ;;
start) common; bash $L start ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
*) echo "STEP=dry|start|wait|status|collect|stop"; exit 1 ;;
esac
