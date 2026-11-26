#!/usr/bin/env bash
# R245 (Q1925): Qwen3.8-Flash-Next's four-token switch point between 0.42 and 0.49, with the per-token sparse attention
# kernel in 5 splits, on the deep lane. The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO
# Q2_0, its n-gram table in a third shard, and the MTP draft, on; since C1175 it sets GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10
# and _MIN_RATIO=0.5, since C905 the taper's cuts=1) on nine P106-090 cards, one build (a pascal-p106-v5 tree).
# The kernel (ggml/src/ggml-cuda/fattn-sparse-pascal*.cu*): with GGML_CUDA_FATTN_SPARSE_PASCAL=1 a decode or verify call
# of n_q query tokens (up to 8) goes to the per-token kernel once the cache holds _MIN_RATIO x 2,051 (the cells a QSA
# layer keeps) x n_q cells. The cache view grows in steps of 256 cells (R240), so a switch point acts from the next
# multiple of 256. Arms, all with _NSM=10 (the preset's two values are dropped and each arm sets both):
#   c      _MIN_RATIO=0.5, the preset as it stands (control): four-token verifies on the sparse kernel from 4,102 cells,
#          a view of 4,352; one-token calls from 1,026 (a view of 1,280)
#   r42    _MIN_RATIO=0.42: four-token verifies from 3,446 cells (3,445.7), a view of 3,584; one-token calls from 862
#   r45    _MIN_RATIO=0.45: four-token verifies from 3,692 cells (3,691.8), a view of 3,840; one-token calls from 923
#   r49    _MIN_RATIO=0.49: four-token verifies from 4,020 cells (4,019.96), a view of 4,096; one-token calls from 1,005
#          (the three arms' one-token calls all switch at a view of 1,024, the control's at 1,280)
# A request at depth D generates 256 tokens, so its verify passes read D to D + 256 cells plus the view's padding: at
# depth 0 every arm runs the dense kernels for four-token verifies; at 3,328 (views of 3,584) only r42 runs the sparse
# one; at 3,584 (views of 3,840) r42 and r45; at 3,840 (views of 4,096) r42, r45 and r49; the control at none of them.
# This is R240's run/r240.sh (docs/data/2026-11-26_r240_flash_ratio_rig2/, C1175's and R175's deep-lane design) with
# the switch points as the arms and the depths around the estimated crossover (R240's README, "The server offset").
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as R69, R175, C1175 and R240), PRESET
# (default release/flash-next-v5/settings.env), OUT (here, for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of c, then two loads an arm with the arms' order
#            reversed in the second round (Q1253): c, r42, r45, r49, r49, r45, r42, c; depths 0, 3,328, 3,584 and
#            3,840; 256 tokens at temperature 0 twice per depth, then once with the preset's sampler (SAMPLER_PASS=1),
#            every reply kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control c, R175's run/steps.py and
# R69's run/replies.py (the commands in this folder's README).
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r245}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
NSM=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 3328 3584 3840" \
    ENV_DROP="$NSM $RATIO" ARMS="c r42 r45 r49" \
    ARM_ENV_c="$NSM=10 $RATIO=0.5" ARM_ENV_r42="$NSM=10 $RATIO=0.42" ARM_ENV_r45="$NSM=10 $RATIO=0.45" \
    ARM_ENV_r49="$NSM=10 $RATIO=0.49" \
    LOADS="warm:c ca:c r42a:r42 r45a:r45 r49a:r49 r49b:r49 r45b:r45 r42b:r42 cb:c"
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
