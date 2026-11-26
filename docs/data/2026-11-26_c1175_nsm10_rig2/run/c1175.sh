#!/usr/bin/env bash
# C1175 (Q1908): Qwen3.8-Flash-Next's four-token verify passes with the per-token sparse attention kernel in 5 splits
# instead of 3, using switches the engine already has, on the deep lane. The v5 preset
# (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table in a third shard, and the MTP
# draft, on) on nine P106-090 cards, one build (a pascal-p106-v5 tree).
# The kernel (ggml/src/ggml-cuda/fattn-sparse-pascal*.cu*): with GGML_CUDA_FATTN_SPARSE_PASCAL=1 a decode or verify call
# of n_q query tokens goes to the per-token kernel once the cache holds _MIN_RATIO x 2,051 (the cells a QSA layer
# keeps) x n_q cells; its splits per list are fasp_pick_splits: ceil(4 x _NSM / (2 KV heads x n_q)). At _NSM=5 (the
# preset) a four-token verify gets 3 splits, 24 blocks, where a P106's 5 SMs hold 20 at once (a second wave of 4);
# at _NSM=10 it gets 5 splits, 40 blocks, two full waves (C839: 590.4 against 968.3 us a call at 4,096 cells), and a
# one-token call 20 splits instead of 10 (+43.3 us a call). Arms (the preset's two values are dropped and each arm
# sets both):
#   c      _NSM=5  _MIN_RATIO=0.75, the preset (control): 3 splits at 4 tokens, the sparse kernel from 6,153 cells
#   n10    _NSM=10 _MIN_RATIO=0.75: 5 splits at 4 tokens, from 6,153 cells
#   r05    _NSM=10 _MIN_RATIO=0.5:  5 splits at 4 tokens, from 4,102 cells
# A request at depth D generates 256 tokens, so its verify passes read D to D + 256 cells plus the cache's padding: at
# depth 0 every arm runs the dense kernel; at 4,096 only r05 runs the sparse one; at 6,144 and 8,192 all three do, n10
# and r05 in 5 splits.
# This is R175's deep-lane design (docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh, from R69's r69.sh) with the
# split count and the switch point as the arms.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as R69 and R175), PRESET (default
# release/flash-next-v5/settings.env), OUT (here, for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of c, then two loads an arm with the arms' order
#            reversed in the second round (Q1253): c, n10, r05, r05, n10, c; depths 0, 4,096, 6,144 and 8,192; 256
#            tokens at temperature 0 twice per depth, then once with the preset's sampler (SAMPLER_PASS=1), every reply
#            kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control c, and this folder's
# run/tables.sh.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c1175}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
NSM=GGML_CUDA_FATTN_SPARSE_PASCAL_NSM
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 4096 6144 8192" \
    ENV_DROP="$NSM $RATIO" ARMS="c n10 r05" \
    ARM_ENV_c="$NSM=5 $RATIO=0.75" ARM_ENV_n10="$NSM=10 $RATIO=0.75" ARM_ENV_r05="$NSM=10 $RATIO=0.5" \
    LOADS="warm:c ca:c n10a:n10 r05a:r05 r05b:r05 n10b:n10 cb:c"
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
