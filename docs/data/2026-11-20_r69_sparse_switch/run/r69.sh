#!/usr/bin/env bash
# R69 (Q1252): where Qwen3.8-Flash-Next's sparse decode attention should take over from the dense kernel for a
# four-token verify pass, on the deep lane. The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5
# GSQ-RCO Q2_0, its n-gram table in a third shard, and the MTP draft, on) on nine P106-090 cards, one build (a
# pascal-p106-v5 tree). Mode 1 of the sparse kernel (GGML_CUDA_FATTN_SPARSE_PASCAL=1) leaves a call to the dense kernel
# while the cache holds fewer than GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO x the cells a QSA layer keeps (2,051) x the
# call's query tokens. This is R205's deep-lane design (docs/data/2026-11-19_r205_flash_glu/run/r205.sh) with the
# ratio as the arm. Arms (the preset's own ratio is dropped and each arm sets its one value):
#   c10    1.0, the preset's value (control): a four-token verify pass goes sparse from 8,204 cells
#   r075   0.75: from 6,153 cells
#   r0625  0.625: from 5,128 cells
# A request at depth D generates 256 tokens, so its verify passes read D to D + 256 cells plus the cache's padding:
# at 4,096 every arm runs the dense kernel, at 5,120 only r0625 runs the sparse one, at 6,144 and 7,168 r075 and r0625
# do, and at 8,192 all three do.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as C865, R162 and R205), PRESET (default
# release/flash-next-v5/settings.env), OUT (here, for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of c10, then two loads an arm with the arms' order
#            reversed in the second round (Q1253): c10, r075, r0625, r0625, r075, c10; depths 4,096 to 8,192 in steps
#            of 1,024; 256 tokens at temperature 0 twice per depth, then once with the preset's sampler
#            (SAMPLER_PASS=1), every reply kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control c10.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r69}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SW=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="4096 5120 6144 7168 8192" \
    ENV_DROP=$SW ARMS="c10 r075 r0625" ARM_ENV_c10="$SW=1.0" ARM_ENV_r075="$SW=0.75" ARM_ENV_r0625="$SW=0.625" \
    LOADS="warm:c10 c10a:c10 r075a:r075 r0625a:r0625 r0625b:r0625 r075b:r075 c10b:c10"
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
