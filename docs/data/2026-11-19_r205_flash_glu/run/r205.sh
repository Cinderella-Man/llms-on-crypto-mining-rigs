#!/usr/bin/env bash
# R205 (Q1594), its Flash-Next half: GGML_CUDA_MMID_GLU_UNFUSE unset, 1 and 2 on Qwen3.8-Flash-Next as its v4 preset
# runs it (release/flash-next-v4/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table in a third shard, and
# the MTP draft, on) on nine P106-090 cards, all three arms on one build (a pascal-p106-v5 tree that carries the
# switch). This is C865's deep-lane design
# (docs/data/2026-11-17_c865_ornith_glu_unfuse/run/c865.sh) with the preset, the model and the draft as variables.
# Arms:
#   off  the switch unset: v4.0's fusions
#   u1   GGML_CUDA_MMID_GLU_UNFUSE=1: a routed experts' gate and up products (MUL_MAT_ID) as two launches, then the
#        GLU; the shared expert stays fused, and the one launch for the routed and the shared triple is not made
#   u2   GGML_CUDA_MMID_GLU_UNFUSE=2: upstream's #32239, no fused mul_mat_vec_q on Pascal: the routed and the shared
#        expert's gate, up and GLU apart, and every product with its bias or scale. It reaches the routed experts of a
#        verify pass of 2 to 8 tokens too. The preset's GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 (the shared expert's gate, up
#        and SWIGLU of a 2-to-8-token pass as one launch, the fork's own kernel) is another path and runs in every arm
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as C865 and R162), PRESET (default
# release/flash-next-v4/settings.env), OUT (here, for collect). No kernel cases run on the rig (C865's STEP=tbo: its
# CPU reference takes hours on the rig's two-core host). STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: loads warm:off, then off, u1, u2 twice, alternated; depths 0 and
#            8,192; 256 tokens at temperature 0 twice per depth (the second pass with the preset's sampler,
#            SAMPLER_PASS=1), every reply kept (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control off.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r205}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SW=GGML_CUDA_MMID_GLU_UNFUSE
common() {
  export PRESET=${PRESET:-release/flash-next-v4/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 8192" \
    ARMS="off u1 u2" ARM_ENV_u1="$SW=1" ARM_ENV_u2="$SW=2" \
    LOADS="warm:off f1:off a1:u1 b1:u2 f2:off a2:u1 b2:u2"
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
