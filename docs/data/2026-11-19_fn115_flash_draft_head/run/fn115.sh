#!/usr/bin/env bash
# FN115 (Q1545): Qwen3.8-Flash-Next as its v4 preset runs it (release/flash-next-v4/settings.env: UkisAI's Swift-1.5
# GSQ-RCO Q2_0, its n-gram table in a third shard, the MTP draft on) on nine P106-090 cards, with two MTP draft heads:
#   own   the preset's head (mtp-Qwen3.8-Flash-Next-draft.gguf: its own token table and Q6_K output head, the layer's
#         experts in Q4_K and IQ4_NL, its layer a sparse-attention (QSA) layer)
#   dist  a published distilled head (mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf, distilled against Unsloth's
#         UD-Q4_K_XL target): no token table and no output head of its own (nextn_shared_target_tensors), so it runs
#         with LLAMA_DRAFT_SHARE=embd,head and reads the target's; the layer's experts in Q8_0; its file marks the layer
#         compress ratio 0, so the layer attends to every cell (no QSA). Every other switch of the preset is the same.
# This is R205's deep-lane design (docs/data/2026-11-19_r205_flash_glu/run/r205.sh) with the arm being the draft file
# instead of a switch. Every timed load also writes a step trace (LLAMA_STEP_TRACE: one line per server step with the
# accepted tokens and host marks), so each load is its own arm in the lane (own1, own2, dist1, dist2: a trace file per
# load); run/relabel.py names the arms own and dist again before the tables. The warm-up load runs the preset as it is.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the preset's MTP head on the rig),
# MODEL_DIST (the published head on the rig), TEXT_LOCAL (the recorded agent text here: the release check's
# proxy.jsonl.bodies.gz, as R205), PRESET (default release/flash-next-v4/settings.env), OUT (here, for collect).
# No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   probe    one untimed-in-the-verdict load of dist (depths 0 and 8,192, the same requests) in WORK_probe, detached:
#            does the head load, fit and answer; then STEP=pwait and STEP=pcollect OUT=<folder>
#   start    bench/deep_lane on the rig, detached: loads warm:own0 f1:own1 d1:dist1 f2:own2 d2:dist2; depths 0 and
#            8,192; 256 tokens at temperature 0 twice per depth (the second pass with the preset's sampler,
#            SAMPLER_PASS=1), every reply kept (KEEP_REPLY=1)
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: OUT=<folder> RES=<tables> bash run/tables.sh
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/fn115}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SHARE=LLAMA_DRAFT_SHARE=embd,head
common() {
  export PRESET=${PRESET:-release/flash-next-v4/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 8192"
  : "${MODEL_DIST:?MODEL_DIST=<the published head on the rig>}"
}
trace() { echo "LLAMA_STEP_TRACE=$1/st-$2.jsonl"; }  # trace OUTDIR LOAD
case ${STEP:-start} in
dry | start)
  common
  O=$WORK/out
  export ARMS="own0 own1 dist1 own2 dist2" LOADS="warm:own0 f1:own1 d1:dist1 f2:own2 d2:dist2"
  export ARM_ENV_own1="$(trace "$O" f1)" ARM_ENV_own2="$(trace "$O" f2)"
  export ARM_ENV_dist1="$SHARE $(trace "$O" d1)" ARM_ENV_dist2="$SHARE $(trace "$O" d2)"
  export ARM_ARGS_dist1="-md $MODEL_DIST" ARM_ARGS_dist2="-md $MODEL_DIST"
  if [ "${STEP:-start}" = dry ]; then DRY=1 bash $L start; else bash $L start; fi
  ;;
probe)
  common
  export WORK=${WORK}_probe
  export ARMS="dist0" LOADS="p1:dist0" ARM_ENV_dist0="$SHARE $(trace "$WORK/out" p1)" ARM_ARGS_dist0="-md $MODEL_DIST"
  bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
pwait) WORK=${WORK}_probe bash $L wait ;;
pcollect) OUT=${OUT:?} WORK=${WORK}_probe bash $L collect ;;
stop) bash $L stop ;;
*) echo "STEP=dry|probe|pwait|pcollect|start|wait|status|collect|stop"; exit 1 ;;
esac
