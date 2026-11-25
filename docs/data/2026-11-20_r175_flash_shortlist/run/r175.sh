#!/usr/bin/env bash
# R175 (Q1507): how many drafting rounds Qwen3.8-Flash-Next's MTP draft runs with its full output head after a target
# token outside its shortlist, on the deep lane. The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5
# GSQ-RCO Q2_0, its n-gram table in a third shard, and the MTP draft, on) on nine P106-090 cards, one build (a
# pascal-p106-v5 tree). LLAMA_DRAFT_SHORTLIST=<k_low>,<m_cap>,<rounds> (common/draft-shortlist.h): the draft's head reads
# the ids below k_low, the special tokens and up to m_cap of the context's tokens; when the target's latest token is
# outside that list, the next <rounds> drafting rounds (the round right after the miss included) read the full head,
# and the token joins the list. Arms (the preset's own value is dropped and each arm sets its one value):
#   c      32768,8192,2, the preset's value (control): two full-head rounds after a miss
#   fb1    32768,8192,1: one full-head round after a miss
# VARIANT=r230 (R230, Q1643, 2026-11-25) runs two other arms, the rest unchanged (WORK's default /tmp/r230):
#   c1     32768,8192,1 (control): one full-head round after a miss, R175's fb1
#   s0     32768,8192,0: no full-head round after a miss; the missed token still joins the list
#   loads  warm:c1 c1a:c1 s0a:s0 s0b:s0 c1b:c1
# The server logs each load's count at its end ("draft head shortlist: N drafts with it, M with the full head, K target
# tokens outside it"), so the arms' full-head rounds can be compared with their steps.
# This is R69's deep-lane design (docs/data/2026-11-20_r69_sparse_switch/run/r69.sh) with the fallback as the arm.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as C865, R162, R205 and R69), PRESET
# (default release/flash-next-v5/settings.env), OUT (here, for collect). No kernel cases run on the rig. STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of c, then two loads an arm with the arms' order
#            reversed in the second round (Q1253): c, fb1, fb1, c; depths 0, 4,096 and 8,192; 256 tokens at
#            temperature 0 twice per depth, then once with the preset's sampler (SAMPLER_PASS=1), every reply kept
#            (KEEP_REPLY=1) so a difference can be found by token
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control c.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
VARIANT=${VARIANT:-r175}
case $VARIANT in r175|r230) ;; *) echo "VARIANT=r175|r230"; exit 1 ;; esac
export RIG WORK=${WORK:-/tmp/$VARIANT}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
SW=LLAMA_DRAFT_SHORTLIST
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="0 4096 8192" \
    ENV_DROP=$SW ARMS="c fb1" ARM_ENV_c="$SW=32768,8192,2" ARM_ENV_fb1="$SW=32768,8192,1" \
    LOADS="warm:c ca:c fb1a:fb1 fb1b:fb1 cb:c"
  if [ "$VARIANT" = r230 ]; then
    export ARMS="c1 s0" ARM_ENV_c1="$SW=32768,8192,1" ARM_ENV_s0="$SW=32768,8192,0" \
      LOADS="warm:c1 c1a:c1 s0a:s0 s0b:s0 c1b:c1"
    unset ARM_ENV_c ARM_ENV_fb1
  fi
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
