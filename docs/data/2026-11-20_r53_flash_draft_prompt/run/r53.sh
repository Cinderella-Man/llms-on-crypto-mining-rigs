#!/usr/bin/env bash
# R53 again (Q1215): how much Qwen3.8-Flash-Next's MTP draft costs in prompt reading on the v5 engine and the v5 preset
# (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table in a third shard, the MTP
# draft) on nine P106-090 cards, one build (a pascal-p106-v5 tree). The draft model reads every prompt token too, so a
# prompt costs more with it loaded; on v3.0's line (2026-11-10, docs/data/2026-11-10_deep_lane/) that was 28.4 % on an
# empty cache and 41.3 % on 32,188 cached tokens. This is R69's deep-lane design
# (docs/data/2026-11-20_r69_sparse_switch/run/r69.sh) with the draft as the arm. Arms:
#   draft    the preset as it is (control)
#   nodraft  the same line without the draft: no -md and none of the draft's flags (--spec-*), every switch kept
# Depths 8,192, 16,384 and 32,768 with the cache extended at each depth, so a load reads three prompt chunks: 8,192
# tokens on an empty cache, then about 8,192 on the first chunk's checkpoint, then about 16,384 on the second's (the
# first three of v3.0's four chunks). 16 tokens at temperature 0 twice per depth (N_PREDICT=16): the run is about the
# prompts. The host's lowest available memory per load is in memory.tsv.
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the release check's proxy.jsonl.bodies.gz, as R69), PRESET (default
# release/flash-next-v5/settings.env), OUT (here, for collect), LOADS (default below). STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached: a warm-up load of draft at depth 0, then two loads an arm in the
#            order draft, nodraft, nodraft, draft
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
#   stop     ends the run; the finished loads stay in WORK/out
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control draft.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r53}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^t1swift/' N_PREDICT=16 DEPTHS="8192 16384 32768" \
    ARMS="draft nodraft" ARM_DRAFT_draft=on ARM_DRAFT_nodraft=off \
    LOADS=${LOADS:-"warm:draft d1:draft n1:nodraft n2:nodraft d2:draft"}
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
