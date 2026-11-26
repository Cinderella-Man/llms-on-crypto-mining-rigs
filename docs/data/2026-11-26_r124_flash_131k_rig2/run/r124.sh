#!/usr/bin/env bash
# R124 (Q1403): does Qwen3.8-Flash-Next's v5 line hold 131,072 tokens of context on nine P106-090 cards and the 4 GB
# host, what it decodes and reads at 49,152, 65,536 and 120,000 tokens, and the MTP draft on against off there.
# The v5 preset (release/flash-next-v5/settings.env: UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table in a third
# shard, and the MTP draft) with two changes, both made by the lane's CTX and UB: -c 131072 instead of 65,536, and
# -ub 256 instead of 512. llama.cpp's own projection (scripts/fit_probe.sh, fit_probe.tsv in this folder) leaves card 8,
# which holds the output head and the draft, 13 MiB at -ub 512 and 250 MiB at -ub 256. Everything else is the preset's,
# --spec-draft-n-ubatch 512 included.
# Arms: draft (the line as above) and nodraft (the same line without -md and the draft's flags, DRAFT_FLAGS of
# bench/deep_lane/deep_lane.sh). A warm-up load of draft at depth 0 (the load check), then two loads an arm with the
# second round's order reversed (Q1253): draft1, nodraft1, nodraft2, draft2.
# Depths 49,152, 65,536 and 120,000 (S38's and R55's, Q1154 and Q1218, on the same recorded text and tags, so the
# prompts are theirs): per depth a prime (the prompt reading on top of the cache), 256 tokens at temperature 0 twice,
# then once with the preset's sampler (SAMPLER_PASS=1); every reply kept (KEEP_REPLY=1).
# Run from the repository's root on a machine that runs only ssh and tar; everything timed runs detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the build's llama-server on the rig), MODEL
# (the model's first shard on the rig; the other shards beside it), MODEL_DRAFT (the MTP draft on the rig), TEXT_LOCAL
# (the recorded agent text here: the v3.0 release check's proxy.jsonl.bodies.gz, with proxy.jsonl beside it, as S38),
# PRESET (default release/flash-next-v5/settings.env), OUT (here, for collect). STEP:
#   dry      the lane's server lines, nothing started
#   start    bench/deep_lane on the rig, detached
#   wait     returns when the lane has ended; status: its last lines
#   collect  the lane's results to OUT; WORK is removed on the rig (CLEAN=0 keeps it)
# The tables then run where OUT is: python3 -I bench/deep_lane/deep_table.py OUT --control nodraft, and this folder's
# run/tables.sh.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/r124}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
common() {
  export PRESET=${PRESET:-release/flash-next-v5/settings.env} BIN=${BIN:?} MODEL=${MODEL:?} \
    MODEL_DRAFT=${MODEL_DRAFT:?} DRAFT=on TEXT_LOCAL=${TEXT_LOCAL:?} \
    TAG='^(t1swift|t1base)/' CTX=131072 UB=256 WD_ENV=MIN_AVAIL_MB=60 \
    SAMPLER_PASS=1 N_PREDICT=256 KEEP_REPLY=1 DEPTHS="49152 65536 120000" \
    ARMS="draft nodraft" ARM_DRAFT_nodraft=off \
    LOADS="warm:draft draft1:draft nodraft1:nodraft nodraft2:nodraft draft2:draft"
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
