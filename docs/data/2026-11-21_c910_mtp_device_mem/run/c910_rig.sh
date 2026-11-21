#!/usr/bin/env bash
# C910's rig check (registry Q1647): --mtp-device 0 on Ornith-1.5-122B Pascal N2P GGUF on nine P106-090 cards and the
# rig's 4 GB host, three loads, nothing timed. Each load is Ornith's v5 preset (release/ornith-v5/settings.env) with its
# MTP draft, as C899's MTP arms ran it (--spec-type draft-mtp --spec-draft-n-max 1, 16,384-token context, micro-batch
# 256), then two 32-token requests at temperature 0 (deep_lane's client, depth 0):
#   unset   the C910 build, --mtp-device unset: the MTP layer on the last card (-ts 6,6,6,6,6,5,5,5,5, C899's)
#   dev0    the C910 build, --mtp-device 0: the MTP layer, its KV and the output head's copy on card 0
#           (-ts 5,6,6,6,6,5,5,5,6: the unit C899 moved off the last card for the MTP layer comes back to it, and card 0
#           gives one up)
#   old0    the base build (pascal-p106-v5 3dfad876f, before C910) with dev0's line: its head copy goes through a host
#           buffer of the head's size, run last (a stop by the RAM watchdog ends the run, RETRY_STOPPED=0)
# What it records per load (bench/deep_lane/README.md): the server log (the "MTP output head copied" line and the model
# buffer sizes per card), fit_<load>.txt (memory used per card and the host's MemAvailable once the server is up),
# ram_watchdog-<load>.tsv (the host's MemAvailable and usable memory every second, from the server's start), the card
# log and the replies' token ids. rig_table.py makes the table.
# Run from the repository's root on the operator's machine, which runs only ssh and tar; the run is detached on the rig.
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN (the C910 build's llama-server on the rig),
# OLD_BIN (the base build's llama-server on the rig), MODEL (Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig),
# TEXT_LOCAL (recorded agent text here: the release check's proxy.jsonl.bodies.gz), OUT (here, for collect), LOADS
# (default "unset:unset dev0:dev0 old0:old0"), TS_DEV0 (dev0's and old0's split, default 5,6,6,6,6,5,5,5,6).
# STEP=start|wait|status|collect|stop; DRY=1 with start prints every load's server line and starts none.
set -uo pipefail
: "${RIG:?RIG=<ssh alias of the rig>}"
export RIG WORK=${WORK:-/tmp/c910}
[ -n "${DRY:-}" ] && export DRY
L=bench/deep_lane/remote.sh
MTP="--spec-type draft-mtp --spec-draft-n-max 1"
TS_DEV0=${TS_DEV0:-5,6,6,6,6,5,5,5,6}
case ${STEP:-start} in
start)
  : "${BIN:?}" "${OLD_BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
  for a in unset dev0 old0; do export "ARM_UB_$a=256" "ARM_CTX_$a=16384"; done
  export ARM_ARGS_unset="$MTP -ts 6,6,6,6,6,5,5,5,5"
  export ARM_ARGS_dev0="$MTP --mtp-device 0 -ts $TS_DEV0"
  export ARM_ARGS_old0="$MTP --mtp-device 0 -ts $TS_DEV0"
  export ARM_BIN_old0="$OLD_BIN"
  PRESET=release/ornith-v5/settings.env \
    BIN=$BIN MODEL=$MODEL DRAFT=off TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
    SAMPLER_PASS=0 PAUSE=0 WARM=0 REPS=1 RETRY_STOPPED=0 \
    DEPTHS=0 N_PREDICT=32 \
    ARMS="unset dev0 old0" \
    LOADS="${LOADS:-unset:unset dev0:dev0 old0:old0}" \
    bash $L start
  ;;
wait) bash $L wait ;;
status) bash $L status ;;
collect) OUT=${OUT:?} bash $L collect ;;
stop) bash $L stop ;;
esac
