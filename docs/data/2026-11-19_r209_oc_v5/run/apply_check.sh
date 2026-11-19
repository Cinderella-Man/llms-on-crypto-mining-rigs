#!/bin/bash
# apply_check.sh: R209's run 3 on the rig, as root, detached. soak_chain.sh's steps 1, 2 and 4 without the soak (the
# profile held its soak in R202b): the kit's start check, Ornith's whole-memory check at stock (two rounds, the
# reference), the watchdog, the kit's `apply` of the profile, then the same check with the offsets set for 30 minutes,
# with the same BIN, SETTINGS and OUT. Any failure once an offset could be set: every offset back to 0 and the watchdog
# stopped. On a pass the offsets stay set and the watchdog runs (the agent checks with the offsets come next).
#
#   KIT=<the offset kit's folder> BIN=<the v5 tree>/bin MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
#     SETTINGS=<KIT>/ornith-v5.env PROFILE=<KIT>/r202b-use.json WORK=<fresh work folder> OUT=<fresh model folder> \
#     setsid -f nohup bash apply_check.sh > <console> 2>&1 < /dev/null
#
# The kit's env.sh gives the check model and the memory steps `apply` uses (and an engine for apply's check servers);
# this script then sets WORK, and BIN only for model_check.sh. SETTINGS is release/ornith-v5/settings.env copied into
# the kit. The done marker WORK/R209C_DONE says "done 0" on a pass, else what failed.
: "${KIT:?KIT=<the offset kit folder>}" "${BIN:?BIN=<the v5 tree's bin folder>}" "${MODEL:?MODEL=<the GGUF>}"
: "${SETTINGS:?SETTINGS=<release settings.env>}" "${PROFILE:?PROFILE=<the use profile>}"
: "${WORK:?WORK=<fresh work folder>}" "${OUT:?OUT=<fresh model check folder>}"
V5=$BIN W=$WORK
cd "$KIT" && . ./env.sh
export WORK=$W OUT MODEL SETTINGS
mkdir -p "$WORK"
log() { echo "$(date '+%F %T') r209c: $*" | tee -a "$WORK/r209c.log"; }
status="ended early"
trap 'echo "done $status" > "$WORK/R209C_DONE"' EXIT
fail() {
  status="$1"; log "FAILED: $1"
  if [ "$2" = reset ]; then
    python3 climb.py stop > /dev/null 2>&1
    python3 climb.py reset-all; log "reset-all rc $?"
    python3 climb.py watchdog stop; log "watchdog stop rc $?"
    python3 climb.py state > "$WORK/state_after_reset.txt" 2>&1
  fi
  exit 1
}
log "start: BIN $V5, MODEL $MODEL, SETTINGS $SETTINGS ($(md5sum < "$SETTINGS" | cut -c1-32)), OUT $OUT, WORK $WORK"
python3 climb.py state > "$WORK/state_start.txt" 2>&1
python3 climb.py start-check || fail "start check" noreset
BIN=$V5 TAG=stock ROUNDS=2 MINUTES=0 bash model_check.sh; rc=$?; log "stock rc $rc"
[ $rc = 0 ] || fail "stock check rc $rc" noreset
python3 climb.py start-check || fail "start check after stock" noreset
python3 climb.py watchdog start || fail "watchdog start" reset
log "watchdog started"
python3 climb.py apply "$PROFILE" || fail "apply did not start" reset
t=0
while [ ! -f "$WORK/APPLY_DONE" ]; do
  sleep 3; t=$((t + 3))
  [ $t -gt 2400 ] && fail "apply: no APPLY_DONE after 2400 s" reset
done
log "apply $(cat "$WORK/APPLY_DONE")"
[ "$(cat "$WORK/APPLY_DONE")" = "exit 0" ] || fail "apply $(cat "$WORK/APPLY_DONE")" reset
python3 climb.py state > "$WORK/state_after_apply.txt" 2>&1
log "pids folder: $(ls "$WORK/pids" 2>&1 | tr '\n' ' ')"
BIN=$V5 TAG=profile ROUNDS=2 MINUTES=30 bash model_check.sh; rc=$?; log "profile rc $rc"
[ $rc = 0 ] || fail "profile check rc $rc" reset
python3 climb.py state > "$WORK/state_end.txt" 2>&1
status=0
log "passed: the offsets stay set, the watchdog running"
