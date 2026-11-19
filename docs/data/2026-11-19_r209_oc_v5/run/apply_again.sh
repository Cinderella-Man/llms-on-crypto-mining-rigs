#!/bin/bash
# apply_again.sh: the profile again on a later fresh boot (R211, Q1621), on a rig where apply_check.sh already passed
# with the same profile, engine and preset. The kit's start check, the watchdog, the kit's `apply` of the profile, then
# the served model's whole-memory check with the offsets for ROUNDS rounds. No stock check and no soak: the stock
# reference is the earlier check's (its greedy-stock1.sha and greedy-stock1.json, copied from REF into a fresh OUT, so
# every round is compared with the same stock answer), and the profile held its soak (R202b). Any failure once an
# offset could be set: every offset back to 0 and the watchdog stopped. On a pass the offsets stay set and the watchdog
# runs, with WORK as its work folder (read the kit's state with an env file that sets this WORK).
#
#   KIT=<the offset kit's folder> BIN=<the v5 tree>/bin MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
#     SETTINGS=<KIT>/ornith-v5.env PROFILE=<KIT>/r202b-use.json REF=<apply_check.sh's OUT> REF_HASH=adf47784f9b02d4b \
#     WORK=<fresh work folder> OUT=<fresh model folder> ROUNDS=2 \
#     setsid -f nohup bash apply_again.sh > <console> 2>&1 < /dev/null
#
# The kit's env.sh gives `apply` its check model, its memory steps and the engine of its check servers; BIN goes only to
# model_check.sh. The done marker WORK/APPLY_AGAIN_DONE says "done 0" on a pass, else what failed.
: "${KIT:?KIT=<the offset kit folder>}" "${BIN:?BIN=<the v5 tree's bin folder>}" "${MODEL:?MODEL=<the GGUF>}"
: "${SETTINGS:?SETTINGS=<release settings.env>}" "${PROFILE:?PROFILE=<the use profile>}"
: "${REF:?REF=<the earlier check's model folder>}" "${REF_HASH:?REF_HASH=<its stock hash>}"
: "${WORK:?WORK=<fresh work folder>}" "${OUT:?OUT=<fresh model check folder>}"
ROUNDS=${ROUNDS:-2}
V5=$BIN W=$WORK
cd "$KIT" && . ./env.sh
export WORK=$W OUT MODEL SETTINGS
mkdir -p "$WORK"
log() { echo "$(date '+%F %T') apply_again: $*" | tee -a "$WORK/apply_again.log"; }
status="ended early"
trap 'echo "done $status" > "$WORK/APPLY_AGAIN_DONE"' EXIT
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
log "start: BIN $V5, MODEL $MODEL, SETTINGS $SETTINGS ($(md5sum < "$SETTINGS" | cut -c1-32)), PROFILE $PROFILE ($(md5sum < "$PROFILE" | cut -c1-32)), REF $REF, OUT $OUT, WORK $WORK, ROUNDS $ROUNDS"
[ -e "$OUT" ] && fail "OUT $OUT exists" noreset
mkdir -p "$OUT" && cp "$REF/greedy-stock1.sha" "$REF/greedy-stock1.json" "$OUT/" || fail "no stock reference in $REF" noreset
[ "$(cat "$OUT/greedy-stock1.sha")" = "$REF_HASH" ] || fail "the reference's hash is $(cat "$OUT/greedy-stock1.sha"), not $REF_HASH" noreset
log "stock reference from $REF: $(cat "$OUT/greedy-stock1.sha") ($(md5sum < "$OUT/greedy-stock1.json" | cut -c1-32) the responses)"
python3 climb.py state > "$WORK/state_start.txt" 2>&1
python3 climb.py start-check || fail "start check" noreset
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
BIN=$V5 TAG=profile ROUNDS=$ROUNDS MINUTES=0 bash model_check.sh; rc=$?; log "profile rc $rc"
[ $rc = 0 ] || fail "profile check rc $rc" reset
python3 climb.py state > "$WORK/state_end.txt" 2>&1
status=0
log "passed: the offsets stay set, the watchdog running"
