#!/bin/bash
# half_check.sh: one half of the card search (R226, Q1640) on a fresh boot, as root, detached. R211's apply_again.sh
# with the kit's trip test and the check's length in minutes: the kit's start check, the trip test, the watchdog, the
# kit's `apply` of the half profile (half_profile.py's file: some cards at the profile's values, the others at 0), then
# the served model's whole-memory check with those offsets, rounds for MINUTES after the load, each compared with the
# stored stock answer (REF's greedy-stock1.sha and greedy-stock1.json, copied into a fresh OUT and its hash checked);
# the check stops at the first round that differs. No stock rounds on this boot. At the end, pass or fail: every
# offset back to 0, the watchdog stopped, the kit's state written. The next half needs a fresh boot.
#
#   KIT=<the offset kit's folder> BIN=<the v5 tree>/bin MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
#     SETTINGS=<KIT>/ornith-v5.env PROFILE=<the half profile> REF=<an earlier check's model folder> \
#     REF_HASH=adf47784f9b02d4b WORK=<fresh work folder> OUT=<fresh model folder> MINUTES=30 \
#     setsid -f nohup bash half_check.sh > <console> 2>&1 < /dev/null
#
# The kit's env.sh gives `apply` its check model, its memory steps and the engine of its check servers; BIN goes only to
# model_check.sh. The done marker WORK/HALF_DONE says "done 0" when every round gave the stock answer, else what
# failed ("profile check rc 1": a round differed, or the load or a round failed; model.log says which).
: "${KIT:?KIT=<the offset kit folder>}" "${BIN:?BIN=<the v5 tree bin folder>}" "${MODEL:?MODEL=<the GGUF>}"
: "${SETTINGS:?SETTINGS=<release settings.env>}" "${PROFILE:?PROFILE=<the half profile>}"
: "${REF:?REF=<the earlier check model folder>}" "${REF_HASH:?REF_HASH=<its stock hash>}"
: "${WORK:?WORK=<fresh work folder>}" "${OUT:?OUT=<fresh model check folder>}"
MINUTES=${MINUTES:-30} ROUNDS=${ROUNDS:-2}
V5=$BIN W=$WORK
cd "$KIT" && . ./env.sh
export WORK=$W OUT MODEL SETTINGS
mkdir -p "$WORK"
log() { echo "$(date '+%F %T') half_check: $*" | tee -a "$WORK/half_check.log"; }
status="ended early"
offsets=no
finish() {
  if [ "$offsets" = yes ]; then
    python3 climb.py stop > /dev/null 2>&1
    python3 climb.py reset-all; log "reset-all rc $?"
    python3 climb.py watchdog stop; log "watchdog stop rc $?"
  fi
  python3 climb.py state > "$WORK/state_end.txt" 2>&1
  echo "done $status" > "$WORK/HALF_DONE"
}
trap finish EXIT
fail() { status="$1"; log "FAILED: $1"; exit 1; }
log "start: BIN $V5, MODEL $MODEL, SETTINGS $SETTINGS ($(md5sum < "$SETTINGS" | cut -c1-32)), PROFILE $PROFILE ($(md5sum < "$PROFILE" | cut -c1-32)), REF $REF, OUT $OUT, WORK $WORK, MINUTES $MINUTES"
[ -e "$OUT" ] && fail "OUT $OUT exists"
mkdir -p "$OUT" && cp "$REF/greedy-stock1.sha" "$REF/greedy-stock1.json" "$OUT/" || fail "no stock reference in $REF"
[ "$(cat "$OUT/greedy-stock1.sha")" = "$REF_HASH" ] || fail "the reference's hash is $(cat "$OUT/greedy-stock1.sha"), not $REF_HASH"
log "stock reference from $REF: $(cat "$OUT/greedy-stock1.sha") ($(md5sum < "$OUT/greedy-stock1.json" | cut -c1-32) the responses)"
python3 climb.py state > "$WORK/state_start.txt" 2>&1
python3 climb.py start-check || fail "start check"
python3 climb.py trip-test > "$WORK/trip_test.txt" 2>&1; rc=$?
log "trip-test rc $rc: $(grep -c ': PASS' "$WORK/trip_test.txt") PASS, $(grep -c ': FAIL' "$WORK/trip_test.txt") FAIL"
[ $rc = 0 ] || fail "trip test rc $rc"
offsets=yes
python3 climb.py watchdog start || fail "watchdog start"
log "watchdog started"
python3 climb.py apply "$PROFILE" || fail "apply did not start"
t=0
while [ ! -f "$WORK/APPLY_DONE" ]; do
  sleep 3; t=$((t + 3))
  [ $t -gt 2400 ] && fail "apply: no APPLY_DONE after 2400 s"
done
log "apply $(cat "$WORK/APPLY_DONE")"
[ "$(cat "$WORK/APPLY_DONE")" = "exit 0" ] || fail "apply $(cat "$WORK/APPLY_DONE")"
python3 climb.py state > "$WORK/state_after_apply.txt" 2>&1
log "pids folder: $(ls "$WORK/pids" 2>&1 | tr '\n' ' ')"
BIN=$V5 TAG=profile ROUNDS=$ROUNDS MINUTES=$MINUTES bash model_check.sh; rc=$?; log "profile rc $rc"
[ $rc = 0 ] || fail "profile check rc $rc"
status=0
log "passed: every round the stock answer; now every offset back to 0 and the watchdog stopped"
