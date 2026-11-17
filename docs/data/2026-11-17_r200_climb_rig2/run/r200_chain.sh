#!/bin/bash
# R200's last try, one boot, no one in the loop: Ornith's stock reference, then `apply` with the profile, the soak of
# all nine cards, and Ornith with the profile set. Run detached, as root, in the kit folder, on a fresh boot:
#
#   PROFILE=<profile.json> SOAK_BY=21:25 ORNITH_BY=21:40 setsid nohup bash r200_chain.sh > chain.console 2>&1 &
#
# env-ornith.sh gives WORK, BIN, CHECK_MODEL, MODEL, SETTINGS and OUT. Every step writes its own marker; the chain
# writes WORK/CHAIN_DONE at the end. A step that would start after its time (SOAK_BY, ORNITH_BY, so that nothing timed
# runs past 22:00) is skipped. If apply or the soak fails, the kit has already set every offset to 0.
set -u -o pipefail
cd "$(dirname "$0")"
. ./env-ornith.sh
: "${PROFILE:?PROFILE=<profile.json>}"
SOAK_BY=${SOAK_BY:-21:25} ORNITH_BY=${ORNITH_BY:-21:40}
mkdir -p "$WORK"
log() { echo "$(date '+%F %T') chain: $*" | tee -a "$WORK/chain.log"; }
end() { log "$1"; echo "$1" > "$WORK/CHAIN_DONE"; exit 0; }
before() { [ "$(date +%H:%M)" \< "$1" ]; }
rm -f "$WORK/CHAIN_DONE"

log "1. Ornith at stock (every offset 0)"
python3 climb.py start-check | tee -a "$WORK/chain.log"
TAG=stock ROUNDS=2 bash ornith_check.sh; ornith_stock=$?
log "Ornith stock: exit $ornith_stock"

log "2. apply $PROFILE"
python3 climb.py start-check | tee -a "$WORK/chain.log" || end "start check failed after Ornith's stock run: stop"
python3 climb.py watchdog start | tee -a "$WORK/chain.log"
python3 climb.py apply "$PROFILE" | tee -a "$WORK/chain.log"
while [ ! -f "$WORK/APPLY_DONE" ]; do sleep 3; done
[ "$(cat "$WORK/APPLY_DONE")" = "exit 0" ] || end "apply failed ($(cat "$WORK/APPLY_DONE")): every offset 0, stop"

log "3. the soak"
before "$SOAK_BY" || end "no soak: it would start after $SOAK_BY; the profile stays set with the watchdog"
python3 soak.py --minutes 30
case "$(cat "$WORK/SOAK_DONE")" in
  "exit 0 "*) ;;
  *) end "soak failed ($(cat "$WORK/SOAK_DONE")): every offset 0, stop" ;;
esac

log "4. Ornith with the profile set"
[ "$ornith_stock" = 0 ] || end "no Ornith reference (exit $ornith_stock): the profile stays set with the watchdog"
before "$ORNITH_BY" || end "no Ornith check: it would start after $ORNITH_BY; the profile stays set with the watchdog"
TAG=profile ROUNDS=10 bash ornith_check.sh; rc=$?
if [ "$rc" = 0 ]; then
  end "done: apply, the soak and Ornith with the profile all passed; the profile stays set with the watchdog"
fi
python3 climb.py reset-all | tee -a "$WORK/chain.log"
end "Ornith with the profile failed (exit $rc): every offset 0"
