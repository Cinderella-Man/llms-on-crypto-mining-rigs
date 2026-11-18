#!/bin/bash
# soak_chain.sh: one fresh boot, no one in the loop. The served model's stock reference, `climb.py apply` with the
# lowest profile, the soak (a ladder when LADDER names more profiles), and the served model again with the offsets
# set. Run detached, as root, in the kit folder:
#
#   PROFILE=<rung 1.json> LADDER="<rung 2.json> <rung 3.json>" setsid nohup bash soak_chain.sh > chain.console 2>&1 < /dev/null &
#
# Environment (ENV_FILE, default ./env-chain.sh, is sourced first if it exists):
#   WORK          a fresh work folder for this boot (it gets every log and marker)
#   BIN, CHECK_MODEL   the kit's settings, as for climb.py
#   PROFILE       the profile `apply` sets: the lowest rung
#   LADDER        the next rungs' profiles, lowest first (optional)
#   MINUTES       minutes a rung [30];  SOAK_ARGS  soak.py's mode [--keep-going]
#   FAILURES      the cards' failures from earlier runs, for the use profile (soak.py --failures; optional)
#   MODEL, SETTINGS, OUT   the whole-memory check (model_check.sh); left out when MODEL is unset
#   STOCK_MINUTES minutes of the model at stock [0: its two rounds only, the reference]
#   MODEL_MINUTES minutes of the model with the offsets set [30]
#   END_BY        no step starts unless it ends by this time of day [21:50]
#
# Every step writes its own marker; the chain writes WORK/CHAIN_DONE at the end. After a soak in which a card
# differed or left, every offset is set to 0; WORK/soak/soaked.json holds the values that held and
# WORK/soak/use.json the setting to use by the rule, the next fresh boot's PROFILE. When everything passes, the
# offsets stay set while the watchdog runs.
set -u -o pipefail
cd "$(dirname "$0")"
ENV_FILE=${ENV_FILE:-./env-chain.sh}
if [ -f "$ENV_FILE" ]; then . "$ENV_FILE"; fi
: "${PROFILE:?PROFILE=<the lowest rung profile.json>}" "${WORK:?WORK=<a fresh work folder>}"
LADDER=${LADDER:-} MINUTES=${MINUTES:-30} MODEL_MINUTES=${MODEL_MINUTES:-30} END_BY=${END_BY:-21:50}
SOAK_ARGS=${SOAK_ARGS:---keep-going} MODEL=${MODEL:-} STOCK_MINUTES=${STOCK_MINUTES:-0} FAILURES=${FAILURES:-}
export WORK
mkdir -p "$WORK"
log() { echo "$(date '+%F %T') chain: $*" | tee -a "$WORK/chain.log"; }
end() { log "$1"; echo "$1" > "$WORK/CHAIN_DONE"; exit 0; }
fits() {   # true when a step of $1 minutes that starts now ends by END_BY today (CHAIN_NOW=HH:MM: the tests' clock)
  local now=${CHAIN_NOW:-$(date +%H:%M)}
  [ $(( 10#${now%:*} * 60 + 10#${now#*:} + $1 )) -le $(( 10#${END_BY%:*} * 60 + 10#${END_BY#*:} )) ]
}
rm -f "$WORK/CHAIN_DONE"
set -- $LADDER
rungs=$(( $# + 1 ))

log "1. start check, then the model at stock (every offset 0)"
python3 climb.py start-check | tee -a "$WORK/chain.log" || end "start check failed: stop (a fresh boot first)"
model_stock=skipped
if [ -n "$MODEL" ] && ! fits "$(awk -v m="$STOCK_MINUTES" 'BEGIN{print int(m + 15 + 0.999)}')"; then
  model_stock="too late"
  log "the model at stock: not run, it would end after $END_BY"
elif [ -n "$MODEL" ]; then
  # MINUTES is the rung's: the stock reference gets its own, STOCK_MINUTES (0: two rounds and done)
  TAG=stock ROUNDS=2 MINUTES=$STOCK_MINUTES bash model_check.sh; model_stock=$?
  log "the model at stock: exit $model_stock"
  python3 climb.py start-check | tee -a "$WORK/chain.log" || end "start check failed after the model's stock run: stop"
fi

log "2. apply $PROFILE"
python3 climb.py watchdog start | tee -a "$WORK/chain.log"
python3 climb.py apply "$PROFILE" | tee -a "$WORK/chain.log"
while [ ! -f "$WORK/APPLY_DONE" ]; do sleep 3; done
[ "$(cat "$WORK/APPLY_DONE")" = "exit 0" ] || end "apply failed ($(cat "$WORK/APPLY_DONE")): every offset 0, stop"

log "3. the soak: $rungs rung(s) of $MINUTES min ($SOAK_ARGS)"
fits "$(awk -v r="$rungs" -v m="$MINUTES" 'BEGIN{print int(r * m + 5 + 0.999)}')" \
  || end "no soak: it would end after $END_BY; the profile stays set with the watchdog"
python3 soak.py --minutes "$MINUTES" $SOAK_ARGS --profile "$PROFILE" ${FAILURES:+--failures "$FAILURES"} \
  ${LADDER:+--ladder $LADDER}
soak_done=$(cat "$WORK/SOAK_DONE" 2>/dev/null || echo "no SOAK_DONE")
case "$soak_done" in
  "exit 0 "*) ;;
  *) python3 climb.py reset-all | tee -a "$WORK/chain.log"
     end "the soak did not pass ($soak_done): every offset 0; the values that held are in $WORK/soak/soaked.json, the setting to use in $WORK/soak/use.json, the next fresh boot's profile" ;;
esac

log "4. the model with the offsets set"
[ "$model_stock" = 0 ] || end "the soak passed; no model reference ($model_stock): the offsets stay set with the watchdog"
fits "$(awk -v m="$MODEL_MINUTES" 'BEGIN{print int(m + 15 + 0.999)}')" \
  || end "the soak passed; no model check: it would end after $END_BY; the offsets stay set with the watchdog"
TAG=profile ROUNDS=2 MINUTES=$MODEL_MINUTES bash model_check.sh; rc=$?
if [ "$rc" = 0 ]; then
  end "done: apply, the soak and the model with the offsets all passed; the offsets stay set with the watchdog"
fi
python3 climb.py reset-all | tee -a "$WORK/chain.log"
end "the model with the offsets failed (exit $rc): every offset 0"
