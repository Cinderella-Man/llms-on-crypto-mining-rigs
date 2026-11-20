#!/usr/bin/env bash
# r183o.sh load|run|end|all: R183's Ornith arm (Q1546) on one rig, the steps in the order they ran. R190's method
# (docs/data/2026-11-17_r183_recall_depth/run/r190.sh) on Ornith-1.5-122B Pascal N2P GGUF: one load of llama-server on
# the v5 preset (release/ornith-v5/settings.env, unchanged; no draft), then recall_depth.py from this machine through an
# ssh tunnel. Ornith's preset holds 49,152 tokens, so the prompts are about 8,000 (the control), 16,000, 32,000 and
# 47,000 tokens; the facts sit 48,000, 32,000, 16,000, 4,000, 1,000 and 300 tokens from the end, and a fact deeper than a
# prompt goes to its start (in the 47,000-token prompt the deepest fact is its first line). Seed 1546, greedy, R190's
# question (QUESTION=kinds: which pet, city, car, job, film and drink), every prompt sent whatever the one before gave
# (ON_MISS=go). Then each prompt that missed a fact is sent once more with R183's question (QUESTION=list: every fact as
# a list), also greedy.
#   load  copies serve.sh, the preset and the RAM watchdog to the rig, starts the server, waits until it answers
#   run   the prompts (the tunnel must be up: ssh -N -L $LPORT:127.0.0.1:$PORT "$RIG")
#   end   stops the server (the idle check after), copies the rig's files to OUT/rig and removes them there
#   all   load, the tunnel, run, end: the server is stopped and collected whatever run gives
# Env: RIG (ssh alias), BIN (the build's bin folder on the rig), MODEL (the model file on the rig), OUT (results here),
# WORK (on the rig, default /tmp/recall-depth), PORT (on the rig, 8080), LPORT (the tunnel's local port, default 18080),
# TARGETS (default "8000 16000 32000 47000"), STOP_AT (a time for date -d, e.g. 21:50: run is stopped there, and all
# stops the server after it; default none). Every call to the rig is under a timeout.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/../../../..
TOOL=$ROOT/bench/recall_depth
: "${RIG:?}" "${OUT:?}"
LPORT=${LPORT:-18080}
PORT=${PORT:-8080}
TARGETS=${TARGETS:-8000 16000 32000 47000}
export RIG BIN MODEL PORT
export DRAFT=''
export PRESET=$ROOT/release/ornith-v5/settings.env CTX=49152
export SEED=1546 MAX_TOKENS=8192 TIMEOUT_S=3600 ON_MISS=go
log() { echo "=== $(date '+%F %T') $*"; }

left() {  # seconds until STOP_AT (a large number without it); at least 1
  [ -z "${STOP_AT:-}" ] && { echo 86400; return; }
  local s=$(( $(date -d "$STOP_AT" +%s) - $(date +%s) ))
  [ "$s" -lt 1 ] && s=1
  echo "$s"
}

client() {  # client QUESTION NAME TARGETS...: recall_depth.py run, stopped at STOP_AT
  local q=$1 name=$2; shift 2
  # shellcheck disable=SC2068
  QUESTION=$q NAME=$name URL=http://127.0.0.1:$LPORT OUT=$OUT/client \
    timeout "$(left)" python3 -I "$TOOL/recall_depth.py" run $@
}

missed() {  # the targets whose kinds prompt missed a fact (results.tsv: target, pass, question)
  awk -F'\t' 'NR > 1 && $19 == "kinds" && $9 == "0" { print $3 }' "$OUT/client/results.tsv" 2> /dev/null
}

load() {
  # start has its own timeout; wait returns when the server answers, or after serve.sh's LOAD_TIMEOUT_S (1,200 s)
  timeout 120 bash "$TOOL/remote.sh" put && bash "$TOOL/remote.sh" start && timeout 1300 bash "$TOOL/remote.sh" wait
}

run() {
  log "R190's question (kinds) at $TARGETS"
  # shellcheck disable=SC2086
  client kinds kinds $TARGETS
  log "kinds exit $?"
  local m
  m=$(missed | tr '\n' ' ')
  if [ -n "$m" ]; then
    log "missed at $m: once more with R183's question (list)"
    # shellcheck disable=SC2086
    client list list $m
    log "list exit $?"
  else
    log "no miss: R183's question not needed"
  fi
}

end() {
  timeout 120 bash "$TOOL/remote.sh" stop
  OUT=$OUT/rig timeout 600 bash "$TOOL/remote.sh" collect
}

case ${1:?usage: r183o.sh load|run|end|all} in
load) load ;;
run)  run ;;
end)  end ;;
all)
  mkdir -p "$OUT"
  log "load"
  if load; then
    log "server up; tunnel on $LPORT"
    timeout "$(left)" ssh -N -o ExitOnForwardFailure=yes -L "$LPORT:127.0.0.1:$PORT" "$RIG" &
    tun=$!
    sleep 5
    run
    kill "$tun" 2> /dev/null
  else
    log "load failed"
  fi
  log "end"
  end
  log "all done"
  ;;
*)    echo "usage: r183o.sh load|run|end|all"; exit 1 ;;
esac
