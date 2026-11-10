#!/usr/bin/env bash
# run_pv_rig.sh: probabilistic verification on the MTP release line, timed on the rig (registry Q731). Every load is one
# llama-server on the rig at the line of release/flash-next-v3/settings.env, started through
# bench/agent_check/rig_server.sh (its monitors and RAM watchdog, as the release's agent arms ran);
# bench/agent_check/replay.py on the client re-sends the recorded requests of one agent-check arm to it, with the same
# prompts, seeds and reply limits in every load.
#
# Three machines, and the one this runs on only runs ssh and tar (as bench/agent_check/release_check.sh):
#   - the rig runs llama-server on its loopback;
#   - the client (any Linux machine with python3 and curl; CLIENT) runs replay.py under nice;
#   - this machine starts and stops everything over ssh and relays the client's requests to the rig through two ssh
#     forwards: -L RELAY_PORT -> rig 127.0.0.1:RIG_PORT and -R client 127.0.0.1:CLIENT_PORT -> RELAY_PORT.
# While a load's requests run, no command runs on the rig: this script polls the client only.
#
#   RIG=<ssh alias> CLIENT=<ssh alias> RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> \
#     BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<the MTP draft on the rig> \
#     RIG_LOGDIR=<folder on the rig> CLIENT_DIR=<folder on the client for the bench files> \
#     CLIENT_WORK=<results folder on the client> LOG=<the recorded proxy.jsonl on the client, its .bodies.gz beside it> \
#     OUT=<local folder> setsid nohup bash run_pv_rig.sh > console.txt 2>&1 < /dev/null &
#
# LOADS (default "warm pv0 pv1 pv0 pv1"): the arms in load order; the n-th load of an arm is labelled <arm>-<n>. An arm
# is the preset (PRESET_ENV, PRESET_ARGS, -md MODEL_DRAFT) plus ARGS_<arm>. Defaults:
#   warm  the preset, replaying only the recorded arm's first trace: it warms the cards (the first load after the rig
#         idled runs slow) and is in no table; its rows go to <label>.jsonl
#   pv0   the preset
#   pv1   ARGS_pv1="--spec-draft-sampling probabilistic --spec-draft-p-min 0"
# Before each load the rig must be quiet (no llama process, every card below FREE_MIB of used memory, default 50); its
# load average, free memory and each card's memory, temperature and clock go to OUT/quiet.txt. After the server is up,
# one untimed request (WARMUP, a request JSON of this repository, default the MTP smoke request; empty for none) takes
# the first-use costs, then the replay runs.
# Env: TAG (the recorded arm's tag pattern, default "^t1swift/"), STEPS ("prep run collect": prep copies replay.py,
# common.py and WARMUP to CLIENT_DIR; collect copies CLIENT_WORK to OUT/client), MIN_AVAIL_MB (the RAM watchdog's
# MemAvailable floor, default 60 as in the release's agent arms), WD_ENV (its other limits), REPO (this repository,
# default the checkout this file is in), PRESET.
# Per load: OUT/rig/<label>/ (server-<label>.log, gpu.csv, vmstat.log, meminfo.log, ram_watchdog-<label>.tsv; the same
# under CLIENT_WORK/rig/ for the tables), CLIENT_WORK/replay-<label>.jsonl, client-<label>.txt, warmup-<label>.json.
# A finished load leaves OUT/load-<label>.done and is skipped by a second start. Writes OUT/done or OUT/failed; prints
# CHAIN_DONE or CHAIN_FAILED as its last line.
set -uo pipefail
: "${RIG:?}" "${CLIENT:?}" "${RIG_PORT:?}" "${RELAY_PORT:?}" "${CLIENT_PORT:?}" "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}"
: "${RIG_LOGDIR:?}" "${CLIENT_DIR:?}" "${CLIENT_WORK:?}" "${LOG:?}" "${OUT:?}"
REPO=${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
LOADS=${LOADS:-warm pv0 pv1 pv0 pv1}
[ -n "${ARGS_pv1+set}" ] || ARGS_pv1='--spec-draft-sampling probabilistic --spec-draft-p-min 0'
TAG=${TAG:-^t1swift/}
STEPS=${STEPS:-prep run collect}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-60}
FREE_MIB=${FREE_MIB:-50}
WARMUP=${WARMUP-docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
RIG_SERVER=$REPO/bench/agent_check/rig_server.sh
CAC=$CLIENT_DIR/bench/agent_check
SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3)
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
rm -f "$OUT/done" "$OUT/failed"
echo $$ > "$OUT/chain.pid"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
cl() { ssh "${SSH_OPTS[@]}" "$CLIENT" "$@"; }
rigsh() { ssh "${SSH_OPTS[@]}" "$RIG" "$@"; }

relay_loop() {  # relay_loop NAME ssh-arguments…: keeps one forward up until killed
  local name=$1
  shift
  while true; do
    ssh -N "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes "$@" < /dev/null &
    echo $! > "$OUT/relay-$name.pid"
    wait $!
    sleep 3
  done
}

relay_start() {
  (relay_loop L -L "127.0.0.1:$RELAY_PORT:127.0.0.1:$RIG_PORT" "$RIG") >> "$OUT/relay.log" 2>&1 &
  RELAY_L=$!
  sleep 2
  (relay_loop R -R "127.0.0.1:$CLIENT_PORT:127.0.0.1:$RELAY_PORT" "$CLIENT") >> "$OUT/relay.log" 2>&1 &
  RELAY_R=$!
  sleep 3
  say "relay: client 127.0.0.1:$CLIENT_PORT -> here 127.0.0.1:$RELAY_PORT -> rig 127.0.0.1:$RIG_PORT"
}

relay_stop() {
  local x v
  for x in L R; do
    v=RELAY_$x
    [ -n "${!v:-}" ] && kill "${!v}" 2> /dev/null
    [ -f "$OUT/relay-$x.pid" ] && kill "$(cat "$OUT/relay-$x.pid")" 2> /dev/null
    rm -f "$OUT/relay-$x.pid"
  done
  RELAY_L='' RELAY_R=''
}

quiet() {  # quiet NOTE: the rig's state into OUT/quiet.txt; fails when a llama process runs or a card is not free
  {
    echo "# $(date -Is) $1"
    rigsh "FREE_MIB=$FREE_MIB bash -s" << 'EOF'
n=$(ps -eo comm | grep -c '^llama-')
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
echo "load $(cut -d' ' -f1-3 /proc/loadavg); $(grep '^MemAvailable:' /proc/meminfo | tr -s ' '); swap used $(awk '/^SwapTotal/ {t = $2} /^SwapFree/ {f = $2} END {print int((t - f) / 1024)}' /proc/meminfo) MiB; llama processes $n; fullest card $m MiB"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.gr --format=csv,noheader
[ "$n" = 0 ] && [ "$m" -lt "$FREE_MIB" ]
EOF
  } >> "$OUT/quiet.txt" 2>&1
}

server_start() {  # server_start LABEL ARM: llama-server on the rig, then /health through the whole relay from the client
  local v=ARGS_$2 args
  args="$PRESET_ARGS -md $MODEL_DRAFT${!v:+ ${!v}}"
  say "starting $1 on the rig: $PRESET_ENV $BIN $args -m $MODEL"
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -Is)" "$PRESET_ENV $BIN $args -m $MODEL" >> "$OUT/loads.tsv"
  RUNNING=$1
  rigsh "LOGDIR=$RIG_LOGDIR/$1 MIN_AVAIL_MB=$MIN_AVAIL_MB ${WD_ENV:-} ACTION=start ARM=$1 BIN=$BIN MODEL=$MODEL \
    PORT=$RIG_PORT SERVER_ENV='$PRESET_ENV' SERVER_ARGS='$args' bash -s" < "$RIG_SERVER" 2>&1 | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  for _ in $(seq 40); do
    cl "curl -sf 127.0.0.1:$CLIENT_PORT/health" > /dev/null 2>&1 && { say "the client reaches $1 through the relay"; return 0; }
    sleep 3
  done
  say "the client does not reach the server through the relay (relay.log)"
  return 1
}

server_end() {  # server_end LABEL: the server and its monitors stopped; the load's logs copied here and to the client,
  # then removed on the rig
  rigsh "LOGDIR=$RIG_LOGDIR/$1 ACTION=end PORT=$RIG_PORT bash -s" < "$RIG_SERVER" 2>&1 | tee -a "$OUT/run.log"
  rigsh "pkill -9 -x llama-server && echo 'a llama-server was still up: killed'" 2>&1 | tee -a "$OUT/run.log"
  RUNNING=''
  rigsh "test -d $RIG_LOGDIR/$1" || return 0
  mkdir -p "$OUT/rig/$1"
  rigsh "tar -C $RIG_LOGDIR/$1 -cf - ." | tar -C "$OUT/rig/$1" -xf - && rigsh "rm -rf $RIG_LOGDIR/$1"
  tar -C "$OUT/rig" -cf - "$1" | cl "mkdir -p $CLIENT_WORK/rig && tar -C $CLIENT_WORK/rig -xf -"
}

warmup() {  # warmup LABEL: one untimed request through the relay
  [ -n "$WARMUP" ] || return 0
  cl "curl -s -m 900 http://127.0.0.1:$CLIENT_PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d @$CLIENT_DIR/$WARMUP > $CLIENT_WORK/warmup-$1.json; rc=\$?; echo \"warm-up request of $1: curl exit \$rc, \
\$(wc -c < $CLIENT_WORK/warmup-$1.json) bytes\"; [ \$rc = 0 ] && grep -q '\"timings\"' $CLIENT_WORK/warmup-$1.json" \
    2>&1 | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ]
}

client_replay() {  # client_replay LABEL TRACES FILE: replay.py detached on the client, followed to its exit code
  local label=$1 rc
  timeout 60 ssh "${SSH_OPTS[@]}" "$CLIENT" "cd $CAC || exit 1; rm -f $CLIENT_WORK/$label.rc; setsid nohup bash -c \
    'nice -n 19 python3 -I replay.py run --log $LOG --traces $2 --server http://127.0.0.1:$CLIENT_PORT \
    --label $label --out $CLIENT_WORK/$3; echo \$? > $CLIENT_WORK/$label.rc' > $CLIENT_WORK/client-$label.txt 2>&1 \
    < /dev/null & echo \$! > $CLIENT_WORK/$label.pid; sleep 1"
  CLIENT_REPLAY=$label
  while true; do
    sleep 15
    rc=$(cl "cat $CLIENT_WORK/$label.rc 2> /dev/null || { kill -0 \$(cat $CLIENT_WORK/$label.pid) 2> /dev/null \
      && echo running; } || echo gone" 2> /dev/null) || continue
    case $rc in
      running) ;;
      gone) say "the replay of $label ended without an exit code"; CLIENT_REPLAY=''; return 1;;
      *) CLIENT_REPLAY=''; say "the replay of $label ended with $rc: $(cl "tail -n 1 $CLIENT_WORK/client-$label.txt")"
         [ "$rc" = 0 ]; return;;
    esac
  done
}

check_rows() {  # check_rows FILE N: the load is whole (N rows, none with an error, the draft ran)
  cl "python3 -I - $CLIENT_WORK/$1 $2" 2>&1 << 'EOF' | tee -a "$OUT/run.log"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [r for r in rows if r.get("error") or r.get("status") != 200 or not r.get("predicted_n")]
drafted = sum(r.get("draft_n") or 0 for r in rows)  # the server reports draft_n only when a request drafted
print(f"{len(rows)} rows, {len(bad)} with an error or without a reply, {drafted} drafted tokens")
sys.exit(0 if len(rows) == int(sys.argv[2]) and not bad and drafted > 0 else 1)
EOF
  [ "${PIPESTATUS[0]}" = 0 ]
}

stop_all() {  # whatever still runs: the client's replay, the rig's server and monitors, the relay
  if [ -n "${CLIENT_REPLAY:-}" ]; then
    cl "kill -- -\$(cat $CLIENT_WORK/$CLIENT_REPLAY.pid) 2> /dev/null" 2> /dev/null
    CLIENT_REPLAY=''
  fi
  [ -n "${RUNNING:-}" ] && server_end "$RUNNING" > /dev/null 2>&1
  relay_stop
}

fail() {
  say "FAILED: $*"
  stop_all
  touch "$OUT/failed"; rm -f "$OUT/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}
trap 'stop_all; [ -e "$OUT/done" ] || touch "$OUT/failed"; rm -f "$OUT/chain.pid"' EXIT
trap 'exit 130' INT TERM

if has prep; then
  say "prep: replay.py, common.py${WARMUP:+ and the warm-up request} to $CLIENT:$CLIENT_DIR"
  # shellcheck disable=SC2086
  tar -C "$REPO" -cf - bench/agent_check/replay.py bench/agent_check/common.py $WARMUP \
    | cl "mkdir -p $CLIENT_DIR $CLIENT_WORK && tar -C $CLIENT_DIR -xf -" || fail "the copy to the client"
fi

if has run; then
  # the recorded arm's traces in the order they ran, its number of requests, and the first trace's
  T=$(cl "python3 -I - $LOG '$TAG'" << 'EOF'
import json, re, sys
seen, n = {}, 0
for l in open(sys.argv[1]):
    if not l.strip():
        continue
    m = re.match(r"^(.+/a\d+/t\d+)/\d+$", json.loads(l).get("tag") or "")
    if m and re.search(sys.argv[2], m.group(1)):
        n += 1
        seen[m.group(1)] = seen.get(m.group(1), 0) + 1
print(n, next(iter(seen.values()), 0), ",".join(seen))
EOF
  ) || fail "could not read $LOG on the client"
  read -r NREQ WARM_NREQ TRACES <<< "$T"
  [ "${NREQ:-0}" -gt 0 ] || fail "no request tagged $TAG in $LOG"
  WARM_TRACES=${TRACES%%,*}
  say "$NREQ requests tagged $TAG; the warm-up load replays $WARM_TRACES ($WARM_NREQ requests)"
  relay_start
  declare -A nth
  for arm in $LOADS; do
    nth[$arm]=$((${nth[$arm]:-0} + 1))
    label=$arm-${nth[$arm]}
    [ -e "$OUT/load-$label.done" ] && { say "LOAD $label: done before"; continue; }
    case $arm in
      warm) traces=$WARM_TRACES; n=$WARM_NREQ; file=$label.jsonl;;
      *) traces=$TRACES; n=$NREQ; file=replay-$label.jsonl;;
    esac
    quiet "before $label" || fail "the rig is not quiet before $label (quiet.txt)"
    say "LOAD $label"
    cl "rm -f $CLIENT_WORK/$file"
    server_start "$label" "$arm" || fail "the server of $label did not start"
    warmup "$label" || fail "the warm-up request of $label"
    say "LOAD $label: replay start"
    client_replay "$label" "$traces" "$file" || fail "the replay of $label"
    say "LOAD $label: replay end"
    server_end "$label"
    check_rows "$file" "$n" || fail "the replay of $label is not whole"
    touch "$OUT/load-$label.done"
    say "LOAD $label: done"
  done
  quiet "after the last load" || fail "the rig is not clean after the last load (quiet.txt)"
  relay_stop
fi

if has collect; then
  mkdir -p "$OUT/client"
  cl "tar -C $CLIENT_WORK --exclude=./rig -cf - ." | tar -C "$OUT/client" -xf - || fail "the copy from the client"
  say "collect: $CLIENT:$CLIENT_WORK -> $OUT/client"
fi
touch "$OUT/done"
rm -f "$OUT/chain.pid"
echo "CHAIN_DONE $(date -Is)"
