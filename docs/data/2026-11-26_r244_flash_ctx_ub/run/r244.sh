#!/usr/bin/env bash
# R244 (Q1924): which of the 131,072-token line's two prompt effects (R243, docs/data/2026-11-26_r243_flash_131k_agent_rig1/)
# comes from the context size (-c) and which from the micro-batch (-ub). One load of one arm of Flash-Next's v5 preset
# (release/flash-next-v5/settings.env as it stands), nothing else changed:
#   ARM=p512  the preset (-c 65536 -ub 512)
#   ARM=u256  the same with -ub 256
#   ARM=l131  the same with -c 131072 -ub 256 (R124's and R243's line)
# The server is started by the agent check's own rig script (bench/agent_check/rig_server.sh: its card log, host log
# and RAM watchdog, MIN_AVAIL_MB=60 as R156's driver passes), with the preset's PRESET_ENV and its PRESET_ARGS in the
# order the agent check's harness writes them (R243's started lines: --metrics --no-webui after the sampling, -md after
# -lm dio), -c and -ub replaced word for word, each found exactly once. Once the server is up, the running process's
# own command line and environment are read back from /proc and written as started_command.txt (the binary's
# environment, then the binary and its arguments without the -m, --host and --port the rig script adds); the script
# stops if they differ from the line it meant to start.
# Then bench/agent_check/replay.py sends R243's recorded requests (LOG, R243's proxy.jsonl with its .bodies.gz) to the
# server through an ssh relay, one at a time, each with its recorded reply length and seed:
#   first   the first request of R243's runs (bugfix-python-affine-cipher's first turn, 2,780 tokens) on the fresh
#           server's empty cache
#   erased  the same request again with "cache_prompt": false added to its body (run/no_cache_log.py writes it as a
#           one-request log), so the server clears the slot's cache and reads all 2,780 tokens again: an empty cache
#           later in a load. The slot-erase endpoint answers 501 without --slot-save-path, which the preset lacks.
#   small   the four first turns that read under 600 new tokens on R243's cached prefix of 1,549 tokens (javascript
#           triangle, python list-ops, javascript grade-school, javascript variable-length-quantity, R243's run order)
#   mid     two first turns of 600 to 1,400 new tokens on the same prefix (javascript phone-number, 610 tokens, and
#           python scale-generator, 1,086; two different exercises, so the second does not reuse the first's task text)
# Rows go to OUT/LOAD/replay.jsonl, labelled LOAD-first, LOAD-erased, LOAD-small and LOAD-mid; replay_window.txt keeps
# the replay's start and end (this machine's local time, for the card log), and the rig's logs (server log, cards.csv,
# meminfo.log, the watchdog's TSV) come back to OUT/LOAD/rig and are removed on the rig.
# Environment: ARM, LOAD (the load's name, e.g. p512a), OUT, RIG (ssh alias of the rig), BIN (llama-server on the rig),
# MODEL and DRAFT (paths on the rig), RIG_PORT (the server's port on the rig's loopback), LOCAL_PORT (this machine's end
# of the relay), LOG, RIGDIR (a folder on the rig for this load's logs; removed after), PRESET (default
# release/flash-next-v5/settings.env). DRY=1 writes OUT/LOAD/dry_line.txt (the line it would start) and stops, without
# touching the rig. Run from the repository root.
set -uo pipefail
: "${ARM:?}" "${LOAD:?}" "${OUT:?}" "${BIN:?}" "${MODEL:?}" "${DRAFT:?}"
PRESET=${PRESET:-release/flash-next-v5/settings.env}
BENCH=bench/agent_check
FIRST=v4r/bugfix-python-affine-cipher/a1/t1
SMALL=v4r/bugfix-javascript-triangle/a1/t1,v4r/bugfix-python-list-ops/a1/t1,v4r/bugfix-javascript-grade-school/a1/t1,v4r/bugfix-javascript-variable-length-quantity/a1/t1
MID=v4r/bugfix-javascript-phone-number/a1/t1,v4r/bugfix-python-scale-generator/a1/t1
NOCACHE=nocache/bugfix-python-affine-cipher/a1/t1
O=$OUT/$LOAD
mkdir -p "$O"
say() { echo "$(date '+%F %T') $LOAD: $*" | tee -a "$O/run.log"; }
count() { local s=$1 n=0; while [[ $s == *"$2"* ]]; do s=${s#*"$2"}; n=$((n + 1)); done; echo $n; }

# shellcheck disable=SC1090
. "$PRESET"
case $ARM in
  p512) ctx=65536; ub=512 ;;
  u256) ctx=65536; ub=256 ;;
  l131) ctx=131072; ub=256 ;;
  *) echo "unknown ARM $ARM (p512, u256 or l131)"; exit 1 ;;
esac
a=" $PRESET_ARGS "
for w in " --min-p 0 " " -lm dio " " -c 65536 " " -ub 512 "; do
  [ "$(count "$a" "$w")" = 1 ] || { say "'${w# }' not found exactly once in PRESET_ARGS"; exit 1; }
done
a=${a/" --min-p 0 "/" --min-p 0 --metrics --no-webui "}
a=${a/" -lm dio "/" -lm dio -md $DRAFT "}
a=${a/" -c 65536 "/" -c $ctx "}
a=${a/" -ub 512 "/" -ub $ub "}
SERVER_ARGS=${a# }; SERVER_ARGS=${SERVER_ARGS% }
SERVER_ENV=$PRESET_ENV
LINE="$SERVER_ENV $BIN $SERVER_ARGS"
printf '%s\n' "$LINE" > "$O/dry_line.txt"
say "arm $ARM: -c $ctx -ub $ub; line $(md5sum < "$O/dry_line.txt" | cut -c1-32), $(wc -c < "$O/dry_line.txt") bytes"
[ "${DRY:-0}" = 1 ] && exit 0

: "${RIG:?}" "${RIG_PORT:?}" "${LOCAL_PORT:?}" "${LOG:?}" "${RIGDIR:?}"
rig() { timeout "${T:-120}" ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$RIG" "$@"; }
RELAY=
finish() {
  [ -n "$RELAY" ] && kill "$RELAY" 2> /dev/null
  T=180 rig "LOGDIR=$RIGDIR ACTION=end PORT=$RIG_PORT bash -s" < "$BENCH/rig_server.sh" 2>&1 | tee -a "$O/run.log"
  mkdir -p "$O/rig"
  if T=300 rig "test -d $RIGDIR && tar -C $RIGDIR -cf - ." | tar -C "$O/rig" -xf -; then
    rig "rm -rf $RIGDIR" && say "rig logs in $O/rig; $RIGDIR removed on the rig"
  fi
  rig "ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
    printf 'Xid lines: '; dmesg | grep -c 'NVRM: Xid'" > "$O/end_state.txt" 2>&1
  say "end state: $(tr '\n' ' ' < "$O/end_state.txt")"
}

# the rig must be quiet: no llama process, nothing on any card
st=$(rig "ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -u | tr '\n' ' '") || { say "rig unreachable"; exit 1; }
say "start state: $st"
[[ $st == "no llama process"*" " ]] && [ "$(echo "${st#no llama process}" | tr -d ' \n')" = 0 ] \
  || { say "the rig is not quiet; nothing started"; exit 1; }
if timeout 3 bash -c 'exec 3<> "/dev/tcp/127.0.0.1/$0"' "$LOCAL_PORT" 2> /dev/null; then
  say "local port $LOCAL_PORT is taken; nothing started"; exit 1
fi

trap 'say "interrupted"; finish; exit 1' INT TERM
say "starting on the rig"
T=1500 rig "LOGDIR=$RIGDIR MIN_AVAIL_MB=60 ACTION=start ARM=$LOAD BIN=$BIN MODEL=$MODEL PORT=$RIG_PORT \
  SERVER_ENV='$SERVER_ENV' SERVER_ARGS='$SERVER_ARGS' bash -s" < "$BENCH/rig_server.sh" 2>&1 | tee -a "$O/run.log"
[ "${PIPESTATUS[0]}" = 0 ] || { say "the server did not come up"; finish; exit 1; }

# the started command, from the running process
pid=$(rig "ps -eo pid,comm | awk '\$2 == \"llama-server\" {print \$1}'")
[ "$(wc -w <<< "$pid")" = 1 ] || { say "not one llama-server process: '$pid'"; finish; exit 1; }
argv=$(rig "tr '\0' '\n' < /proc/$pid/cmdline") || { say "no command line for $pid"; finish; exit 1; }
argv=$(tr '\n' ' ' <<< "$argv"); argv=${argv% }
want="$BIN $SERVER_ARGS -m $MODEL --host 127.0.0.1 --port $RIG_PORT"
[ "$argv" = "$want" ] || { say "the process's arguments differ from the line"; diff <(tr ' ' '\n' <<< "$want") <(tr ' ' '\n' <<< "$argv") | tee -a "$O/run.log"; finish; exit 1; }
penv=$(rig "tr '\0' '\n' < /proc/$pid/environ")
env_started=
for kv in $SERVER_ENV; do
  v=$(grep -m1 "^${kv%%=*}=" <<< "$penv") || { say "${kv%%=*} not in the process's environment"; finish; exit 1; }
  env_started="$env_started $v"
done
env_started=${env_started# }
[ "$env_started" = "$SERVER_ENV" ] || { say "the process's environment differs from PRESET_ENV"; finish; exit 1; }
printf '%s\n' "$env_started ${argv% -m $MODEL --host 127.0.0.1 --port $RIG_PORT}" > "$O/started_command.txt"
cmp -s "$O/started_command.txt" "$O/dry_line.txt" || { say "started command differs from the dry line"; finish; exit 1; }
say "started command (pid $pid) equals the dry line: $(md5sum < "$O/started_command.txt" | cut -c1-32)"

ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$RIG_PORT" "$RIG" \
  > "$O/relay.log" 2>&1 < /dev/null &
RELAY=$!
for _ in $(seq 30); do curl -sf "127.0.0.1:$LOCAL_PORT/health" > /dev/null 2>&1 && break; sleep 1; done
curl -sf "127.0.0.1:$LOCAL_PORT/health" > /dev/null 2>&1 || { say "the relay does not reach the server"; finish; exit 1; }

rp() {  # rp LABEL LOG REPLAY-ARGS...: one replay.py call, its console appended
  local label=$1 log=$2; shift 2
  say "replay $LOAD-$label"
  timeout 900 python3 -I "$BENCH/replay.py" run --log "$log" --server "http://127.0.0.1:$LOCAL_PORT" \
    --label "$LOAD-$label" --out "$O/replay.jsonl" "$@" >> "$O/replay.console" 2>&1
  local rc=$?
  [ $rc = 0 ] || say "replay $LOAD-$label exit $rc"
  return $rc
}
python3 -I "$(dirname "$0")/no_cache_log.py" "$LOG" "$FIRST" "$O/nocache.jsonl" >> "$O/run.log" 2>&1 \
  || { say "no_cache_log.py failed"; finish; exit 1; }
date '+%F %T' > "$O/replay_window.txt"
ok=1
rp first "$LOG" --traces "$FIRST" --requests 0 || ok=0
rp erased "$O/nocache.jsonl" --traces "$NOCACHE" --requests 0 || ok=0
rp small "$LOG" --traces "$SMALL" --requests 0 || ok=0
rp mid "$LOG" --traces "$MID" --requests 0 || ok=0
date '+%F %T' >> "$O/replay_window.txt"
grep -q "^$LOAD-erased .*: prompt 2780 (cache 0) " "$O/replay.console" || { say "the erased request did not read 2,780 tokens on an empty cache"; ok=0; }
grep -q 'ERROR' "$O/replay.console" && { say "a request failed"; ok=0; }
say "replay rows: $(wc -l < "$O/replay.jsonl" 2> /dev/null || echo 0)"
finish
[ $ok = 1 ] && echo ok > "$O/done" && say "done"
[ $ok = 1 ]
