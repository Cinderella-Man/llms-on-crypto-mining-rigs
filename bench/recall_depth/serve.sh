#!/usr/bin/env bash
# serve.sh start|wait|stop: one llama-server on a release preset's line, on the rig, for recall_depth.py.
# Runs on the rig, from WORK, which holds settings.env (the preset), ram_watchdog.sh and this script (remote.sh put
# copies them there).
#   start  checks the rig is idle (no llama process, at most QUIET_MAX_MIB on every card: WORK/quiet_start.txt), then
#          starts the server detached, on the preset's PRESET_ENV and PRESET_ARGS unchanged, with the draft (DRAFT),
#          --host 127.0.0.1 --port PORT, its log in WORK/server.log, and the RAM watchdog on it (scripts/ram_watchdog.sh,
#          at real-time priority, its table on tmpfs)
#   wait   returns when /health answers (0) or the server has exited or LOAD_TIMEOUT_S passed (1); writes WORK/fit.txt
#          (memory on every card and of the host once loaded)
#   stop   stops the server and the watchdog; the watchdog's table to WORK/ram_watchdog.tsv; WORK/quiet_end.txt
# Env: WORK (default /tmp/recall-depth), BIN (llama-server, or its folder), MODEL (the model's first shard), DRAFT
# (the draft model; empty: none), PORT (8080), QUIET_MAX_MIB (100), LOAD_TIMEOUT_S (1200).
set -uo pipefail
STEP=${1:?usage: serve.sh start|wait|stop}
WORK=${WORK:-/tmp/recall-depth}
PORT=${PORT:-8080}
QUIET_MAX_MIB=${QUIET_MAX_MIB:-100}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-1200}
WD_TSV=/dev/shm/recall-depth-$PORT.tsv
cd "$WORK" || exit 1

quiet() {  # quiet FILE: llama processes, every card's memory, host memory; 0 when idle
  local gpu
  gpu=$(nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader,nounits 2>&1)
  { date -Is
    ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-perplexity|llama-cli|test-backend-ops' || echo "no llama process"
    echo "$gpu"
    grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$1" 2>&1
  grep -q "no llama process" "$1" && echo "$gpu" | awk -F', *' -v m="$QUIET_MAX_MIB" '$2 !~ /^[0-9]+$/ || $2 + 0 > m { bad = 1 } END { exit bad }'
}

case $STEP in
start)
  : "${BIN:?BIN=<llama-server>}" "${MODEL:?MODEL=<first shard>}"
  [ -d "$BIN" ] && BIN=$BIN/llama-server
  quiet quiet_start.txt || { cat quiet_start.txt; echo "the rig is not idle"; exit 1; }
  PRESET_ENV='' PRESET_ARGS=''
  # shellcheck disable=SC1091
  . ./settings.env
  read -ra ENVS <<< "$PRESET_ENV"
  read -ra LINE <<< "$PRESET_ARGS"
  [ -n "${DRAFT:-}" ] && LINE=(-md "$DRAFT" "${LINE[@]}")
  echo "=== $(date -Is) env ${ENVS[*]} $BIN -m $MODEL ${LINE[*]} --host 127.0.0.1 --port $PORT" > server.log
  # its own session, so that it outlives the ssh call; setsid and nohup exec, so $! is the server's pid
  setsid nohup env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT" >> server.log 2>&1 < /dev/null &
  echo $! > server.pid
  pre=''
  chrt -f 10 true 2> /dev/null && pre="chrt -f 10"
  rm -f "$WD_TSV"
  # shellcheck disable=SC2086
  PID=$(cat server.pid) LOG=$WD_TSV SERVER_LOG=$WORK/server.log setsid nohup $pre bash ./ram_watchdog.sh \
    > watchdog.out 2>&1 < /dev/null &
  echo $! > watchdog.pid
  echo "server $(cat server.pid), watchdog $(cat watchdog.pid)"
  ;;
wait)
  t0=$(date +%s)
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$(cat server.pid)" 2> /dev/null; then echo "the server exited while loading:"; tail -n 20 server.log; exit 1; fi
    if [ $(($(date +%s) - t0)) -gt "$LOAD_TIMEOUT_S" ]; then echo "load timeout after $LOAD_TIMEOUT_S s"; exit 1; fi
    sleep 5
  done
  { date -Is
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>&1
    grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > fit.txt 2>&1
  echo "server up ($(($(date +%s) - t0)) s in this wait)"; cat fit.txt
  ;;
stop)
  srv=$(cat server.pid 2> /dev/null)
  if [ -n "$srv" ]; then
    kill "$srv" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$srv" 2> /dev/null || break; sleep 1; done
    kill -9 "$srv" 2> /dev/null
  fi
  wd=$(cat watchdog.pid 2> /dev/null)
  if [ -n "$wd" ]; then
    kill "$wd" 2> /dev/null
    for _ in 1 2 3 4 5 6; do kill -0 "$wd" 2> /dev/null || break; sleep 0.5; done
    kill -9 "$wd" 2> /dev/null
  fi
  [ -f "$WD_TSV" ] && mv "$WD_TSV" ram_watchdog.tsv
  sleep 3
  if quiet quiet_end.txt; then echo "stopped; the rig is idle"; else cat quiet_end.txt; echo "not idle after the stop"; exit 1; fi
  ;;
*)
  echo "usage: serve.sh start|wait|stop"; exit 1
  ;;
esac
