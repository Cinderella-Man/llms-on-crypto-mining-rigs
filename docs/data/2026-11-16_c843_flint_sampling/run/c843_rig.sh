#!/usr/bin/env bash
# c843_rig.sh: C843 (Q1488) on the rig, detached (c843.sh starts it). Whether v4's Flint preset changes what Flint
# samples: one load per arm of Flint's server line, each replaying the same recorded agent requests with a fixed seed
# per request at Flint's own sampling (c843_client.py), so that the arms' token streams can be compared token by token.
# Per load: a quiet check (no llama process, at most QUIET_MAX_MIB MiB on any card), the server with the RAM watchdog,
# the server's LLAMA_/GGML_ variables as it runs, the client, the server stopped. Needs bash, curl, python3, nvidia-smi.
# Env (paths on the rig): WORK (this run's folder: run/ holds the scripts and requests.jsonl), BIN_V3 (v3.0's
# llama-server), BIN_V4 (v4-rc2's), MODEL (Flint Pascal Q8 GGUF), N_PREDICT (2048), SEEDS ("11 12 13"), REPEAT
# ("A1:11 B1:11": requests sent again at the end of each load), ARMS ("v3 v4 v4n"), PORT (8297), QUIET_MAX_MIB (100).
# The arms' lines come from the release files beside this script (v3.env, v4.env, the public release/flint-v3 and
# release/flint-v4 settings.env): v3 = v3.0's preset (no switch) on BIN_V3; v4 = the v4 preset (its three switches) on
# BIN_V4; v4n = the v4 preset without LLAMA_SAMPLING_TOP_K_ROWS on BIN_V4. Every line adds what the release agent check
# adds (--jinja --reasoning-format deepseek --metrics --no-webui); the script stops if the two files' lines differ.
# Output in WORK/out: server-<arm>.log, environ-<arm>.txt, quiet-<arm>.txt, fit-<arm>.txt, ram_watchdog-<arm>.tsv,
# <arm>.jsonl (one line per request: token ids, reply, end, counts), loads.tsv, then done (or failed).
set -uo pipefail
: "${WORK:?}" "${BIN_V3:?}" "${BIN_V4:?}" "${MODEL:?}"
R=$WORK/run
O=$WORK/out
N_PREDICT=${N_PREDICT:-2048}
SEEDS=${SEEDS:-11 12 13}
REPEAT=${REPEAT:-A1:11 B1:11}
ARMS=${ARMS:-v3 v4 v4n}
PORT=${PORT:-8297}
QUIET_MAX_MIB=${QUIET_MAX_MIB:-100}
mkdir -p "$O"
echo $$ > "$O/driver.pid"
say() { echo "$(date +%H:%M:%S) $*"; }
fail() { say "C843_FAILED: $*"; echo "$*" > "$O/failed"; exit 1; }
SRV='' WD=''
stop_server() {
  if [ -n "$SRV" ]; then
    kill "$SRV" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
    kill -9 "$SRV" 2> /dev/null; wait "$SRV" 2> /dev/null; SRV=''
  fi
  if [ -n "$WD" ]; then
    kill "$WD" 2> /dev/null; sleep 1; kill -9 "$WD" 2> /dev/null; WD=''
    [ -f /dev/shm/c843-wd.tsv ] && mv /dev/shm/c843-wd.tsv "$O/ram_watchdog-$ARM.tsv"
  fi
}
trap 'stop_server; rm -f "$O/driver.pid"' EXIT
trap 'exit 130' INT TERM

PRESET_ARGS='' PRESET_ENV=''
. "$R/v3.env"; V3_ARGS=$PRESET_ARGS; V3_ENV=$PRESET_ENV
. "$R/v4.env"; V4_ARGS=$PRESET_ARGS; V4_ENV=$PRESET_ENV
[ "$V3_ARGS" = "$V4_ARGS" ] || fail "release/flint-v4's line is not release/flint-v3's"
[ -z "$V3_ENV" ] || fail "release/flint-v3 sets variables: $V3_ENV"
# PRESET_ARGS' own --alias and sampling stay; the agent check adds the chat template, the reasoning format, metrics
ADD="--jinja --reasoning-format deepseek --metrics --no-webui"
arm_line() {  # arm_line ARM: A_BIN, A_ENV
  case $1 in
    v3) A_BIN=$BIN_V3; A_ENV=$V3_ENV ;;
    v4) A_BIN=$BIN_V4; A_ENV=$V4_ENV ;;
    v4n) A_BIN=$BIN_V4; A_ENV=$(for w in $V4_ENV; do [ "${w%%=*}" = LLAMA_SAMPLING_TOP_K_ROWS ] || echo "$w"; done | xargs) ;;
    *) fail "unknown arm $1" ;;
  esac
}
printf 'arm\tstarted\tload_s\tbin\tenv\tline\n' > "$O/loads.tsv"
say "C843: model $(basename "$MODEL"); arms $ARMS; seeds $SEEDS; $N_PREDICT tokens at most; repeats $REPEAT"
for ARM in $ARMS; do
  arm_line "$ARM"
  waited=0
  while ps -eo comm | grep -q -x -E 'llama-server|llama-bench|llama-perplexity|llama-cli' || \
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v m="$QUIET_MAX_MIB" '$1 + 0 > m { b = 1 } END { exit !b }'; do
    [ "$waited" -ge 60 ] && fail "the rig is not idle before arm $ARM"
    sleep 5; waited=$((waited + 5))
  done
  { date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-perplexity|llama-cli' || echo "no llama process"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$O/quiet-$ARM.txt" 2>&1
  log=$O/server-$ARM.log
  started=$(date -Is)
  echo "=== $started $ARM: env $A_ENV $A_BIN -m $MODEL $V3_ARGS $ADD --host 127.0.0.1 --port $PORT" > "$log"
  say "ARM $ARM: env $A_ENV $(basename "$(dirname "$(dirname "$A_BIN")")")"
  # shellcheck disable=SC2086
  env $A_ENV "$A_BIN" -m "$MODEL" $V3_ARGS $ADD --host 127.0.0.1 --port "$PORT" >> "$log" 2>&1 < /dev/null &
  SRV=$!
  rm -f /dev/shm/c843-wd.tsv
  pre=''; chrt -f 10 true 2> /dev/null && pre="chrt -f 10"
  PID=$SRV LOG=/dev/shm/c843-wd.tsv SERVER_LOG=$log setsid nohup $pre bash "$R/ram_watchdog.sh" > "$O/watchdog-$ARM.out" 2>&1 < /dev/null &
  WD=$!
  sleep 3
  tr '\0' '\n' < "/proc/$SRV/environ" | grep -E '^(LLAMA|GGML)_' > "$O/environ-$ARM.txt" || echo "no LLAMA_/GGML_ variable" > "$O/environ-$ARM.txt"
  t0=$(date +%s)
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 "$SRV" 2> /dev/null || { tail -n 20 "$log"; fail "arm $ARM: the server exited while loading"; }
    [ $(($(date +%s) - t0)) -gt 1800 ] && fail "arm $ARM: load timeout"
    sleep 2
  done
  load_s=$(($(date +%s) - t0))
  { date -Is; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$O/fit-$ARM.txt" 2>&1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ARM" "$started" "$load_s" "$A_BIN" "$A_ENV" "$V3_ARGS $ADD" >> "$O/loads.tsv"
  say "server $ARM up after $load_s s"
  python3 -I "$R/c843_client.py" --port "$PORT" --arm "$ARM" --requests "$R/requests.jsonl" --seeds "$SEEDS" \
    --repeat "$REPEAT" --n-predict "$N_PREDICT" --out "$O/$ARM.jsonl" || { kill -0 "$SRV" 2> /dev/null || tail -n 20 "$log"; fail "arm $ARM: the client failed"; }
  stop_server
  say "ARM $ARM done"
done
{ date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-perplexity|llama-cli' || echo "no llama process"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; } > "$O/quiet-end.txt" 2>&1
touch "$O/done"
say "C843_DONE"
