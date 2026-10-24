# rig_lib.sh: shared steps of pilot.sh and full_run.sh, sourced on the laptop. The laptop runs pi, the proxy and the
# tasks' tests; the rig runs llama-server only. Environment (the machine-specific values; see the README):
#   RIG          ssh alias of the rig (default rig)
#   RIG_PORT     llama-server's port on the rig's loopback       TUNNEL_PORT  the laptop end of the ssh forward
#   PROXY_PORT   the logging proxy on the laptop (pi talks to it)
#   V1_BIN, V2_BIN, STOCK_BIN   llama-server binaries on the rig (stock only for its replay)
#   MODEL        UD-Q4_K_XL on the rig                           MODEL_K2H    the lead file (file replay only)
#   AC_HOME, AC_NODE_DIR, AC_PI_ROOT   the laptop's shared test environments, node and pi release (README "Setup")
#   RIG_LOGDIR   the rig's log folder for this run (default /tmp/agent-check; copied back and removed at the end)
# Server flags: every arm gets the same template, sampling, context, checkpoints and metrics (COMMON); the arms differ
# only in binary, environment and the flags of arm_config.
AC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RIG=${RIG:-rig}
RIG_LOGDIR=${RIG_LOGDIR:-/tmp/agent-check}
: "${RIG_PORT:?}" "${TUNNEL_PORT:?}" "${PROXY_PORT:?}" "${AC_HOME:?}" "${AC_NODE_DIR:?}" "${AC_PI_ROOT:?}"
COMMON="-ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
SB=(--sandbox bwrap --node-dir "$AC_NODE_DIR" --pi-root "$AC_PI_ROOT" --ac-node "$AC_HOME/node" --ac-py "$AC_HOME/py")

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }

arm_config() {  # sets SERVER_BIN, SERVER_ENV, SERVER_ARGS, SERVER_MODEL for an arm
  SERVER_MODEL=$MODEL; SERVER_ENV=""
  case $1 in
    v1-ud) SERVER_BIN=$V1_BIN; SERVER_ARGS="$COMMON -b 512 -ub 128";;
    v2-ud) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"; SERVER_ARGS="$COMMON -b 2048 -ub 512";;
    v2-ud-ub256) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"; SERVER_ARGS="$COMMON -b 2048 -ub 256";;
    v2-ud-v1flags) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"; SERVER_ARGS="$COMMON -b 512 -ub 128";;
    v2-ud-ckpt) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"
                SERVER_ARGS="$COMMON -b 2048 -ub 512 --ctx-checkpoints 6 --checkpoint-min-step 0";;
    v2-ud-t1) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"; SERVER_ARGS="$COMMON -b 2048 -ub 512 -t 1 -tb 1";;
    stock-ud) SERVER_BIN=$STOCK_BIN; SERVER_ARGS="$COMMON -b 512 -ub 128";;
    v2-k2h) SERVER_BIN=$V2_BIN; SERVER_ENV="LLAMA_NO_MOE_DOWN_RESCALE=tg"; SERVER_ARGS="$COMMON -b 2048 -ub 512"
            SERVER_MODEL=$MODEL_K2H;;
    *) echo "unknown arm $1"; return 1;;
  esac
  # --ctx-checkpoints given twice: llama.cpp takes the last one
  : "${SERVER_BIN:?binary for $1 not set}" "${SERVER_MODEL:?model for $1 not set}"
}

tunnel_start() {
  if curl -sf "127.0.0.1:$TUNNEL_PORT/health" > /dev/null 2>&1 || ss -ltn | grep -q "127.0.0.1:$TUNNEL_PORT "; then
    say "tunnel port $TUNNEL_PORT already open; using it"
    return
  fi
  (while true; do
     ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
       -L "127.0.0.1:$TUNNEL_PORT:127.0.0.1:$RIG_PORT" "$RIG" &
     echo $! > "$OUT/tunnel.pid"
     wait $!
     sleep 3
   done) > "$OUT/tunnel.log" 2>&1 &
  TUNNEL_LOOP=$!
  sleep 3
  say "tunnel 127.0.0.1:$TUNNEL_PORT -> rig 127.0.0.1:$RIG_PORT"
}

proxy_start() {
  python3 -I "$AC_DIR/proxy.py" --listen "127.0.0.1:$PROXY_PORT" --upstream "127.0.0.1:$TUNNEL_PORT" \
    --log "$OUT/proxy.jsonl" >> "$OUT/proxy.out" 2>&1 &
  PROXY_PID=$!
  sleep 1
  kill -0 $PROXY_PID || { say "proxy failed to start"; return 1; }
}

server_start() {  # server_start ARM
  arm_config "$1" || return 1
  say "starting $1 on the rig"
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=start ARM=$1 BIN=$SERVER_BIN MODEL=$SERVER_MODEL PORT=$RIG_PORT SERVER_ENV='$SERVER_ENV' \
    SERVER_ARGS='$SERVER_ARGS' bash -s" < "$AC_DIR/rig_server.sh" | tee -a "$OUT/run.log"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  until curl -sf "127.0.0.1:$TUNNEL_PORT/health" > /dev/null; do sleep 3; done
}

server_stop() {
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=stop PORT=$RIG_PORT bash -s" < "$AC_DIR/rig_server.sh" | tee -a "$OUT/run.log"
}

rig_end() {  # stop everything on the rig, copy its logs to OUT/rig/ and remove them there
  ssh "$RIG" "LOGDIR=$RIG_LOGDIR ACTION=end PORT=$RIG_PORT bash -s" < "$AC_DIR/rig_server.sh" | tee -a "$OUT/run.log"
  mkdir -p "$OUT/rig"
  ssh "$RIG" "tar -C $RIG_LOGDIR -cf - ." | tar -C "$OUT/rig" -xf - && ssh "$RIG" "rm -rf $RIG_LOGDIR"
  say "rig logs in $OUT/rig"
}

laptop_stop() {
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2> /dev/null
  if [ -n "${TUNNEL_LOOP:-}" ]; then
    kill "$TUNNEL_LOOP" 2> /dev/null
    [ -f "$OUT/tunnel.pid" ] && kill "$(cat "$OUT/tunnel.pid")" 2> /dev/null
    rm -f "$OUT/tunnel.pid"
  fi
}

run_arm() {  # run_arm ARM TASKS [run.py options]
  local arm=$1 tasks=$2
  shift 2
  python3 "$AC_DIR/run.py" --arm "$arm" --base-url "http://127.0.0.1:$PROXY_PORT/v1" --out "$OUT/res" \
    --tasks "$tasks" "${SB[@]}" "$@" 2>&1 | tee -a "$OUT/run.log"
}

replay_run() {  # replay_run LABEL TRACES [replay.py run options]; straight to the server, not through the proxy
  local label=$1 traces=$2
  shift 2
  python3 -I "$AC_DIR/replay.py" run --log "$OUT/proxy.jsonl" --traces "$traces" \
    --server "http://127.0.0.1:$TUNNEL_PORT" --label "$label" --out "$OUT/replay.jsonl" --erase-slot "$@" \
    2>&1 | tee -a "$OUT/run.log"
}

disk_check() {
  local free
  free=$(df -Pm "$OUT" | awk 'NR == 2 {print $4}')
  [ "$free" -gt 1024 ] || { echo "only $free MB free where $OUT is; stopping"; return 1; }
}
