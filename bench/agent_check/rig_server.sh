#!/usr/bin/env bash
# rig_server.sh: start and stop the agent check's llama-server on the rig, with its monitors. Runs on the rig, read
# from stdin over ssh (rig_lib.sh does this):
#   ssh rig "ACTION=start ARM=v2-ud BIN=… MODEL=… PORT=… SERVER_ENV='…' SERVER_ARGS='…' bash -s" < rig_server.sh
#   ssh rig "ACTION=stop PORT=… bash -s" < rig_server.sh       # the server only
#   ssh rig "ACTION=end PORT=… bash -s" < rig_server.sh        # the server and the monitors
# Everything goes to LOGDIR (default /tmp/agent-check): server-<arm>.log, vmstat.log, gpu.csv (temperature, power,
# clocks, memory and utilization per card every 10 s) and meminfo.log ("<epoch> <MemAvailable kB> <swap used kB>" every
# 2 s). A watchdog kills llama-server when MemAvailable drops below MIN_AVAIL_MB (default 400), as in
# scripts/server_bench.sh: the host has 4 GB of RAM. The laptop copies LOGDIR back and removes it after the run.
set -uo pipefail
ACTION=${ACTION:?ACTION=start|stop|end}
PORT=${PORT:?}
LOGDIR=${LOGDIR:-/tmp/agent-check}
if [ "$ACTION" = start ]; then
  mkdir -p "$LOGDIR" && cd "$LOGDIR" || exit 1
else
  cd "$LOGDIR" 2> /dev/null || exit 0  # nothing started, or already collected and removed
fi

stop_server() {
  [ -f watchdog.pid ] && kill "$(cat watchdog.pid)" 2> /dev/null
  if [ -f server.pid ]; then
    srv=$(cat server.pid)
    kill "$srv" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$srv" 2> /dev/null || break; sleep 1; done
    kill -9 "$srv" 2> /dev/null
    echo "server stopped"
  fi
  rm -f server.pid watchdog.pid
}

case $ACTION in
start)
  : "${BIN:?}" "${MODEL:?}" "${ARM:?}"
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "a server already answers on port $PORT"; exit 1; fi
  if [ ! -f monitors.pid ]; then
    setsid nohup vmstat -t 10 > vmstat.log 2>&1 < /dev/null &
    echo $! > monitors.pid
    setsid nohup nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
      --format=csv,noheader -l 10 > gpu.csv 2>&1 < /dev/null &
    echo $! >> monitors.pid
    setsid nohup bash -c 'while true; do
        echo "$(date +%s) $(awk "/^MemAvailable/ {a=\$2} /^SwapTotal/ {t=\$2} /^SwapFree/ {f=\$2} END {print a, t - f}" /proc/meminfo)"
        sleep 2; done' > meminfo.log 2>&1 < /dev/null &
    echo $! >> monitors.pid
  fi
  echo "=== $(date -Is) $ARM: $BIN ${SERVER_ENV:-} ${SERVER_ARGS:-} -m $MODEL" >> "server-$ARM.log"
  # shellcheck disable=SC2086
  env ${SERVER_ENV:-} setsid nohup "$BIN" ${SERVER_ARGS:-} -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
    >> "server-$ARM.log" 2>&1 < /dev/null &
  srv=$!
  echo $srv > server.pid
  setsid nohup bash -c "while kill -0 $srv 2> /dev/null; do
      a=\$(awk '/^MemAvailable/ {print int(\$2 / 1024)}' /proc/meminfo)
      if [ \"\$a\" -lt ${MIN_AVAIL_MB:-400} ]; then
        echo \"watchdog: MemAvailable \$a MiB, killing llama-server\" >> server-$ARM.log; kill -9 $srv; break
      fi
      sleep 0.5; done" > /dev/null 2>&1 < /dev/null &
  echo $! > watchdog.pid
  t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 $srv 2> /dev/null; then echo "server exited while loading:"; tail -20 "server-$ARM.log"; exit 1; fi
    if [ $(($(date +%s) - t0)) -gt "${LOAD_TIMEOUT_S:-1200}" ]; then echo "load timeout"; stop_server; exit 1; fi
    sleep 2
  done
  echo "server $ARM up after $(($(date +%s) - t0)) s"
  ;;
stop)
  stop_server
  ;;
end)
  stop_server
  [ -f monitors.pid ] && xargs -r kill < monitors.pid 2> /dev/null
  rm -f monitors.pid
  echo "monitors stopped"
  ;;
esac
