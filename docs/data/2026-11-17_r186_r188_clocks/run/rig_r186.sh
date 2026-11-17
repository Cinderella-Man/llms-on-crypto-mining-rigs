#!/usr/bin/env bash
# rig_r186.sh (R186 Q1551, R188 Q1553): the rig side of run.sh. Runs on the rig as root from WORK, one ACTION per call:
#   snap TAG       host memory (/proc/meminfo), the largest resident processes, the NVIDIA kernel modules (lsmod),
#                  the NVIDIA lines of /proc/interrupts and each card's memory, P-state and clock
#   irq TAG        only the epoch and the NVIDIA lines of /proc/interrupts (read tight around one request)
#   start ARM      llama-server from BIN with SERVER_ENV and SERVER_ARGS on MODEL, on 127.0.0.1:PORT, guarded by
#                  scripts/ram_watchdog.sh (copied to WORK) at real-time priority; waits until /health answers
#   log_start TAG  nvidia-smi every SMI_MS ms for all cards: timestamp, index, P-state, SM and memory clock, power,
#                  the active clock-event (throttle) reasons and utilization, to WORK/smi-TAG.csv
#   log_stop TAG   stops that log
#   stop ARM       stops the watchdog, then the server
# Environment: WORK, BIN (the build's bin folder), MODEL, PORT (8186), SERVER_ENV, SERVER_ARGS, SMI_MS (100).
set -uo pipefail
: "${WORK:?}" "${ACTION:?}"
PORT=${PORT:-8186}; SMI_MS=${SMI_MS:-100}; TAG=${TAG:-x}; ARM=${ARM:-x}
cd "$WORK" || exit 1
export LC_ALL=C

case $ACTION in
snap)
  echo "## snap $TAG $(date +%s.%N)"
  grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|AnonPages|Shmem|Mlocked|Unevictable|Slab|SUnreclaim|KernelStack|PageTables|VmallocUsed|CmaTotal|CmaFree|SwapTotal|SwapFree):' /proc/meminfo
  echo "# largest resident processes (rss KiB)"; ps -eo pid,rss,comm --sort=-rss | head -n 9
  echo "# lsmod (size in bytes, users)"; lsmod | awk 'NR == 1 || /nvidia/'
  echo "# /proc/interrupts, NVIDIA lines"; grep -i nvidia /proc/interrupts
  echo "# cards"; nvidia-smi --query-gpu=index,memory.used,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader
  echo "# llama processes"; ps -eo comm | grep -x -E 'llama-server|llama-bench' || echo none
  ;;
irq)
  echo "## irq $TAG $(date +%s.%N)"; grep -i nvidia /proc/interrupts
  ;;
start)
  : "${BIN:?}" "${MODEL:?}"
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "a server already answers on port $PORT"; exit 1; fi
  echo "=== $(date -Is) $ARM: ${SERVER_ENV:-} $BIN/llama-server ${SERVER_ARGS:-} -m $MODEL" >> "server-$ARM.log"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  env ${SERVER_ENV:-} setsid nohup "$BIN/llama-server" ${SERVER_ARGS:-} -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
    >> "server-$ARM.log" 2>&1 < /dev/null &
  srv=$!; echo "$srv" > server.pid
  PID=$srv LOG=/dev/shm/r186-wd-$ARM.tsv SERVER_LOG=$WORK/server-$ARM.log MIN_AVAIL_MB=400 \
    setsid nohup chrt -f 10 bash "$WORK/ram_watchdog.sh" > "watchdog-$ARM.out" 2>&1 < /dev/null &
  echo $! > watchdog.pid
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 "$srv" 2> /dev/null || { echo "server exited while loading:"; tail -n 20 "server-$ARM.log"; exit 1; }
    sleep 2
  done
  echo "server $ARM up after $(($(date +%s) - t0)) s"
  ;;
log_start)
  setsid nohup nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active,utilization.gpu \
    --format=csv,noheader,nounits -lms "$SMI_MS" > "smi-$TAG.csv" 2> "smi-$TAG.err" < /dev/null &
  echo $! > "smi-$TAG.pid"; echo "log $TAG started $(date +%s.%N), every $SMI_MS ms"
  ;;
log_stop)
  kill "$(cat "smi-$TAG.pid")" 2> /dev/null; rm -f "smi-$TAG.pid"; echo "log $TAG stopped $(date +%s.%N)"
  ;;
stop)
  [ -f watchdog.pid ] && { kill "$(cat watchdog.pid)" 2> /dev/null; sleep 1; rm -f watchdog.pid; }
  if [ -f server.pid ]; then
    srv=$(cat server.pid); kill "$srv" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$srv" 2> /dev/null || break; sleep 1; done
    kill -9 "$srv" 2> /dev/null; rm -f server.pid
  fi
  [ -f "/dev/shm/r186-wd-$ARM.tsv" ] && mv "/dev/shm/r186-wd-$ARM.tsv" "watchdog-$ARM.tsv"
  echo "server $ARM stopped $(date +%s.%N)"
  ;;
*) echo "unknown ACTION $ACTION"; exit 2 ;;
esac
