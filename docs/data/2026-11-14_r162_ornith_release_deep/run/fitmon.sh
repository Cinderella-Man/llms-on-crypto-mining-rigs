#!/usr/bin/env bash
# fitmon.sh: per-card memory and the host's MemAvailable every second, for a fit load; stopped with kill
D=${D:?D=<folder>}
mkdir -p "$D"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > "$D/gpu1s.csv" 2>&1 < /dev/null &
echo $! > "$D/smi.pid"
while true; do
  printf '%s %s\n' "$(date +%s)" "$(awk '/^MemAvailable|^CmaFree|^SwapFree/ { printf "%s %s ", $1, $2 }' /proc/meminfo)"
  sleep 1
done > "$D/mem1s.txt" 2>&1
