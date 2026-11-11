#!/usr/bin/env bash
# lv4_loads.sh: one more untimed load of each arm of a finished deep-lane run, with -lv 4, for the library's buffer lines
# per card (llama-server's default verbosity, 3, hides the library's INFO lines, among them "compute buffer size").
# Each load takes exactly the server line and environment of a load of the run (the "===" line of its collected
# server-<load>.log), adds -lv 4, and runs through bench/agent_check/rig_server.sh (the RAM watchdog with it); 20 s after
# it answers, the cards' memory; then it stops. No request is sent. Run from the repository root on a machine that only
# runs ssh, after the run has ended.
#   RIG=<ssh alias> LANE=<the collected deep-lane OUT> RIGDIR=<a folder on the rig> OUT=<a local folder> \
#     LOADS="v4c1 rows1" bash lv4_loads.sh
# Writes OUT/<load>lv/: console.txt, server-<load>lv.log, mem.csv (index, MiB used, MiB total), meminfo.txt, buffers.txt
# (the buffer and KV lines of the log).
set -uo pipefail
: "${RIG:?}" "${LANE:?}" "${RIGDIR:?}" "${OUT:?}"
LOADS=${LOADS:-v4c1 rows1}
PORT=${PORT:-8097}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-60}
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$OUT"
for load in $LOADS; do
  line=$(grep -m 1 '^=== ' "$LANE/server-$load.log") || { say "no server line in $LANE/server-$load.log"; exit 1; }
  rest=${line#*: env }
  read -ra w <<< "$rest"
  envs=() args=() bin='' model='' i=0
  while [ "$i" -lt "${#w[@]}" ] && [[ ${w[i]} == *=* ]]; do envs+=("${w[i]}"); i=$((i + 1)); done
  bin=${w[i]}; i=$((i + 1))
  while [ "$i" -lt "${#w[@]}" ]; do
    case ${w[i]} in
      -m) model=${w[i + 1]}; i=$((i + 2)) ;;
      --host|--port) i=$((i + 2)) ;;
      *) args+=("${w[i]}"); i=$((i + 1)) ;;
    esac
  done
  name=${load}lv d=$RIGDIR/load-$name o=$OUT/$name up=0
  mkdir -p "$o"
  say "load $name: ${envs[*]} $bin -m $model ${args[*]} -lv 4"
  ssh "$RIG" "LOGDIR=$d MIN_AVAIL_MB=$MIN_AVAIL_MB ACTION=start ARM=$name BIN=$bin MODEL=$model PORT=$PORT \
      SERVER_ENV='${envs[*]}' SERVER_ARGS='${args[*]} -lv 4' bash -s" < bench/agent_check/rig_server.sh > "$o/console.txt" 2>&1 && up=1
  if [ $up = 1 ]; then
    sleep 20
    ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem.csv"
    ssh "$RIG" "grep -E '^(MemTotal|MemAvailable|CmaFree|SwapTotal|SwapFree):' /proc/meminfo" > "$o/meminfo.txt"
  fi
  ssh "$RIG" "LOGDIR=$d ACTION=end PORT=$PORT bash -s" < bench/agent_check/rig_server.sh >> "$o/console.txt" 2>&1
  ssh "$RIG" "tar -C $d -cf - ." | tar -C "$o" -xf - && ssh "$RIG" "rm -rf $d"
  grep -aE 'buffer size|KV self size|h_nextn|projected|failed|error' "$o/server-$name.log" > "$o/buffers.txt"
  say "load $name: up $up; $(grep -ac 'compute buffer size' "$o/buffers.txt") compute buffer lines"
  [ $up = 1 ] || exit 1
done
ssh "$RIG" "rmdir $RIGDIR 2> /dev/null; ps -eo comm | grep -x llama-server || echo 'no llama-server'"
