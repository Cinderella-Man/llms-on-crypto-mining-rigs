#!/usr/bin/env bash
# mem_guard.sh: a host-memory guard for llama-server on a rig with 4 GB of RAM (Q558). Start it on the rig before the
# server, at real-time priority, so it still gets the CPU while the kernel reclaims (in Q226's draft smoke test the
# agent check's watchdog got no CPU for 8 minutes). Every second it
#   - gives each llama-server oom_score_adj 1000, so the kernel's OOM killer takes it before the desktop;
#   - logs host memory: MemAvailable counts CmaFree, which this kernel reports without a CMA area and which pinned
#     buffers cannot use (Q559, Q560), so the log carries MemAvailable minus CmaFree too;
#   - kills llama-server (SIGKILL) when more than KILL_SWAPOUT_MIB (default 600) MiB went out to swap in the last
#     WINDOW_S (default 10) seconds, or when memory pressure (/proc/pressure/memory, "full avg10") stays above
#     KILL_FULL_PCT (default 60) percent for KILL_FULL_S (default 15) seconds.
# The defaults sit between the no-draft agent check (at most 397 MiB of swap in 10 s, right after loading) and the
# draft line that stalled the host (687 MiB in 10 s, then 8 minutes of reclaim).
# Bash builtins only (no fork per sample). Log line, MiB unless named:
#   epoch avail cma avail-minus-cma swap-used swapout-in-window psi-some-avg10 psi-full-avg10 server-pid
# Usage, as root on the rig (the log on tmpfs, so writing it never waits on a disk):
#   LOG=/dev/shm/mem_guard.log setsid nohup chrt -f 10 bash mem_guard.sh > /dev/null 2>&1 < /dev/null &
#   kill "$(cat /dev/shm/mem_guard.log.pid)"    # to stop it
set -u
LOG=${LOG:?}
KILL_SWAPOUT_MIB=${KILL_SWAPOUT_MIB:-600}
WINDOW_S=${WINDOW_S:-10}
KILL_FULL_PCT=${KILL_FULL_PCT:-60}
KILL_FULL_S=${KILL_FULL_S:-15}
echo $$ > "$LOG.pid"
echo -1000 > /proc/$$/oom_score_adj
echo "# $(date -Is) mem_guard: swap-out > $KILL_SWAPOUT_MIB MiB in $WINDOW_S s, or full avg10 > $KILL_FULL_PCT % for $KILL_FULL_S s" >> "$LOG"
exec {tick}<> <(:)  # read -t on this fd is a sleep without a fork
declare -a ring
i=0 pid="" high=0
while true; do
  while read -r k v _; do
    case $k in
      MemAvailable:) avail=$((v / 1024)) ;;
      CmaFree:) cma=$((v / 1024)) ;;
      SwapTotal:) st=$v ;;
      SwapFree:) sf=$v ;;
    esac
  done < /proc/meminfo
  while read -r k v; do [ "$k" = pswpout ] && { out=$v; break; }; done < /proc/vmstat
  while read -r kind a10 _; do
    a10=${a10#avg10=}
    if [ "$kind" = some ]; then some=$a10; else full=$a10; fi
  done < /proc/pressure/memory
  slot=$((i % WINDOW_S))
  prev=${ring[$slot]:-$out}
  ring[$slot]=$out
  win=$(((out - prev) * 4 / 1024))
  if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
    pid=""
    for d in /proc/[0-9]*; do
      { read -r c < "$d/comm"; } 2> /dev/null && [ "$c" = llama-server ] && { pid=${d#/proc/}; break; }
    done
    [ -n "$pid" ] && echo 1000 > "/proc/$pid/oom_score_adj" && echo "# $(printf '%(%s)T') llama-server $pid: oom_score_adj 1000" >> "$LOG"
  fi
  if [ "${full%.*}" -gt "$KILL_FULL_PCT" ]; then high=$((high + 1)); else high=0; fi
  echo "$(printf '%(%s)T') $avail $cma $((avail - cma)) $(((st - sf) / 1024)) $win $some $full ${pid:--}" >> "$LOG"
  if [ -n "$pid" ]; then
    why=""
    [ "$win" -gt "$KILL_SWAPOUT_MIB" ] && why="swap-out $win MiB in $WINDOW_S s"
    [ "$high" -ge "$KILL_FULL_S" ] && why="full avg10 $full % for $high s"
    if [ -n "$why" ]; then
      kill -9 "$pid" 2> /dev/null
      echo "# $(printf '%(%s)T') KILL llama-server $pid: $why" >> "$LOG"
      pid=""
    fi
  fi
  i=$((i + 1))
  read -rt 1 -u "$tick"
done
