#!/usr/bin/env bash
# ram_watchdog_test.sh: self-test of scripts/ram_watchdog.sh and of the two scripts that embed it
# (bench/agent_check/rig_server.sh and scripts/server_bench.sh). It needs no GPU, no model and no llama-server, and it
# never touches a process it did not start. Run it on a machine that is not a rig (a rig has no spare second):
#   tar -C <repo> -cf - scripts/ram_watchdog.sh scripts/ram_watchdog_test.sh scripts/server_bench.sh \
#     bench/agent_check/rig_server.sh | ssh <machine> 'd=$(mktemp -d /tmp/rwt.XXXXXX); tar -xf - -C "$d"; \
#     REPO=$d bash $d/scripts/ram_watchdog_test.sh; rm -rf "$d"'
# Sections (SECTIONS, default all): unit, real, sync, rig_server, server_bench.
#   unit        the watchdog on synthetic /proc files (MEMINFO, VMSTAT, PSI) and a dummy process: each limit fires, a
#               brief excursion does not, the limits switch off, the TSV has nine columns and the right usable, the
#               guarded process gets oom_score_adj 1000, bad input is refused, the scan mode (PROC) kills only a
#               process of that name and keeps running
#   real        the watchdog on this machine's own /proc for REAL_S seconds (default 60; KEEP=<folder> keeps the TSV) with
#               every limit off: the rows must match /proc/meminfo, /proc/vmstat and /proc/pressure/memory; RT=1 (default) also runs a
#               few seconds under chrt -f 10 and checks that it recorded the real-time policy
#   sync        the copies embedded in rig_server.sh and server_bench.sh are byte for byte scripts/ram_watchdog.sh
#   rig_server  rig_server.sh start, a kill by the embedded watchdog, end: with a stand-in llama-server and nvidia-smi
#   server_bench  server_bench.sh PROMPTS=none with the stand-in: the watchdog runs, logs and is gone afterwards
# Prints one line per check and "ram_watchdog_test: N passed, M failed"; the exit status is 1 when anything failed.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
W=$REPO/scripts/ram_watchdog.sh
SECTIONS=${SECTIONS:-"unit real sync rig_server server_bench"}
REAL_S=${REAL_S:-60}
RT=${RT:-1}
T=$(mktemp -d /tmp/ram_watchdog_test.XXXXXX)
pass=0 fail=0
pids=()
cleanup() {
  for p in "${pids[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2> /dev/null; done
  rm -rf "$T"
}
trap cleanup EXIT
has() { [[ " $SECTIONS " == *" $1 "* ]]; }
ok() { pass=$((pass + 1)); echo "ok    $*"; }
bad() { fail=$((fail + 1)); echo "FAIL  $*"; }
check() { local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }
alive() { kill -0 "$1" 2> /dev/null; }
wait_gone() {  # wait_gone PID SECONDS: true when the process ended within the time
  local n=$(($2 * 4))
  while [ "$n" -gt 0 ]; do alive "$1" || return 0; sleep 0.25; n=$((n - 1)); done
  ! alive "$1"
}
# put DIR AVAIL_MIB CMA_MIB PSWPOUT_PAGES PSI_SOME PSI_FULL: the synthetic /proc files of one case, written atomically
put() {
  local d=$1 av=$(($2 * 1024)) cma=$(($3 * 1024))
  printf 'MemTotal:        3929088 kB\nMemFree:          %d kB\nMemAvailable:     %d kB\nShmem:             100000 kB\nSwapTotal:       3891200 kB\nSwapFree:        3591200 kB\nCmaTotal:                0 kB\nCmaFree:          %d kB\n' \
    $((av / 2)) "$av" "$cma" > "$d/meminfo.new" && mv "$d/meminfo.new" "$d/meminfo"
  printf 'pgfault 1\npswpin 0\npswpout %d\npgmajfault 3\n' "$4" > "$d/vmstat.new" && mv "$d/vmstat.new" "$d/vmstat"
  printf 'some avg10=%s avg60=0.00 avg300=0.00 total=0\nfull avg10=%s avg60=0.00 avg300=0.00 total=0\n' "$5" "$6" \
    > "$d/psi.new" && mv "$d/psi.new" "$d/psi"
}
# mk NAME AVAIL CMA PSWPOUT SOME FULL: a case folder with its inputs; sets C
mk() { C=$T/$1; mkdir -p "$C"; shift; put "$C" "$@"; }
# start_wd [VAR=value ...]: a dummy process and the watchdog guarding it, on the case's synthetic files; sets DUMMY and WD
start_wd() {
  sleep 600 &
  DUMMY=$!
  disown "$DUMMY"
  pids+=("$DUMMY")
  env MEMINFO="$C/meminfo" VMSTAT="$C/vmstat" PSI="$C/psi" PID=$DUMMY LOG="$C/log.tsv" SERVER_LOG="$C/server.log" \
    USABLE_FLOOR_S=2 PSI_FULL_S=3 PSI_SOME_S=3 WINDOW_S=3 "$@" bash "$W" > "$C/out" 2>&1 &
  WD=$!
  disown "$WD"
  pids+=("$WD")
}
stop_wd() { alive "$WD" && kill -TERM "$WD" 2> /dev/null; wait_gone "$WD" 3; kill -9 "$DUMMY" 2> /dev/null; }
killed() { wait_gone "$DUMMY" "${1:-8}" && grep -q '^# [0-9]* KILL ' "$C/log.tsv"; }
survives() { sleep "${1:-5}"; alive "$DUMMY" && alive "$WD" && ! grep -q 'KILL' "$C/log.tsv"; }
reason() { grep -q "^# [0-9]* KILL .*$1" "$C/log.tsv"; }
rows_ok() {  # nine tab-separated columns on every sample row, at least N rows, usable = avail - cma
  awk -F'\t' -v n="$1" '!/^#/ { r++; if (NF != 9) b++; if ($4 != $2 - $3) b++ } END { exit !(r >= n && !b) }' "$C/log.tsv"
}

if has unit; then
  echo "== unit: synthetic inputs"
  # a quiet host: no kill, rows well formed, oom_score_adj, the events, a clean stop
  mk quiet 1200 400 1000 1.00 0.50
  start_wd
  check "quiet: nothing killed in 6 s" survives 6
  check "quiet: nine tab-separated columns, usable = MemAvailable - CmaFree" rows_ok 5
  check "quiet: usable is 800" awk -F'\t' '!/^#/ && $4 == 800 { f = 1 } END { exit !f }' "$C/log.tsv"
  check "quiet: the guarded process has oom_score_adj 1000" test "$(cat /proc/$DUMMY/oom_score_adj)" = 1000
  check "quiet: the watchdog has oom_score_adj -1000" test "$(cat /proc/$WD/oom_score_adj)" = -1000
  check "quiet: events name the limits, oom_score_adj and the scheduling" \
    bash -c "grep -q 'ram_watchdog: usable < 20 MiB' '$C/log.tsv' && grep -q 'oom_score_adj 1000 (set 1000), SCHED_' '$C/log.tsv'"
  check "quiet: the header names the columns" grep -q '^#epoch	avail_mib	cma_free_mib	usable_mib' "$C/log.tsv"
  stop_wd
  check "quiet: SIGTERM is logged as stopped" grep -q '^# [0-9]* stopped' "$C/log.tsv"

  # the floor on MemAvailable minus CmaFree
  mk floor 949 967 1000 0.00 0.00  # Q226's reading: MemAvailable 949 MiB, CmaFree 967 MiB
  start_wd
  check "floor: S12's reading (949 - 967 = -18 MiB) kills the process" killed 8
  check "floor: the reason names MemAvailable minus CmaFree" reason 'MemAvailable minus CmaFree -18 MiB'
  check "floor: the watchdog ends with the process" wait_gone "$WD" 3
  check "floor: the server's log gets the watchdog line and the numbers" \
    bash -c "grep -q '^watchdog: MemAvailable minus CmaFree -18' '$C/server.log' && grep -q 'CmaFree: ' '$C/server.log' && grep -q 'psi some' '$C/server.log'"
  mk floor_ok 100 70 1000 0.00 0.00  # usable 30 MiB, above the floor of 20
  start_wd
  check "floor: usable 30 MiB is above the default floor" survives 5
  stop_wd
  mk floor_off 100 100 1000 0.00 0.00
  start_wd MIN_USABLE_MB=off
  check "floor: off disables it" survives 5
  stop_wd
  mk floor_blip 1200 400 1000 0.00 0.00
  start_wd
  sleep 2
  put "$C" 100 100 1000 0.00 0.00
  sleep 0.7
  put "$C" 1200 400 1000 0.00 0.00
  check "floor: under a second below it is not enough (USABLE_FLOOR_S 2)" survives 4
  stop_wd

  # the legacy floor on MemAvailable alone
  mk legacy 100 0 1000 0.00 0.00
  start_wd MIN_AVAIL_MB=150
  check "legacy: MemAvailable 100 MiB below 150 kills (usable is 100, above the floor)" killed 6
  check "legacy: the reason says so" reason 'MemAvailable 100 MiB, below 150'
  mk legacy_off 100 0 1000 0.00 0.00
  start_wd
  check "legacy: MIN_AVAIL_MB defaults to off" survives 4
  stop_wd

  # memory pressure
  mk full 1200 400 1000 20.00 45.00
  start_wd
  check "psi: full avg10 45 % for 3 s kills" killed 8
  check "psi: the reason names full avg10" reason 'memory pressure full avg10 45.00 %, above 40 % for 3 s'
  mk some 1200 400 1000 45.00 5.00
  start_wd
  check "psi: some avg10 45 % for 3 s kills" killed 8
  check "psi: the reason names some avg10" reason 'memory pressure some avg10 45.00 %'
  mk psi_low 1200 400 1000 39.00 39.00
  start_wd
  check "psi: 39 % does not" survives 6
  stop_wd
  mk psi_blip 1200 400 1000 0.00 0.00
  start_wd PSI_FULL_S=4 PSI_SOME_S=4
  sleep 1
  put "$C" 1200 400 1000 50.00 50.00
  sleep 2.2
  put "$C" 1200 400 1000 0.00 0.00
  check "psi: a 2 s excursion is not 4 s in a row" survives 4
  stop_wd
  mk psi_off 1200 400 1000 90.00 90.00
  start_wd PSI_FULL_PCT=off PSI_SOME_PCT=off
  check "psi: off disables both" survives 5
  stop_wd
  mk psi_decimal 1200 400 1000 0.00 40.01
  start_wd
  check "psi: 40.01 % is above 40 (the compare keeps the decimals)" killed 8

  # the swap-out rate
  mk swap 1200 400 0 0.00 0.00
  start_wd
  ( n=0; while sleep 0.5; do n=$((n + 100000)); put "$C" 1200 400 "$n" 0.00 0.00; done ) &  # 781 MiB a second
  feeder=$!
  disown "$feeder"
  pids+=("$feeder")
  check "swap-out: 781 MiB a second kills" killed 10
  check "swap-out: the reason names the rate" reason 'swap-out [0-9]* MiB in 3 s, above 600'
  kill "$feeder" 2> /dev/null
  mk swap_slow 1200 400 0 0.00 0.00
  start_wd
  ( n=0; while sleep 0.5; do n=$((n + 5000)); put "$C" 1200 400 "$n" 0.00 0.00; done ) &  # 39 MiB a second
  feeder=$!
  disown "$feeder"
  pids+=("$feeder")
  check "swap-out: 39 MiB a second does not" survives 7
  kill "$feeder" 2> /dev/null
  stop_wd

  # input checks
  check "input: a limit that is not an integer is refused" bash -c "! LOG=$T/bad.tsv PSI_FULL_PCT=abc bash '$W' 2> /dev/null"
  check "input: a process that does not exist is refused" bash -c "! LOG=$T/bad.tsv PID=999999 bash '$W' 2> /dev/null"
  check "input: no LOG is refused" bash -c "! env -u LOG bash '$W' 2> /dev/null"

  # the scan mode: kills the process named PROC and nothing else, and keeps running
  mk scan 100 100 1000 0.00 0.00
  name=rwt$$
  cp /bin/bash "$T/$name"
  "$T/$name" -c 'exec {f}<> <(:); read -t 600 -u $f' &
  victim=$!
  disown "$victim"
  pids+=("$victim")
  sleep 600 &
  bystander=$!
  disown "$bystander"
  pids+=("$bystander")
  env MEMINFO="$C/meminfo" VMSTAT="$C/vmstat" PSI="$C/psi" PROC="$name" LOG="$C/log.tsv" USABLE_FLOOR_S=2 bash "$W" \
    > "$C/out" 2>&1 &
  WD=$!
  pids+=("$WD")
  DUMMY=$victim
  check "scan: the process named PROC is found and killed" killed 8
  check "scan: a process of another name is untouched" alive "$bystander"
  check "scan: the watchdog keeps running afterwards" alive "$WD"
  kill -9 "$bystander" 2> /dev/null
  stop_wd
fi

if has real; then
  echo "== real: this machine's own /proc for ${REAL_S} s, every limit off"
  C=$T/real
  mkdir -p "$C"
  sleep $((REAL_S + 30)) &
  DUMMY=$!
  disown "$DUMMY"
  pids+=("$DUMMY")
  env PID=$DUMMY LOG="$C/log.tsv" MIN_USABLE_MB=off PSI_FULL_PCT=off PSI_SOME_PCT=off KILL_SWAPOUT_MIB=off bash "$W" \
    > "$C/out" 2>&1 &
  WD=$!
  pids+=("$WD")
  sleep "$REAL_S"
  alive "$WD" && ok "real: still running after ${REAL_S} s, nothing killed" || bad "real: ended early"
  # what /proc says now, for a comparison with the last rows
  read -r a < <(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)
  read -r cmaf < <(awk '/^CmaFree:/ { print int($2 / 1024) }' /proc/meminfo)
  read -r swp < <(awk '/^SwapTotal:/ { t = $2 } /^SwapFree:/ { f = $2 } END { print int((t - f) / 1024) }' /proc/meminfo)
  stop_wd
  [ -n "${KEEP:-}" ] && cp "$C/log.tsv" "$KEEP/real_${REAL_S}s.tsv"  # KEEP=<folder>: the sample TSV, for the data folder
  rows=$(grep -vc '^#' "$C/log.tsv")
  check "real: about one row a second ($rows rows in ${REAL_S} s)" test "$rows" -ge $((REAL_S - 3)) -a "$rows" -le $((REAL_S + 3))
  check "real: nine tab-separated columns, usable = avail - cma" rows_ok $((REAL_S - 3))
  check "real: epochs rise by about a second" awk -F'\t' '!/^#/ { if (p && ($1 - p < 0 || $1 - p > 3)) b++; p = $1 } END { exit b > 0 }' "$C/log.tsv"
  check "real: MemAvailable (now $a MiB) is within 25 % + 200 MiB of the last row" \
    awk -F'\t' -v a="$a" '!/^#/ { l = $2 } END { d = l - a; if (d < 0) d = -d; exit !(d <= a / 4 + 200) }' "$C/log.tsv"
  check "real: CmaFree (now $cmaf MiB) is within 200 MiB of the last row" \
    awk -F'\t' -v c="$cmaf" '!/^#/ { l = $3 } END { d = l - c; if (d < 0) d = -d; exit !(d <= 200) }' "$C/log.tsv"
  check "real: swap used (now $swp MiB) is within 200 MiB of the last row" \
    awk -F'\t' -v s="$swp" '!/^#/ { l = $5 } END { d = l - s; if (d < 0) d = -d; exit !(d <= 200) }' "$C/log.tsv"
  check "real: swap-out and pressure columns are numbers (pressure from /proc/pressure/memory)" \
    awk -F'\t' '!/^#/ && ($6 !~ /^-?[0-9]+$/ || $7 !~ /^[0-9]+\.[0-9][0-9]$/ || $8 !~ /^[0-9]+\.[0-9][0-9]$/) { b++ } END { exit b > 0 }' "$C/log.tsv"
  check "real: the pid column holds the guarded pid" awk -F'\t' -v p="$DUMMY" '!/^#/ && $9 != p { b++ } END { exit b > 0 }' "$C/log.tsv"
  if [ "$RT" = 1 ] && command -v chrt > /dev/null; then
    C=$T/real_rt
    mkdir -p "$C"
    sleep 20 &
    DUMMY=$!
    disown "$DUMMY"
    pids+=("$DUMMY")
    env PID=$DUMMY LOG="$C/log.tsv" MIN_USABLE_MB=off PSI_FULL_PCT=off PSI_SOME_PCT=off KILL_SWAPOUT_MIB=off \
      chrt -f 10 bash "$W" > "$C/out" 2>&1 &
    WD=$!
    pids+=("$WD")
    sleep 4
    check "real: under chrt -f 10 it records SCHED_FIFO rt 10 for itself and SCHED_OTHER for the guarded process" \
      bash -c "grep -q 'watchdog pid [0-9]*: .*SCHED_FIFO rt 10' '$C/log.tsv' && grep -q 'oom_score_adj 1000 (set 1000), SCHED_OTHER' '$C/log.tsv'"
    stop_wd
  fi
fi

if has sync; then
  echo "== sync: the embedded copies"
  for f in bench/agent_check/rig_server.sh scripts/server_bench.sh; do
    if [ -f "$REPO/$f" ]; then
      sed -n "/<< 'RAM_WATCHDOG'\$/,/^RAM_WATCHDOG\$/p" "$REPO/$f" | sed '1d;$d' > "$T/embedded.sh"
      check "$f embeds scripts/ram_watchdog.sh unchanged" cmp -s "$T/embedded.sh" "$W"
    else
      bad "$f is not in $REPO"
    fi
  done
fi

# stand-ins for llama-server (answers /health on --port with 200) and nvidia-smi (no output), in $T/stubbin
make_stubs() {
  mkdir -p "$T/stubbin"
  cat > "$T/stubbin/llama-server" << 'STUB'
#!/usr/bin/env python3
import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
STUB
  printf '#!/bin/sh\ncase " $*" in *" -l "*) sleep 3600 ;; esac\n' > "$T/stubbin/nvidia-smi"  # silent; the monitor (-l 10) keeps running
  chmod +x "$T/stubbin/llama-server" "$T/stubbin/nvidia-smi"
}
free_port() { echo $((20000 + RANDOM % 20000)); }

if has rig_server; then
  echo "== rig_server: start, a kill by the embedded watchdog, end (stand-ins for the server and nvidia-smi)"
  make_stubs
  mk rs 1200 400 1000 0.00 0.00
  port=$(free_port)
  L=$T/rs/logs
  out=$(env PATH="$T/stubbin:$PATH" LOGDIR="$L" PORT=$port MEMINFO="$C/meminfo" VMSTAT="$C/vmstat" PSI="$C/psi" \
    ACTION=start ARM=t BIN="$T/stubbin/llama-server" MODEL=/dev/null MIN_AVAIL_MB=400 SERVER_ARGS="" \
    bash "$REPO/bench/agent_check/rig_server.sh" < /dev/null 2>&1)
  rc=$?
  check "rig_server: start returns 0 ($(echo "$out" | tail -n 1))" test "$rc" = 0
  srv=$(cat "$L/server.pid" 2> /dev/null)
  wd=$(cat "$L/watchdog.pid" 2> /dev/null)
  pids+=("$srv" "$wd")
  check "rig_server: the server and the watchdog run" bash -c "kill -0 '$srv' && kill -0 '$wd'"
  check "rig_server: the server has oom_score_adj 1000" test "$(cat /proc/$srv/oom_score_adj 2> /dev/null)" = 1000
  check "rig_server: the watchdog has oom_score_adj -1000" test "$(cat /proc/$wd/oom_score_adj 2> /dev/null)" = -1000
  sleep 3
  put "$C" 500 500 1000 0.00 0.00  # usable 0, MemAvailable 500 (above the old floor of 400)
  check "rig_server: usable 0 MiB kills the server" wait_gone "$srv" 10
  check "rig_server: the watchdog ends with the server" wait_gone "$wd" 5
  check "rig_server: the server log has the watchdog line" grep -q '^watchdog: .*MemAvailable minus CmaFree 0 MiB' "$L/server-t.log"
  env PATH="$T/stubbin:$PATH" LOGDIR="$L" PORT=$port ACTION=end bash "$REPO/bench/agent_check/rig_server.sh" \
    < /dev/null > "$T/rs/end.out" 2>&1
  check "rig_server: end copies the TSV into LOGDIR with the KILL event" grep -q '^# [0-9]* KILL llama-server' "$L/ram_watchdog-t.tsv"
  check "rig_server: the TSV has rows" bash -c "grep -vc '^#' '$L/ram_watchdog-t.tsv' | awk '{ exit !(\$1 >= 3) }'"
  check "rig_server: the watchdog is gone and /dev/shm is clean" bash -c "! kill -0 '$wd' 2> /dev/null && ! ls /dev/shm/ram_watchdog-$port.* 2> /dev/null | grep -q ."
  check "rig_server: the monitors are gone" bash -c "! ps -eo pid,args | grep -F '$L' | grep -v grep | grep -q ."
  # a start whose watchdog cannot run refuses to leave the server unguarded (a bad limit makes the watchdog exit)
  mk rs2 1200 400 1000 0.00 0.00
  L2=$T/rs2/logs
  port2=$(free_port)
  out=$(env PATH="$T/stubbin:$PATH" LOGDIR="$L2" PORT=$port2 MEMINFO="$C/meminfo" VMSTAT="$C/vmstat" PSI="$C/psi" \
    PSI_FULL_PCT=abc ACTION=start ARM=t BIN="$T/stubbin/llama-server" MODEL=/dev/null MIN_AVAIL_MB=400 SERVER_ARGS="" \
    bash "$REPO/bench/agent_check/rig_server.sh" < /dev/null 2>&1)
  rc=$?
  check "rig_server: a watchdog that cannot start fails the start (rc $rc)" test "$rc" != 0
  check "rig_server: and no server is left running" bash -c "! ps -eo args | grep -F -- \"--port $port2\" | grep -v grep | grep -q ."
  env PATH="$T/stubbin:$PATH" LOGDIR="$L2" PORT=$port2 ACTION=end bash "$REPO/bench/agent_check/rig_server.sh" < /dev/null > /dev/null 2>&1
fi

if has server_bench; then
  echo "== server_bench: PROMPTS=none with the stand-in server"
  make_stubs
  mk sb 1200 400 1000 0.00 0.00
  tag=wdtest-$$
  port=$(free_port)
  out=$(env PATH="$T/stubbin:$PATH" BIN="$T/stubbin" MODEL=/dev/null TAG=$tag PROMPTS=none PORT=$port SERVER_ARGS="" \
    MEMINFO="$C/meminfo" VMSTAT="$C/vmstat" PSI="$C/psi" bash "$REPO/scripts/server_bench.sh" < /dev/null 2>&1)
  rc=$?
  check "server_bench: returns 0" test "$rc" = 0
  O=/tmp/server_bench/$tag
  check "server_bench: the TSV is in the output folder, with rows and no KILL" \
    bash -c "[ \$(grep -vc '^#' '$O/ram_watchdog.tsv') -ge 1 ] && ! grep -q KILL '$O/ram_watchdog.tsv'"
  check "server_bench: the watchdog and the server are gone, /dev/shm is clean" \
    bash -c "! ps -eo args | grep -F 'ram_watchdog-$tag' | grep -v grep | grep -q . && ! ls /dev/shm/ram_watchdog-$tag.* 2> /dev/null | grep -q ."
  check "server_bench: no stand-in server is left" bash -c "! ps -eo args | grep -F -- '--port $port' | grep -v grep | grep -q ."
  rm -rf "$O"
  rmdir /tmp/server_bench 2> /dev/null  # the folder server_bench.sh makes, if nothing else is in it
fi

echo "ram_watchdog_test: $pass passed, $fail failed"
[ "$fail" = 0 ]
