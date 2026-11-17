#!/usr/bin/env bash
# rig_server.sh: start and stop the agent check's llama-server on the rig, with its monitors. Runs on the rig, read
# from stdin over ssh (rig_lib.sh does this):
#   ssh rig "ACTION=start ARM=v2-ud BIN=… MODEL=… PORT=… SERVER_ENV='…' SERVER_ARGS='…' bash -s" < rig_server.sh
#   ssh rig "ACTION=stop PORT=… bash -s" < rig_server.sh       # the server only
#   ssh rig "ACTION=end PORT=… bash -s" < rig_server.sh        # the server and the monitors
# Everything goes to LOGDIR (default /tmp/agent-check): server-<arm>.log, vmstat.log, cards.csv (the card log,
# scripts/card_log.py embedded below: each card's P-state, SM and memory clock, power, temperature, memory used,
# utilization and throttle reasons every CARD_LOG_S seconds, default 0.5, read-only through NVML, one process for the
# whole run; scripts/card_table.py summarises it), meminfo.log ("<epoch> <MemAvailable kB> <swap used kB>" every
# 2 s) and ram_watchdog-<arm>.tsv (the watchdog below, one row a second). The host has 4 GB of RAM, and MemAvailable alone
# does not show when it is out (Q226): a RAM watchdog, scripts/ram_watchdog.sh embedded below, starts with the server at
# real-time priority (chrt -f 10), gives it oom_score_adj 1000 and kills it on MemAvailable minus CmaFree below
# MIN_USABLE_MB (20), on memory pressure (/proc/pressure/memory, some or full avg10 above 40 % for 10 s), on more than
# 600 MiB swapped out in 10 s, or on MemAvailable below MIN_AVAIL_MB (default 400, the old floor). Every limit is an
# environment variable of the watchdog (MIN_USABLE_MB, USABLE_FLOOR_S, PSI_FULL_PCT, PSI_FULL_S, PSI_SOME_PCT, PSI_SOME_S,
# KILL_SWAPOUT_MIB, WINDOW_S; "off" disables one), given on the ssh command line like MIN_AVAIL_MB; the start fails
# when the watchdog does not come up. The numbers behind each default: docs/data/2026-11-04_ram_watchdog/. The laptop
# copies LOGDIR back and removes it after the run.
set -uo pipefail
ACTION=${ACTION:?ACTION=start|stop|end}
PORT=${PORT:?}
LOGDIR=${LOGDIR:-/tmp/agent-check}
if [ "$ACTION" = start ]; then
  mkdir -p "$LOGDIR" && cd "$LOGDIR" || exit 1
else
  cd "$LOGDIR" 2> /dev/null || exit 0  # nothing started, or already collected and removed
fi

write_watchdog() {  # scripts/ram_watchdog.sh, byte for byte (scripts/ram_watchdog_test.sh checks it)
  cat << 'RAM_WATCHDOG'
#!/usr/bin/env bash
# ram_watchdog.sh: stops llama-server before a 4 GB host runs out of memory (Q559; from Q558's mem_guard.sh). A host that
# swaps itself to a stop does not look out of memory to MemAvailable alone: on the rigs' kernels 410-970 MiB of the 3.8 GiB
# are CMA-marked pageblocks that MemAvailable counts and that pinned buffers and kernel allocations cannot use, so it read
# ~949 MiB while the host was already out of memory (Q226). Every second this script samples
#   usable   MemAvailable minus CmaFree: what pinned buffers and the kernel's own allocations can take
#   swap-out pages written to swap in the last WINDOW_S samples (/proc/vmstat pswpout), in MiB
#   pressure /proc/pressure/memory, "some" and "full" avg10, in percent
# and kills the guarded llama-server (SIGKILL) when one of these holds, every limit an environment variable:
#   MIN_USABLE_MB  (20)   usable below it for USABLE_FLOOR_S (2) samples in a row
#   PSI_FULL_PCT   (40)   full avg10 above it for PSI_FULL_S (10) samples in a row
#   PSI_SOME_PCT   (40)   some avg10 above it for PSI_SOME_S (10) samples in a row
#   KILL_SWAPOUT_MIB (600) more than that written to swap within WINDOW_S (10) seconds
#   MIN_AVAIL_MB   (0)    MemAvailable alone below it (the agent check's old floor; 0 = off, as the floor above is the
#                         one that counts CMA; bench/agent_check/rig_server.sh and scripts/server_bench.sh pass 400)
# A limit set to "off" is disabled. The defaults sit between what runs that finished reached and what the runs that failed
# reached (docs/data/2026-11-04_ram_watchdog/thresholds.tsv has each number's source). It also
#   - gives the guarded process oom_score_adj OOM_SCORE_ADJ (1000), so the kernel's OOM killer takes it before the desktop
#     or the watchdog (which gets -1000), and reads the value back;
#   - records the scheduling policy of itself and of the guarded process: start it under a real-time policy
#     (chrt -f 10) so that it still gets a CPU while the kernel reclaims (in Q226 the old watchdog's loop got none for
#     8 minutes);
#   - uses bash builtins only in its loop (no fork per sample).
# Which process: PID=<pid> guards that process and exits when it ends; without PID it guards every process whose
# /proc/<pid>/comm is PROC (llama-server; llama-bench for Q583) and keeps running until it is killed.
# Log: LOG (a TSV, put it on tmpfs so that writing it never waits on a disk), one row a second, tab separated, the first
# nine columns those of mem_guard.sh:
#   epoch  avail_mib  cma_free_mib  usable_mib  swap_used_mib  swapout_window_mib  psi_some_avg10  psi_full_avg10  pid
# The first line names them; lines starting with # are events (the limits, the process found, its oom_score_adj and
# scheduling, KILL and its reason, the stop): read it with comment='#'. After a kill the numbers behind it are also
# appended to SERVER_LOG (if set) behind a line starting "watchdog:", as the agent check's logs have always had.
# MEMINFO, VMSTAT and PSI name other files to read from (the self-test feeds synthetic numbers).
#   PID=$srv LOG=/dev/shm/ram_watchdog.tsv SERVER_LOG=server.log setsid nohup chrt -f 10 bash ram_watchdog.sh &
#   LOG=/dev/shm/ram_watchdog.tsv setsid nohup chrt -f 10 bash ram_watchdog.sh &    # every llama-server, until killed
# Stop it with SIGTERM (it writes "stopped"). Run as root (oom_score_adj and SIGKILL of the server's user).
set -u
LOG=${LOG:?LOG=<the TSV to write, e.g. /dev/shm/ram_watchdog.tsv>}
PID=${PID:-}
PROC=${PROC:-llama-server}
SERVER_LOG=${SERVER_LOG:-}
MIN_USABLE_MB=${MIN_USABLE_MB:-20}
USABLE_FLOOR_S=${USABLE_FLOOR_S:-2}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-0}
PSI_FULL_PCT=${PSI_FULL_PCT:-40}
PSI_FULL_S=${PSI_FULL_S:-10}
PSI_SOME_PCT=${PSI_SOME_PCT:-40}
PSI_SOME_S=${PSI_SOME_S:-10}
KILL_SWAPOUT_MIB=${KILL_SWAPOUT_MIB:-600}
WINDOW_S=${WINDOW_S:-10}
OOM_SCORE_ADJ=${OOM_SCORE_ADJ:-1000}
MEMINFO=${MEMINFO:-/proc/meminfo}
VMSTAT=${VMSTAT:-/proc/vmstat}
PSI=${PSI:-/proc/pressure/memory}

for v in MIN_USABLE_MB MIN_AVAIL_MB; do [ "${!v}" = off ] && printf -v "$v" %s -1000000; done
for v in PSI_FULL_PCT PSI_SOME_PCT KILL_SWAPOUT_MIB; do [ "${!v}" = off ] && printf -v "$v" %s 1000000; done
for v in MIN_USABLE_MB USABLE_FLOOR_S MIN_AVAIL_MB PSI_FULL_PCT PSI_FULL_S PSI_SOME_PCT PSI_SOME_S KILL_SWAPOUT_MIB \
  WINDOW_S OOM_SCORE_ADJ; do
  [[ ${!v} =~ ^-?[0-9]+$ ]] || { echo "ram_watchdog: $v must be an integer or off, not '${!v}'" >&2; exit 2; }
done
[ "$WINDOW_S" -ge 1 ] || { echo "ram_watchdog: WINDOW_S must be at least 1" >&2; exit 2; }
if [ -n "$PID" ]; then
  [[ $PID =~ ^[0-9]+$ ]] && kill -0 "$PID" 2> /dev/null || { echo "ram_watchdog: no process '$PID'" >&2; exit 2; }
fi
exec {lf}>> "$LOG" || { echo "ram_watchdog: cannot write $LOG" >&2; exit 1; }
exec {tick}<> <(:)  # read -t on this fd is a sleep without a fork
page=$(getconf PAGESIZE 2> /dev/null || echo 4096)
have_psi=0
[ -r "$PSI" ] && have_psi=1

now=0
ev() { printf '# %s %s\n' "$now" "$*" >&"$lf"; }
sched_of() {  # sched_of PID: SCHED = "SCHED_FIFO rt 10", "SCHED_OTHER nice 0", ...
  local s f
  s=$(< "/proc/$1/stat") 2> /dev/null || { SCHED=unknown; return; }
  s=${s##*) }
  read -ra f <<< "$s"
  case ${f[38]:-x} in
    0) SCHED="SCHED_OTHER nice ${f[16]}" ;;
    1) SCHED="SCHED_FIFO rt ${f[37]}" ;;
    2) SCHED="SCHED_RR rt ${f[37]}" ;;
    3) SCHED="SCHED_BATCH nice ${f[16]}" ;;
    5) SCHED="SCHED_IDLE" ;;
    6) SCHED="SCHED_DEADLINE" ;;
    *) SCHED="unknown" ;;
  esac
}
find_pid() {  # the first process whose comm is PROC, in pid
  local d c
  pid=""
  for d in /proc/[0-9]*; do
    { read -r c < "$d/comm"; } 2> /dev/null && [ "$c" = "$PROC" ] && { pid=${d#/proc/}; return; }
  done
}
dump() {  # dump REASON: the numbers behind a kill, appended to the server's log
  [ -n "$SERVER_LOG" ] || return 0
  {
    echo "watchdog: $1, killing $PROC"
    while read -r k v u; do
      case $k in MemAvailable: | MemFree: | CmaFree: | Shmem: | Mlocked: | Unevictable: | SwapFree:) echo "  $k $v $u" ;; esac
    done < "$MEMINFO"
    [ "$have_psi" = 1 ] && while read -r l; do echo "  psi $l"; done < "$PSI"
  } >> "$SERVER_LOG" 2> /dev/null
}
trap 'ev stopped; exit 0' TERM INT

printf -v now '%(%s)T' -1
printf '#epoch\tavail_mib\tcma_free_mib\tusable_mib\tswap_used_mib\tswapout_window_mib\tpsi_some_avg10\tpsi_full_avg10\tpid\n' >&"$lf"
printf -v when '%(%Y-%m-%dT%H:%M:%S%z)T' -1
target="every $PROC"
[ -n "$PID" ] && target="pid $PID"
ev "$when ram_watchdog: usable < $MIN_USABLE_MB MiB for $USABLE_FLOOR_S s; MemAvailable < $MIN_AVAIL_MB MiB; psi full > $PSI_FULL_PCT % for $PSI_FULL_S s, some > $PSI_SOME_PCT % for $PSI_SOME_S s; swap-out > $KILL_SWAPOUT_MIB MiB in $WINDOW_S s; guards $target"
echo -1000 > /proc/$$/oom_score_adj 2> /dev/null || ev "could not set the watchdog's own oom_score_adj"
sched_of $$
ev "watchdog pid $$: oom_score_adj $({ read -r x < /proc/$$/oom_score_adj; } 2> /dev/null && echo "$x"), $SCHED"
[ "$have_psi" = 1 ] || ev "no $PSI: the pressure limits are off"

declare -a ring
i=0 nu=0 nf=0 ns=0 adj_pid=""
pid=$PID
while :; do
  printf -v now '%(%s)T' -1
  if [ -n "$PID" ]; then
    kill -0 "$pid" 2> /dev/null || { ev "$PROC $pid ended"; break; }
  elif [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
    find_pid
  fi
  if [ -n "$pid" ] && [ "$pid" != "$adj_pid" ]; then
    adj_pid=$pid
    echo "$OOM_SCORE_ADJ" > "/proc/$pid/oom_score_adj" 2> /dev/null
    back=$({ read -r x < "/proc/$pid/oom_score_adj"; } 2> /dev/null && echo "$x")
    sched_of "$pid"
    ev "$PROC $pid: oom_score_adj ${back:-unreadable} (set $OOM_SCORE_ADJ), $SCHED"
  fi
  avail="" cma=0 st=0 sf=0 out="" some=0.00 full=0.00
  while read -r k v _; do
    case $k in
      MemAvailable:) avail=$((v / 1024)) ;;
      CmaFree:) cma=$((v / 1024)) ;;
      SwapTotal:) st=$v ;;
      SwapFree:) sf=$v ;;
    esac
  done < "$MEMINFO"
  while read -r k v; do [ "$k" = pswpout ] && { out=$v; break; }; done < "$VMSTAT"
  if [ "$have_psi" = 1 ]; then
    while read -r kind a10 _; do
      a10=${a10#avg10=}
      if [ "$kind" = some ]; then some=$a10; else full=$a10; fi
    done < "$PSI"
  fi
  if [ -z "$avail" ] || [ -z "$out" ]; then  # a sample that could not be read is not a number to act on
    ev "unreadable sample (MemAvailable '${avail}', pswpout '${out}')"
    read -rt 1 -u "$tick"
    continue
  fi
  usable=$((avail - cma))
  slot=$((i % WINDOW_S))
  prev=${ring[$slot]:-$out}
  ring[$slot]=$out
  win=$(((out - prev) * page / 1048576))
  s100=$((10#${some/./})) f100=$((10#${full/./}))
  why=""
  if [ -n "$pid" ]; then
    [ "$usable" -lt "$MIN_USABLE_MB" ] && nu=$((nu + 1)) || nu=0
    [ "$s100" -gt $((PSI_SOME_PCT * 100)) ] && ns=$((ns + 1)) || ns=0
    [ "$f100" -gt $((PSI_FULL_PCT * 100)) ] && nf=$((nf + 1)) || nf=0
    [ "$avail" -lt "$MIN_AVAIL_MB" ] && why="MemAvailable $avail MiB, below $MIN_AVAIL_MB"
    [ "$nu" -ge "$USABLE_FLOOR_S" ] && why="${why:+$why; }MemAvailable minus CmaFree $usable MiB (MemAvailable $avail, CmaFree $cma), below $MIN_USABLE_MB for $nu s"
    [ "$ns" -ge "$PSI_SOME_S" ] && why="${why:+$why; }memory pressure some avg10 $some %, above $PSI_SOME_PCT % for $ns s"
    [ "$nf" -ge "$PSI_FULL_S" ] && why="${why:+$why; }memory pressure full avg10 $full %, above $PSI_FULL_PCT % for $nf s"
    [ "$win" -gt "$KILL_SWAPOUT_MIB" ] && why="${why:+$why; }swap-out $win MiB in $WINDOW_S s, above $KILL_SWAPOUT_MIB"
    if [ -n "$why" ]; then
      kill -9 "$pid" 2> /dev/null  # first, before anything that can wait on a disk
    fi
  else
    nu=0 nf=0 ns=0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$avail" "$cma" "$usable" "$(((st - sf) / 1024))" "$win" "$some" \
    "$full" "${pid:--}" >&"$lf"
  if [ -n "$why" ]; then
    ev "KILL $PROC $pid: $why"
    dump "$why"
    [ -n "$PID" ] && break
    pid="" nu=0 nf=0 ns=0
  fi
  i=$((i + 1))
  read -rt 1 -u "$tick"
done
RAM_WATCHDOG
}

write_card_log() {  # scripts/card_log.py, byte for byte (scripts/card_log_test.sh checks it)
  cat << 'CARD_LOG'
#!/usr/bin/env python3
"""card_log.py: a read-only log of every NVIDIA card's clocks, power, temperature and throttle reasons, through NVML.

One process for the whole run samples every card every --interval seconds (default 0.5) and appends one CSV row per
sample: NVML through ctypes on libnvidia-ml.so.1, so no install, no root and no nvidia-smi process per sample. It only
reads. It calls no NVML function that sets anything, and it never asks for a card's PCI fields, UUID or serial number
(the PCI path is the one NVIDIA's 580.196.07 fixes for frequent queries).

Columns: epoch (Unix seconds), time (local, to the millisecond), took_ms (how long the sample took), then for each
card i in NVML order (PCI bus order, as nvidia-smi numbers them):
  g<i>_pstate   P-state (P0 busy or boosting, P8 asleep)
  g<i>_sm_mhz   SM clock, the clock the card runs (on driver 580.178.04 it falls after work: R186)
  g<i>_mem_mhz  memory clock
  g<i>_w        power draw, W
  g<i>_c        temperature, degrees C
  g<i>_used_mib memory used, MiB
  g<i>_util     GPU utilization, %: the share of NVML's last sample period in which a kernel ran
  g<i>_reasons  the active clock event (throttle) reasons, a hex bit mask: 0x1 idle, 0x2 applications clocks,
                0x4 software power cap, 0x8 hardware slowdown, 0x10 sync boost, 0x20 software thermal,
                0x40 hardware thermal, 0x80 power brake, 0x100 display clock; 0x0 none
An empty field is a value NVML did not give. The per-card names are those of scripts/nvml_watchdog.py's log, so
scripts/card_table.py reads both. Lines starting with # are events: the start (driver version, cards, each card's
power limit and clock offsets as read), and the stop (why it stopped, samples, the process's CPU seconds, how long
sampling took, samples that started more than one interval late, and the settings again).

  card_log.py --log cards.csv [--interval 0.5] [--parent PID] [--duration SECONDS] [--stop STOPFILE]

It stops on SIGTERM, SIGINT or SIGHUP, when the --parent process is gone, after --duration seconds, or when the
--stop file exists. The log is appended to (the header is written to an empty file only). Exit status 0 on a clean
stop, 2 when NVML cannot be loaded or answers no card.
"""
import argparse
import ctypes
import os
import signal
import sys
import time

OK = 0
CLOCK_SM, CLOCK_MEM = 1, 2
TEMPERATURE_GPU = 0
FIELDS = ("pstate", "sm_mhz", "mem_mhz", "w", "c", "used_mib", "util", "reasons")


class MemInfo(ctypes.Structure):  # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Utilization(ctypes.Structure):  # nvmlUtilization_t
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def stamp(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


class Nvml:
    """The NVML calls this logger makes: every one a Get."""

    def __init__(self, lib="libnvidia-ml.so.1"):
        self.lib = ctypes.CDLL(lib)
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        r = self.lib.nvmlInit_v2()
        if r != OK:
            raise RuntimeError(f"nvmlInit: {self.lib.nvmlErrorString(r).decode()}")
        # newer drivers name it ClocksEventReasons, older ones ClocksThrottleReasons (same bits)
        self.reasons_fn = next((n for n in ("nvmlDeviceGetCurrentClocksEventReasons",
                                            "nvmlDeviceGetCurrentClocksThrottleReasons")
                                if getattr(self.lib, n, None) is not None), None)

    def call(self, name, *args):
        f = getattr(self.lib, name, None)
        return 13 if f is None else f(*args)  # 13: NVML_ERROR_FUNCTION_NOT_FOUND

    def count(self):
        n = ctypes.c_uint()
        return n.value if self.call("nvmlDeviceGetCount_v2", ctypes.byref(n)) == OK else 0

    def handle(self, i):
        h = ctypes.c_void_p()
        return h if self.call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(i), ctypes.byref(h)) == OK else None

    def uint(self, name, h, *args):
        v = ctypes.c_uint()
        return v.value if self.call(name, h, *args, ctypes.byref(v)) == OK else None

    def sint(self, name, h):
        v = ctypes.c_int()
        return v.value if self.call(name, h, ctypes.byref(v)) == OK else None

    def driver(self):
        buf = ctypes.create_string_buffer(96)
        r = self.call("nvmlSystemGetDriverVersion", buf, ctypes.c_uint(96))
        return buf.value.decode() if r == OK else "?"

    def name(self, h):
        buf = ctypes.create_string_buffer(96)
        r = self.call("nvmlDeviceGetName", h, buf, ctypes.c_uint(96))
        return buf.value.decode() if r == OK else "?"

    def sample(self, h):
        """One card's fields, in FIELDS order, as strings ('' where NVML gave nothing)."""
        ps = self.sint("nvmlDeviceGetPerformanceState", h)
        sm = self.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(CLOCK_SM))
        mem = self.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(CLOCK_MEM))
        mw = self.uint("nvmlDeviceGetPowerUsage", h)
        tc = self.uint("nvmlDeviceGetTemperature", h, ctypes.c_int(TEMPERATURE_GPU))
        mi = MemInfo()
        used = mi.used >> 20 if self.call("nvmlDeviceGetMemoryInfo", h, ctypes.byref(mi)) == OK else None
        ut = Utilization()
        util = ut.gpu if self.call("nvmlDeviceGetUtilizationRates", h, ctypes.byref(ut)) == OK else None
        ev = ctypes.c_ulonglong()
        rs = ev.value if self.reasons_fn and self.call(self.reasons_fn, h, ctypes.byref(ev)) == OK else None
        return [f"P{ps}" if ps is not None and 0 <= ps <= 15 else "",
                "" if sm is None else str(sm), "" if mem is None else str(mem),
                "" if mw is None else f"{mw / 1000:.2f}", "" if tc is None else str(tc),
                "" if used is None else str(used), "" if util is None else str(util),
                "" if rs is None else f"0x{rs:x}"]

    def settings(self, i, h):
        """What a card is set to, for the start and stop lines: power limit and both clock offsets, as read."""
        lim = self.uint("nvmlDeviceGetEnforcedPowerLimit", h)
        mo = self.sint("nvmlDeviceGetMemClkVfOffset", h)
        go = self.sint("nvmlDeviceGetGpcClkVfOffset", h)
        return (f"g{i} limit {'?' if lim is None else f'{lim / 1000:.0f}'} W, memory offset "
                f"{'?' if mo is None else mo}, core offset {'?' if go is None else go}")

    def shutdown(self):
        self.call("nvmlShutdown")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Stop(Exception):
    pass


SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def on_signal(*_):
    for s in SIGNALS:  # one stop: a second signal must not cut the stop line
        signal.signal(s, signal.SIG_IGN)
    raise Stop()


def run(a, nv):
    n = nv.count()
    handles = [nv.handle(i) for i in range(n)]
    if not n or any(h is None for h in handles):
        print(f"card_log: NVML answers {n} cards, handles for {sum(h is not None for h in handles)}", file=sys.stderr)
        return 2
    log = open(a.log, "a", buffering=1)
    if log.tell() == 0:
        log.write(",".join(["epoch", "time", "took_ms"] + [f"g{i}_{k}" for i in range(n) for k in FIELDS]) + "\n")
    t = time.time()
    names = sorted(set(nv.name(h) for h in handles))
    log.write(f"# {stamp(t)} card_log start: pid {os.getpid()}, driver {nv.driver()}, {n} cards ({', '.join(names)}), "
              f"interval {a.interval} s; {'; '.join(nv.settings(i, h) for i, h in enumerate(handles))}\n")
    cpu0 = os.times()
    took, late, samples = [], 0, 0
    t_end = time.monotonic() + a.duration if a.duration else None
    nxt = time.monotonic()
    why = "stopped"
    try:
        for s in SIGNALS:  # a signal ends the run at once, even in the middle of a sample (that row is not written)
            signal.signal(s, on_signal)
        while True:
            m0 = time.monotonic()
            if samples and m0 - nxt > a.interval:  # this sample started more than one interval late
                late += 1
                nxt = m0
            t0 = time.time()
            row = []
            for h in handles:
                row += nv.sample(h)
            ms = (time.monotonic() - m0) * 1000
            line = f"{t0:.3f},{stamp(t0)},{ms:.1f}," + ",".join(row) + "\n"
            log.write(line)
            took.append(ms)
            samples += 1
            if a.parent and not alive(a.parent):
                why = f"parent {a.parent} is gone"
                break
            if a.stop and os.path.exists(a.stop):
                why = "stop file"
                break
            if t_end and m0 >= t_end:
                why = f"after {a.duration:g} s"
                break
            nxt += a.interval
            time.sleep(max(0.0, nxt - time.monotonic()))
    except Stop:
        why = "signal"
    for s in SIGNALS:
        signal.signal(s, signal.SIG_IGN)
    cpu = os.times()
    took.sort()
    med = took[len(took) // 2] if took else 0.0
    t = time.time()
    log.write(f"# {stamp(t)} card_log stop ({why}): {samples} samples, "
              f"{cpu.user - cpu0.user + cpu.system - cpu0.system:.2f} s CPU (user+sys), sampling {med:.1f} ms median, "
              f"{took[-1] if took else 0.0:.1f} ms max, {late} samples late by more than one interval; "
              f"{'; '.join(nv.settings(i, h) for i, h in enumerate(handles))}\n")
    log.close()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="the CSV file (appended to)")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between samples (default 0.5)")
    p.add_argument("--parent", type=int, default=0, help="stop when this process is gone")
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0: until stopped)")
    p.add_argument("--stop", help="stop when this file exists")
    a = p.parse_args()
    if a.interval <= 0:
        print("card_log: --interval must be above 0", file=sys.stderr)
        return 2
    try:
        nv = Nvml()
    except (OSError, RuntimeError) as e:
        print(f"card_log: no NVML: {e}", file=sys.stderr)
        return 2
    try:
        return run(a, nv)
    finally:
        nv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
CARD_LOG
}

# card_log_start: the card log on every card to LOGDIR/cards.csv for the whole run (from the first start to end), its
# script on tmpfs like the watchdog's. It stops by itself after 15 hours in case the run's end never comes. A log that
# does not come up is reported, not fatal: the run goes on without it.
card_log_start() {
  local pid n=0 py=/dev/shm/card_log-$PORT.py
  write_card_log > "$py"
  setsid nohup python3 -I "$py" --log "$PWD/cards.csv" --interval "${CARD_LOG_S:-0.5}" --duration 54000 \
    > card_log.out 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > card_log.pid
  until grep -qs '^[0-9]' cards.csv; do
    n=$((n + 1))
    if ! kill -0 "$pid" 2> /dev/null; then echo "card log: not running: $(tr '\n' ' ' < card_log.out)"; return 0; fi
    if [ "$n" -gt 20 ]; then echo "card log: no sample after 10 s (it keeps trying; card_log.out)"; return 0; fi
    sleep 0.5
  done
  echo "card log: every card every ${CARD_LOG_S:-0.5} s to cards.csv"
}

# card_log_stop: ends the card log and waits up to 3 s for its stop line
card_log_stop() {
  local pid n=0
  [ -f card_log.pid ] || return 0
  pid=$(cat card_log.pid)
  kill "$pid" 2> /dev/null
  while kill -0 "$pid" 2> /dev/null && [ "$n" -lt 6 ]; do sleep 0.5; n=$((n + 1)); done
  rm -f card_log.pid "/dev/shm/card_log-$PORT.py"
  [ -s card_log.out ] || rm -f card_log.out
}

# wd_start ARM PID: the RAM watchdog on the server, at real-time priority, its script and TSV on tmpfs so that neither
# waits on a disk; the TSV is copied to LOGDIR/ram_watchdog-<arm>.tsv by stop_server. Fails when it does not come up.
wd_start() {
  local sh=/dev/shm/ram_watchdog-$PORT.sh tsv=/dev/shm/ram_watchdog-$PORT.tsv pre n=0
  write_watchdog > "$sh"
  rm -f "$tsv"
  if command -v chrt > /dev/null; then pre="chrt -f ${WATCHDOG_RT_PRIO:-10}"; else pre="nice -n -20"; fi
  # shellcheck disable=SC2086
  PID=$2 LOG=$tsv SERVER_LOG=$PWD/server-$1.log MIN_AVAIL_MB=${MIN_AVAIL_MB:-400} \
    setsid nohup $pre bash "$sh" > watchdog.out 2>&1 < /dev/null &
  echo $! > watchdog.pid
  echo "$tsv ram_watchdog-$1.tsv $sh" > watchdog.files
  until grep -qs '^[0-9]' "$tsv"; do  # the first sample row
    n=$((n + 1))
    if [ "$n" -gt 12 ] || ! kill -0 "$(cat watchdog.pid)" 2> /dev/null; then
      echo "the RAM watchdog did not come up; it said:"
      cat watchdog.out
      return 1
    fi
    sleep 0.5
  done
  grep -m 1 'ram_watchdog:' "$tsv" | sed 's/^# [0-9]* /watchdog up: /'
}

stop_server() {
  if [ -f watchdog.pid ]; then
    wd=$(cat watchdog.pid)
    kill "$wd" 2> /dev/null
    for _ in 1 2 3 4 5 6; do kill -0 "$wd" 2> /dev/null || break; sleep 0.5; done
    kill -9 "$wd" 2> /dev/null
  fi
  if [ -f server.pid ]; then
    srv=$(cat server.pid)
    kill "$srv" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$srv" 2> /dev/null || break; sleep 1; done
    kill -9 "$srv" 2> /dev/null
    echo "server stopped"
  fi
  if [ -f watchdog.files ]; then
    read -r tsv dest sh < watchdog.files
    [ -f "$tsv" ] && cp "$tsv" "$dest"
    rm -f "$tsv" "$sh" watchdog.files
    [ -s watchdog.out ] || rm -f watchdog.out
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
    setsid nohup bash -c 'while true; do
        echo "$(date +%s) $(awk "/^MemAvailable/ {a=\$2} /^SwapTotal/ {t=\$2} /^SwapFree/ {f=\$2} END {print a, t - f}" /proc/meminfo)"
        sleep 2; done' > meminfo.log 2>&1 < /dev/null &
    echo $! >> monitors.pid
    card_log_start
  fi
  echo "=== $(date -Is) $ARM: $BIN ${SERVER_ENV:-} ${SERVER_ARGS:-} -m $MODEL" >> "server-$ARM.log"
  # shellcheck disable=SC2086
  env ${SERVER_ENV:-} setsid nohup "$BIN" ${SERVER_ARGS:-} -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
    >> "server-$ARM.log" 2>&1 < /dev/null &
  srv=$!
  echo $srv > server.pid
  if ! wd_start "$ARM" "$srv"; then stop_server; exit 1; fi
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
  card_log_stop
  [ -f monitors.pid ] && xargs -r kill < monitors.pid 2> /dev/null
  rm -f monitors.pid
  echo "monitors stopped"
  ;;
esac
