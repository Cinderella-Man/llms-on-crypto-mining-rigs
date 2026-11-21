#!/usr/bin/env bash
# rig_taper.sh: C218r (Q771, Q770's first test), ON the rig and detached, from a folder that holds this script,
# taper_replay.py and the agent check's turns.csv (docs/data/2026-11-03_retype_agent/agent/turns.csv). Ends with OUT/DONE.
# Two server loads of the taper build (pascal-p106-v3-b12076-taper) with the agent check's line; the taper is off on the
# server and set per request ("ubatch_taper"), so each load runs both arms:
#   A  --ctx-checkpoints 4 (as the agent check): the replay of the check's 62 prompts in PASSES_A (default off, on, on,
#      off), then FRESH fresh prompts of FRESH_N tokens with the taper off and on
#   B  --ctx-checkpoints 0 (no checkpoint drains, Q770's no-code test): the same with PASSES_B (default off, on), so
#      that A's and B's fresh prompts give the drains' cost, and B with the taper on approximates no drains plus taper
# Every load logs its time, memory per card and on the host, the checkpoint and taper lines of its server log, and a
# watchdog stops the server when MemAvailable falls below MIN_AVAIL_MB.
# Since C804 (Q1408, 2026-11-12) every load also counts the target graph's reallocations per request from its server
# log's print_timing lines (`graph reallocs = S / D`, as scripts/realloc_counts.py counts them): the table in
# load-<L>/reallocs.tsv and one line in run.log ("load <L>: graph reallocations: W of N requests (S at the same size,
# D changed)"). The loads and the requests are the same as before; a switch whose loads reallocate where its line
# without it does not is marked with scripts/realloc_counts.py --marks.
# Since R138 (Q1430, 2026-11-12) the watchdog writes SwapFree beside MemAvailable: load-<L>/memavail.txt has three
# columns, epoch seconds, MemAvailable MiB and SwapFree MiB, every 2 s (readers that take the first two are unchanged),
# and the end-of-load line in run.log also gives the lowest SwapFree.
# Since C873 (Q1576, 2026-11-21) a card log runs for the whole run: scripts/card_log.py, embedded below byte for byte
# (the launchers copy this one script to the rig), samples every card every CARD_LOG_S seconds (0.5) through NVML,
# read-only, one process: P-state, SM and memory clock, power, temperature, memory used, utilization and throttle
# reasons, to OUT/cards.csv (scripts/card_table.py summarises it). It starts before the first load and stops when the
# run ends, when this script is gone (--parent) or when OUT/cards.csv.stop exists. run.log says "card log: every card
# every 0.5 s to cards.csv", or "card log: not running: ..." (OUT/card_log.out says why; the run goes on without it).
# CARD_LOG=0 turns it off. The loads, the replay and every other output are as before.
# Env: BUILDS (folder of the builds), BUILD (the taper build's folder), MODEL (first shard), OUT (default ./out),
#      CTX (65536), ARGS (default '-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio'), PORT (8095),
#      LOADS ("A B"), PASSES_A ("off on on off"; a spec such as short=2048:128 is a pass too), PASSES_B ("off on"),
#      FRESH (4), FRESH_N (2048), GEN (8), PAD (0),
#      TASKS (all), MIN_AVAIL_MB (150), FORCE=1 starts even with a compute process on the cards, CARD_LOG (1),
#      CARD_LOG_S (0.5).
#   ssh $RIG "BUILDS=<builds> BUILD=<taper build> MODEL=<first shard> OUT=<work>/out \
#     setsid nohup bash <work>/rig_taper.sh > <work>/rig_taper.out 2>&1 < /dev/null &"
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${BUILDS:?BUILDS=<folder of the builds>}" "${BUILD:?BUILD=<the taper build>}" "${MODEL:?MODEL=<first shard>}"
OUT=${OUT:-$here/out}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
PORT=${PORT:-8095}
LOADS=${LOADS:-A B}
PASSES_A=${PASSES_A:-off on on off}
PASSES_B=${PASSES_B:-off on}
FRESH=${FRESH:-4}
FRESH_N=${FRESH_N:-2048}
GEN=${GEN:-8}
PAD=${PAD:-0}
TASKS=${TASKS:-}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-150}
FORCE=${FORCE:-0}
CARD_LOG=${CARD_LOG:-1}
CARD_LOG_S=${CARD_LOG_S:-0.5}
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
# realloc_count SERVER_LOG TSV (C804): the target graph's reallocations per request from print_timing's running count
# (`graph reallocs = S / D` since the load, after the request's `prompt eval time = X ms / N tokens`); writes one row
# per request to TSV and prints the summary
realloc_count() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" 2> /dev/null | awk -v out="$2" '
        BEGIN { print "req\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed" > out }
        /prompt eval time =/ { t = $0; sub(/.*prompt eval time = */, "", t); split(t, p, " "); ms = p[1]; n = p[4]; cur = 1; next }
        cur && /graph reallocs =/ { t = $0; sub(/.*graph reallocs = */, "", t); split(t, q, " ")
            s = q[1] - ps; d = q[3] - pd; ps = q[1]; pd = q[3]; cur = 0
            printf "%d\t%s\t%s\t%d\t%d\n", k, n, ms, s, d > out
            k++; ts += s; td += d; if (s + d > 0) w++ }
        END { printf "graph reallocations: %d of %d requests (%d at the same size, %d changed)\n", w, k, ts, td }'
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

# card_log_start: the card log on every card to OUT/cards.csv until card_log_stop, until this script is gone
# (--parent $$) or until OUT/cards.csv.stop exists; its first sample is waited for up to 10 s. A log that does not
# come up is reported, not fatal.
CL_PID=""
card_log_start() {
    local n=0
    [ "$CARD_LOG" = 1 ] || return 0
    write_card_log > "$OUT/card_log.py"
    rm -f "$OUT/cards.csv.stop"
    python3 -I "$OUT/card_log.py" --log "$OUT/cards.csv" --interval "$CARD_LOG_S" --parent $$ --stop "$OUT/cards.csv.stop" \
        > "$OUT/card_log.out" 2>&1 < /dev/null &
    CL_PID=$!
    until grep -qs '^[0-9]' "$OUT/cards.csv"; do
        n=$((n + 1))
        if ! kill -0 "$CL_PID" 2> /dev/null; then log "card log: not running: $(tr '\n' ' ' < "$OUT/card_log.out")"; CL_PID=""; return 0; fi
        if [ "$n" -gt 20 ]; then log "card log: no sample after 10 s (it keeps trying; card_log.out)"; return 0; fi
        sleep 0.5
    done
    log "card log: every card every $CARD_LOG_S s to cards.csv"
}

# card_log_stop: ends the card log (SIGTERM: it writes its stop line) and waits up to 3 s; never waits on a hung one
card_log_stop() {
    local n=0
    [ -n "$CL_PID" ] || return 0
    kill "$CL_PID" 2> /dev/null
    while kill -0 "$CL_PID" 2> /dev/null && [ "$n" -lt 6 ]; do sleep 0.5; n=$((n + 1)); done
    kill -9 "$CL_PID" 2> /dev/null
    CL_PID=""
    rm -f "$OUT/card_log.py"
    [ -s "$OUT/card_log.out" ] || rm -f "$OUT/card_log.out"
}

apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then
    log "compute processes on the cards (FORCE=1 to go on): $apps"
    touch "$OUT/DONE"
    exit 2
fi
{ uptime; free -m; nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv; } \
    > "$OUT/quiet_check.txt" 2>&1
md5sum "$here/rig_taper.sh" "$here/taper_replay.py" "$here/turns.csv" "$BUILDS/$BUILD/bin/llama-server" > "$OUT/md5.txt" 2>&1
grep -E '^(name|commit)' "$BUILDS/$BUILD/BUILDINFO.txt" >> "$OUT/md5.txt" 2>/dev/null
card_log_start

for L in $LOADS; do
    case $L in
        A) CK=4; PASSES=$PASSES_A ;;
        B) CK=0; PASSES=$PASSES_B ;;
        *) log "unknown load $L"; continue ;;
    esac
    d=$OUT/load-$L
    mkdir -p "$d"
    log "load $L: --ctx-checkpoints $CK, passes $PASSES, $FRESH fresh prompts of $FRESH_N tokens"
    pkill -x llama-server; sleep 2
    # shellcheck disable=SC2086
    "$BUILDS/$BUILD/bin/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" $ARGS -np 1 -bs --ctx-checkpoints "$CK" \
        --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5 --host 127.0.0.1 --port "$PORT" \
        > "$d/server.log" 2>&1 < /dev/null &
    srv=$!
    # host-memory watchdog: the 4 GB host must not swap the server out or start the OOM killer
    ( while kill -0 $srv 2> /dev/null; do
          read -r a sf <<< "$(awk '/^MemAvailable:/ { a = int($2 / 1024) } /^SwapFree:/ { s = int($2 / 1024) } END { print a, s }' /proc/meminfo)"
          if [ "$a" -lt "$MIN_AVAIL_MB" ]; then
              echo "$(date -Is) watchdog: MemAvailable $a MiB < $MIN_AVAIL_MB, stopping the server" >> "$OUT/run.log"
              kill -INT $srv; break
          fi
          echo "$(date +%s) $a $sf" >> "$d/memavail.txt"
          sleep 2
      done ) &
    wd=$!
    t0=$(date +%s)
    for _ in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    log "load $L: up after $(( $(date +%s) - t0 )) s"
    grep -E 'model buffer size|KV buffer size|RS buffer size|compute buffer size|n_ctx *=|n_ubatch|checkpoints|out of memory|failed|error' \
        "$d/server.log" | grep -v 'unused tensor' | cut -c1-220 | head -60 > "$d/load_lines.txt"
    { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; free -m; } > "$d/memory_loaded.txt"
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        python3 -I "$here/taper_replay.py" --port "$PORT" --turns "$here/turns.csv" --passes "$PASSES" --gen "$GEN" \
            --pad "$PAD" --fresh "$FRESH" --fresh-n "$FRESH_N" ${TASKS:+--tasks "$TASKS"} --label "$L" --out "$d" \
            > "$d/replay.out" 2>&1
        log "load $L: replay exit $?; $(tr '\n' ' ' < "$d/summary.txt" 2>/dev/null | cut -c1-600)"
    else
        log "load $L: server did not start"; tail -20 "$d/server.log" >> "$OUT/run.log"
    fi
    { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; free -m; } > "$d/memory_after.txt"
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server; kill $wd 2> /dev/null
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -oE '(created|restored) context checkpoint.*' > "$d/checkpoints.txt"
    sed 's/\x1b\[[0-9;]*m//g' "$d/server.log" | grep -o 'ubatch taper: .*' > "$d/taper_lines.txt"
    log "load $L: $(wc -l < "$d/checkpoints.txt") checkpoint lines, $(wc -l < "$d/taper_lines.txt") taper lines, lowest MemAvailable $(sort -k2 -n "$d/memavail.txt" | head -n 1 | cut -d' ' -f2) MiB, lowest SwapFree $(sort -k3 -n "$d/memavail.txt" | head -n 1 | cut -d' ' -f3) MiB"
    log "load $L: $(realloc_count "$d/server.log" "$d/reallocs.tsv")"
done
card_log_stop
log "done"
touch "$OUT/DONE"
