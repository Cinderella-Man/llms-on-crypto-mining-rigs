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
              [--fast] [--slow-every N] [--profile]

It stops on SIGTERM, SIGINT or SIGHUP, when the --parent process is gone, after --duration seconds, or when the
--stop file exists. The log is appended to (the header is written to an empty file only). Exit status 0 on a clean
stop, 2 when NVML cannot be loaded or answers no card.

Three options for the host's CPU, all off unless asked for (each also from the environment, so a driver that starts
this script passes them on unchanged):
  --fast (CARD_LOG_FAST=1)  the same NVML calls in the same order and the same rows, with less Python work: each NVML
      function looked up once, each card's arguments and result buffers made once, and the rows written to the file
      in blocks (the start, the first row at once, then every 5 s and at the stop; killed with SIGKILL, it loses up to
      5 s of rows). A second # line after the start line says it is on.
  --slow-every N (CARD_LOG_SLOW_EVERY=N, implies --fast)  reads the memory clock, temperature and memory used of a
      card only on every Nth sample and on any sample where its P-state changed; the rows in between repeat the
      value last read. That leaves 5 of a card's 8 NVML calls on the other samples.
  --profile (CARD_LOG_PROFILE=1)  times every NVML call on this thread's CPU clock and writes one more # line just
      before the stop line: the process's CPU a sample (user and system), the share of it inside NVML, and for each
      function its calls a sample and CPU ms a call, the costliest first.
A slower --interval (the drivers' CARD_LOG_S, for one) cuts the cost in proportion.
"""
import argparse
import ctypes
import os
import resource
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


FLUSH_S = 5.0  # --fast: seconds between writes of the buffered rows to the file


class FastCard:
    """One card's NVML arguments and result buffers for FastSample, made once."""
    __slots__ = ("ps", "sm", "mem", "mw", "tc", "mi", "ut", "ev", "a_ps", "a_sm", "a_mem", "a_mw", "a_tc", "a_mi",
                 "a_ut", "a_ev", "last")

    def __init__(self, h):
        self.ps, self.sm, self.mem, self.mw, self.tc = (ctypes.c_int(), ctypes.c_uint(), ctypes.c_uint(),
                                                        ctypes.c_uint(), ctypes.c_uint())
        self.mi, self.ut, self.ev = MemInfo(), Utilization(), ctypes.c_ulonglong()
        b = ctypes.byref
        self.a_ps, self.a_mw, self.a_mi, self.a_ut, self.a_ev = ((h, b(self.ps)), (h, b(self.mw)), (h, b(self.mi)),
                                                                 (h, b(self.ut)), (h, b(self.ev)))
        self.a_sm = (h, ctypes.c_int(CLOCK_SM), b(self.sm))
        self.a_mem = (h, ctypes.c_int(CLOCK_MEM), b(self.mem))
        self.a_tc = (h, ctypes.c_int(TEMPERATURE_GPU), b(self.tc))
        self.last = None  # --slow-every: (P-state, memory clock, temperature, memory used) as last read


class FastSample:
    """--fast: Nvml.sample's calls for every card, in its order, with each NVML function looked up once and each card's
    arguments and buffers made once; row() gives the same fields as Nvml.sample does for all cards. With slow_every N
    above 1, a card's memory clock, temperature and memory used are read on every Nth sample and when its P-state
    changes, and repeated from the last read in between."""

    def __init__(self, nv, handles, slow_every=1):
        def fn(name):
            return getattr(nv.lib, name, None) if name else None
        self.fns = tuple(fn(n) for n in ("nvmlDeviceGetPerformanceState", "nvmlDeviceGetClockInfo",
                                         "nvmlDeviceGetPowerUsage", "nvmlDeviceGetTemperature",
                                         "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
                                         nv.reasons_fn))
        self.cards = [FastCard(h) for h in handles]
        self.every, self.n = max(1, slow_every), 0

    def row(self):
        f_ps, f_clk, f_mw, f_tc, f_mi, f_ut, f_ev = self.fns
        due = self.n % self.every == 0
        self.n += 1
        out = []
        for c in self.cards:
            p = c.ps.value if f_ps is not None and f_ps(*c.a_ps) == OK else None
            ps = f"P{p}" if p is not None and 0 <= p <= 15 else ""
            sm = str(c.sm.value) if f_clk is not None and f_clk(*c.a_sm) == OK else ""
            last = c.last
            slow = due or last is None or last[0] != ps
            if slow:
                mem = str(c.mem.value) if f_clk is not None and f_clk(*c.a_mem) == OK else ""
            mw = f"{c.mw.value / 1000:.2f}" if f_mw is not None and f_mw(*c.a_mw) == OK else ""
            if slow:
                tc = str(c.tc.value) if f_tc is not None and f_tc(*c.a_tc) == OK else ""
                used = str(c.mi.used >> 20) if f_mi is not None and f_mi(*c.a_mi) == OK else ""
                c.last = (ps, mem, tc, used)
            else:
                _, mem, tc, used = last
            util = str(c.ut.gpu) if f_ut is not None and f_ut(*c.a_ut) == OK else ""
            rs = f"0x{c.ev.value:x}" if f_ev is not None and f_ev(*c.a_ev) == OK else ""
            out += (ps, sm, mem, mw, tc, used, util, rs)
        return out


class Timed:
    """--profile: stands in for the NVML library, timing each call of each function on this thread's CPU clock."""

    def __init__(self, lib):
        self.lib, self.stats = lib, {}

    def __getattr__(self, name):  # once per function: the wrapper is then kept as an attribute
        f = getattr(self.lib, name)  # a function the library lacks stays missing (AttributeError)
        st = self.stats.setdefault(name, [0, 0.0])
        clock = time.thread_time

        def timed(*args):
            c0 = clock()
            r = f(*args)
            st[1] += clock() - c0
            st[0] += 1
            return r
        setattr(self, name, timed)
        return timed

    def line(self, samples, cpu_s, user_s, sys_s):
        """The profile line's text: CPU a sample, the share in NVML, and each function's calls and cost."""
        k = max(samples, 1)
        fns = sorted(self.stats.items(), key=lambda kv: -kv[1][1])
        nvml = sum(s for _, s in self.stats.values())
        each = "; ".join(f"{name} {n / k:g} a sample, {s / max(n, 1) * 1000:.3f} ms a call" for name, (n, s) in fns)
        return (f"{samples} samples, CPU {cpu_s / k * 1000:.2f} ms a sample (user {user_s / k * 1000:.2f}, system "
                f"{sys_s / k * 1000:.2f}), in NVML calls {nvml / k * 1000:.2f} ms a sample on this thread's clock; "
                f"{each or 'no NVML calls'}")


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
    log = open(a.log, "a", buffering=1 << 16 if a.fast else 1)  # --fast: rows go to the file every FLUSH_S
    if log.tell() == 0:
        log.write(",".join(["epoch", "time", "took_ms"] + [f"g{i}_{k}" for i in range(n) for k in FIELDS]) + "\n")
    t = time.time()
    names = sorted(set(nv.name(h) for h in handles))
    log.write(f"# {stamp(t)} card_log start: pid {os.getpid()}, driver {nv.driver()}, {n} cards ({', '.join(names)}), "
              f"interval {a.interval} s; {'; '.join(nv.settings(i, h) for i, h in enumerate(handles))}\n")
    fast = None
    if a.fast:
        reads = (f"memory clock, temperature and memory used on one sample in {a.slow_every} and at a P-state change, "
                 f"the other fields on every sample" if a.slow_every > 1 else "every field on every sample")
        log.write(f"# {stamp(t)} card_log fast: NVML functions and buffers set up once, rows written every "
                  f"{FLUSH_S:g} s; reads {reads}{', profiled' if a.profile else ''}\n")
        log.flush()
    timed = None
    if a.profile:  # from here on, every NVML call is timed: the samples and the stop line's settings
        timed = nv.lib = Timed(nv.lib)
    if a.fast:
        fast = FastSample(nv, handles, a.slow_every).row
    flush_at = None  # --fast: when the buffered rows next go to the file (the first row at once: drivers wait for it)
    cpu0 = os.times()
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if timed else None
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
            if fast:
                row = fast()
            else:
                row = []
                for h in handles:
                    row += nv.sample(h)
            ms = (time.monotonic() - m0) * 1000
            line = f"{t0:.3f},{stamp(t0)},{ms:.1f}," + ",".join(row) + "\n"
            log.write(line)
            took.append(ms)
            samples += 1
            if fast and (flush_at is None or m0 >= flush_at):
                log.flush()
                flush_at = m0 + FLUSH_S
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
    if timed:
        ru = resource.getrusage(resource.RUSAGE_SELF)  # microseconds, where os.times() counts 10 ms ticks
        user, system = ru.ru_utime - ru0.ru_utime, ru.ru_stime - ru0.ru_stime
        log.write(f"# {stamp(t)} card_log profile: {timed.line(samples, user + system, user, system)}\n")
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
    env = os.environ.get
    p.add_argument("--fast", action="store_true", default=env("CARD_LOG_FAST", "0") not in ("", "0"),
                   help="the same rows with less Python work, written every 5 s (CARD_LOG_FAST=1)")
    p.add_argument("--slow-every", type=int, default=env("CARD_LOG_SLOW_EVERY") or "1", metavar="N",
                   help="memory clock, temperature and memory used on one sample in N and at a P-state change; "
                        "implies --fast (CARD_LOG_SLOW_EVERY, default 1: every sample)")
    p.add_argument("--profile", action="store_true", default=env("CARD_LOG_PROFILE", "0") not in ("", "0"),
                   help="time every NVML call; one more # line before the stop line (CARD_LOG_PROFILE=1)")
    a = p.parse_args()
    if a.interval <= 0:
        print("card_log: --interval must be above 0", file=sys.stderr)
        return 2
    if a.slow_every < 1:
        print("card_log: --slow-every (CARD_LOG_SLOW_EVERY) must be 1 or more", file=sys.stderr)
        return 2
    a.fast = a.fast or a.slow_every > 1
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
