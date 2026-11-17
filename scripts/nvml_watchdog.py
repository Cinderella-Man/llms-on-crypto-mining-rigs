#!/usr/bin/env python3
"""A safety watchdog for clock-offset tests on NVIDIA cards, through NVML (ctypes on libnvidia-ml.so.1, no install).

Samples every card every INTERVAL seconds (temperature, power, power limit, SM and memory clocks, P-state, memory used,
clock event reasons, and the memory and graphics clock offsets of the card under test) and logs every sample to a CSV
file. It reads the kernel log (/dev/kmsg) as it grows. It trips on any of:
  - any card above --temp-max degrees C;
  - the card under test above --power-max W for --power-secs seconds in a row;
  - a new kernel log line that contains --xid-pattern (default "NVRM: Xid").
On a trip it at once sets both clock offsets of the card under test back to 0 (both NVML offset APIs), reads them
back, kills every process whose name is in --kill (TERM, then KILL after 2 s), and appends a line to the --marker
file. It keeps sampling afterwards; a condition that persists trips again, at most once every 5 s.

While it runs it also holds every card open, so a power limit or an offset set by a short-lived process stays in
place with persistence mode off. Stop it with SIGTERM or SIGINT, or by creating the --stop file.

  nvml_watchdog.py --gpu 0 --log wd.csv --marker TRIPPED [--temp-max 70] [--power-max 55] [--power-secs 5]
                   [--interval 0.5] [--xid-pattern 'NVRM: Xid'] [--kill mmvq_pascal,cuda_memtest,llama-server,...]
                   [--stop STOPFILE] [--duration SECONDS]

Needs root (the kernel log and the offset calls). Exit status 0 on a clean stop, 2 on a usage or NVML error.
"""
import argparse
import ctypes
import errno
import os
import signal
import sys
import time

OK = 0
CLOCK_GRAPHICS, CLOCK_SM, CLOCK_MEM = 0, 1, 2
DEFAULT_KILL = ("mmvq_pascal,cuda_memtest,llama-server,llama-bench,llama-perplexit,llama-completi,llama-cli,"
                "test-backend-o,memoc_steps.sh,offset_climb.sh")


class ClockOffset(ctypes.Structure):  # nvmlClockOffset_v1_t
    _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int),
                ("clockOffsetMHz", ctypes.c_int), ("minClockOffsetMHz", ctypes.c_int),
                ("maxClockOffsetMHz", ctypes.c_int)]


CLOCK_OFFSET_V1 = ctypes.sizeof(ClockOffset) | (1 << 24)


class MemInfo(ctypes.Structure):  # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        r = self.lib.nvmlInit_v2()
        if r != OK:
            sys.exit(f"nvmlInit: {self.lib.nvmlErrorString(r).decode()}")

    def call(self, name, *args):
        f = getattr(self.lib, name, None)
        return 13 if f is None else f(*args)

    def handle(self, i):
        h = ctypes.c_void_p()
        r = self.call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(i), ctypes.byref(h))
        if r != OK:
            sys.exit(f"gpu {i}: NVML error {r}")
        return h

    def uint(self, name, h, *args):
        v = ctypes.c_uint()
        return v.value if self.call(name, h, *args, ctypes.byref(v)) == OK else None

    def vf(self, h, dom):
        v = ctypes.c_int()
        return v.value if self.call(f"nvmlDeviceGet{dom}VfOffset", h, ctypes.byref(v)) == OK else None

    def reset_offsets(self, h):
        """Both offsets to 0 through both APIs; returns the read-back (mem, gpc)."""
        for dom in ("MemClk", "GpcClk"):
            self.call(f"nvmlDeviceSet{dom}VfOffset", h, ctypes.c_int(0))
        for clock in (CLOCK_MEM, CLOCK_GRAPHICS):
            info = ClockOffset(version=CLOCK_OFFSET_V1, type=clock, pstate=0, clockOffsetMHz=0)
            self.call("nvmlDeviceSetClockOffsets", h, ctypes.byref(info))
        return self.vf(h, "MemClk"), self.vf(h, "GpcClk")


class Kmsg:
    """New kernel log records since start, read without blocking."""

    def __init__(self):
        self.fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
        os.lseek(self.fd, 0, os.SEEK_END)

    def new_lines(self):
        out = []
        while True:
            try:
                rec = os.read(self.fd, 8192)
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    return out
                if e.errno == errno.EPIPE:  # records overwritten before we read them: carry on
                    continue
                raise
            if not rec:
                return out
            text = rec.decode(errors="replace")
            out.append(text.split(";", 1)[-1].strip())


def procs_named(names):
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
        except OSError:
            continue
        if comm in names:
            found.append((int(pid), comm))
    return found


def kill_named(names):
    victims = procs_named(names)
    for pid, _ in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if victims:
        time.sleep(2)
        for pid, _ in procs_named(names):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return victims


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0, help="the card under test (NVML index, PCI bus order)")
    p.add_argument("--log", required=True)
    p.add_argument("--marker", required=True)
    p.add_argument("--temp-max", type=float, default=70.0)
    p.add_argument("--power-max", type=float, default=55.0)
    p.add_argument("--power-secs", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--xid-pattern", default="NVRM: Xid")
    p.add_argument("--kill", default=DEFAULT_KILL, help="comma-separated process names (as in /proc/PID/comm)")
    p.add_argument("--stop", help="exit cleanly when this file exists")
    p.add_argument("--duration", type=float, default=0, help="exit after this many seconds (0: run until stopped)")
    a = p.parse_args()
    if os.geteuid() != 0:
        print("needs root", file=sys.stderr)
        return 2
    kill_names = set(x for x in a.kill.split(",") if x)
    nv = Nvml()
    n = ctypes.c_uint()
    nv.call("nvmlDeviceGetCount_v2", ctypes.byref(n))
    handles = [nv.handle(i) for i in range(n.value)]
    ht = handles[a.gpu]
    kmsg = Kmsg()
    stop = {"now": False}
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.__setitem__("now", True))

    log = open(a.log, "a", buffering=1)
    if log.tell() == 0:
        cols = ["time", "tripped", "gpu_mem_offset", "gpu_gpc_offset"]
        for i in range(len(handles)):
            cols += [f"g{i}_{k}" for k in ("pstate", "sm_mhz", "mem_mhz", "w", "limit_w", "c", "used_mib", "reasons")]
        log.write(",".join(cols) + "\n")

    def event(msg, marker=True):
        line = f"{now()} {msg}"
        if marker:  # the marker file exists only after a trip
            with open(a.marker, "a") as m:
                m.write(line + "\n")
        log.write(f"# {line}\n")

    event(f"watchdog start: pid {os.getpid()}, gpu {a.gpu}, {len(handles)} cards, temp-max {a.temp_max} C, "
          f"power-max {a.power_max} W for {a.power_secs} s, xid pattern {a.xid_pattern!r}, interval {a.interval} s",
          marker=False)
    over_since = None
    last_trip = 0.0
    trips = 0
    t_end = time.time() + a.duration if a.duration else None
    while not stop["now"]:
        t0 = time.time()
        reasons = []
        row = []
        for i, h in enumerate(handles):
            p_ = ctypes.c_int()
            ps = p_.value if nv.call("nvmlDeviceGetPerformanceState", h, ctypes.byref(p_)) == OK else None
            sm = nv.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(CLOCK_SM))
            mem = nv.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(CLOCK_MEM))
            mw = nv.uint("nvmlDeviceGetPowerUsage", h)
            lim = nv.uint("nvmlDeviceGetEnforcedPowerLimit", h)
            tc = nv.uint("nvmlDeviceGetTemperature", h, ctypes.c_int(0))
            mi = MemInfo()
            used = mi.used // (1 << 20) if nv.call("nvmlDeviceGetMemoryInfo", h, ctypes.byref(mi)) == OK else None
            ev = ctypes.c_ulonglong()
            evv = ev.value if nv.call("nvmlDeviceGetCurrentClocksEventReasons", h, ctypes.byref(ev)) == OK else None
            w = mw / 1000 if mw is not None else None
            row += [f"P{ps}" if ps is not None else "", sm, mem, f"{w:.2f}" if w is not None else "",
                    f"{lim / 1000:.0f}" if lim is not None else "", tc, used, f"0x{evv:x}" if evv is not None else ""]
            if tc is None or w is None:
                reasons.append(f"gpu {i} does not answer NVML (temperature {tc}, power {w})")
            elif tc > a.temp_max:
                reasons.append(f"gpu {i} at {tc} C > {a.temp_max} C")
            if i == a.gpu and w is not None:
                if w > a.power_max:
                    over_since = over_since or t0
                    if t0 - over_since >= a.power_secs:
                        reasons.append(f"gpu {i} above {a.power_max} W for {t0 - over_since:.1f} s (now {w:.1f} W)")
                else:
                    over_since = None
        for line in kmsg.new_lines():
            if a.xid_pattern in line:
                reasons.append(f"kernel log: {line}")
        mem_off, gpc_off = nv.vf(ht, "MemClk"), nv.vf(ht, "GpcClk")
        log.write(",".join(str(x) if x is not None else "" for x in [now(), trips, mem_off, gpc_off] + row) + "\n")
        if reasons and t0 - last_trip >= 5:
            trips += 1
            last_trip = t0
            back = nv.reset_offsets(ht)
            killed = kill_named(kill_names)
            event(f"TRIP {trips}: {'; '.join(reasons)} | gpu {a.gpu} offsets now mem {back[0]}, gpc {back[1]} | "
                  f"killed {', '.join(f'{c}[{p}]' for p, c in killed) or 'nothing'}")
        if a.stop and os.path.exists(a.stop):
            break
        if t_end and t0 >= t_end:
            break
        time.sleep(max(0.0, a.interval - (time.time() - t0)))
    log.write(f"# {now()} watchdog stop after {trips} trips\n")
    nv.call("nvmlShutdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
