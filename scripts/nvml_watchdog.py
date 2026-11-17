#!/usr/bin/env python3
"""A safety watchdog for clock-offset tests on NVIDIA cards, through NVML (ctypes on libnvidia-ml.so.1, no install).

Samples every card every INTERVAL seconds (temperature, power, power limit, SM and memory clocks, P-state, memory used,
clock event reasons, and the memory and graphics clock offsets of the cards under test) and logs every sample to a CSV
file. It reads the kernel log (/dev/kmsg) as it grows. It trips on any of:
  - any card above --temp-max degrees C;
  - a card under test above --power-max W for --power-secs seconds in a row;
  - a new kernel log line that contains --xid-pattern (default "NVRM: Xid").
On a trip it at once sets both clock offsets of every card under test back to 0 (both NVML offset APIs), reads them
back, kills every process whose name is in --kill (TERM, then KILL after 2 s), and appends a line to the --marker
file. It keeps sampling afterwards; a condition that persists trips again, at most once every 5 s.

The cards under test are --gpu (one card), or --gpus (a comma-separated list, or "all"). With --gpus the log gets two
more columns per card under test at the end of each row, g<N>_mem_off and g<N>_gpc_off.

With --per-card-xid (bench/mem_offset/climb.py uses it), an Xid line that names a card under test by its PCI address
("NVRM: Xid (PCI:0000:05:00): 31, ...") trips that card only: its two offsets go to 0 and are read back, the processes
listed in --card-pids DIR/<card>.pids and the process the line names (pid=N) are killed, and a line "CARD <card> TRIP"
goes to --card-marker (not to --marker). The other cards keep their offsets and loads. Heat, power, a card that does
not answer NVML, and an Xid line that names no card under test still trip every card.

With --ledger FILE, every offset it sets back to 0 from a non-zero value is appended to FILE with a tag of the current
boot (a hash of /proc/sys/kernel/random/boot_id), so that a later run can tell that an offset stepped down since the
last boot (Q1563: a card stays slow after a step down until a reboot).

While it runs it also holds every card open, so a power limit or an offset set by a short-lived process stays in
place with persistence mode off. Stop it with SIGTERM or SIGINT, or by creating the --stop file.

  nvml_watchdog.py (--gpu 0 | --gpus all) --log wd.csv --marker TRIPPED [--temp-max 70] [--power-max 55] [--power-secs 5]
                   [--interval 0.5] [--xid-pattern 'NVRM: Xid'] [--kill mmvq_pascal,cuda_memtest,llama-server,...]
                   [--stop STOPFILE] [--duration SECONDS]
                   [--per-card-xid --card-pids DIR --card-marker FILE] [--ledger FILE]

Needs root (the kernel log and the offset calls). Exit status 0 on a clean stop, 2 on a usage or NVML error.
"""
import argparse
import ctypes
import errno
import hashlib
import os
import re
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


class PciInfo(ctypes.Structure):  # nvmlPciInfo_t
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


PCI_RE = re.compile(r"\(PCI:([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})(?:\.[0-7])?\)")


def xid_card(line, bus_of):
    """The card a kernel-log line names by PCI address, from {card: (domain, bus, device)}; None if it names none."""
    m = PCI_RE.search(line)
    if not m:
        return None
    key = tuple(int(v, 16) for v in m.groups())
    for g, k in bus_of.items():
        if k == key:
            return g
    return None


def xid_pid(line):
    m = re.search(r"\bpid=(\d+)", line)
    return int(m.group(1)) if m else None


def boot_tag():
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return hashlib.sha256(f.read().strip().encode()).hexdigest()[:12]
    except OSError:
        return "unknown"


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

    def pci(self, h):
        p = PciInfo()
        return (p.domain, p.bus, p.device) if self.call("nvmlDeviceGetPciInfo_v3", h, ctypes.byref(p)) == OK else None

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


def comm_of(pid):
    with open(f"/proc/{pid}/comm") as f:
        return f.read().strip()


def kill_pids(pids):
    """TERM, then KILL after 2 s to each pid still running the same program; returns [(pid, comm)] signalled."""
    sent = []
    for pid in sorted(set(pids)):
        if pid == os.getpid():
            continue
        try:
            comm = comm_of(pid)
            os.kill(pid, signal.SIGTERM)
            sent.append((pid, comm))
        except OSError:
            pass
    if sent:
        time.sleep(2)
        for pid, comm in sent:
            try:
                if comm_of(pid) == comm:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return sent


def card_pids(directory, card):
    try:
        with open(os.path.join(directory, f"{card}.pids")) as f:
            return [int(v) for v in f.read().split() if v.isdigit()]
    except OSError:
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0, help="the card under test (NVML index, PCI bus order)")
    p.add_argument("--gpus", help='several cards under test: a comma-separated list of NVML indices, or "all"')
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
    p.add_argument("--per-card-xid", action="store_true",
                   help="an Xid line naming a card under test by PCI address trips that card only")
    p.add_argument("--card-pids", help="folder of <card>.pids files: the processes a card trip kills")
    p.add_argument("--card-marker", help="file that gets one line per card trip (default: the --marker file + .cards)")
    p.add_argument("--ledger", help="append every offset set back to 0 from a non-zero value, tagged with the boot")
    a = p.parse_args()
    if os.geteuid() != 0:
        print("needs root", file=sys.stderr)
        return 2
    kill_names = set(x for x in a.kill.split(",") if x)
    nv = Nvml()
    n = ctypes.c_uint()
    nv.call("nvmlDeviceGetCount_v2", ctypes.byref(n))
    handles = [nv.handle(i) for i in range(n.value)]
    if a.gpus:
        tested = list(range(len(handles))) if a.gpus == "all" else [int(x) for x in a.gpus.split(",")]
    else:
        tested = [a.gpu]
    if not tested or any(g < 0 or g >= len(handles) for g in tested):
        print(f"cards under test {tested}: not all among the {len(handles)} cards", file=sys.stderr)
        return 2
    ht = handles[tested[0]]
    bus_of = {i: nv.pci(handles[i]) for i in tested}
    card_marker = a.card_marker or a.marker + ".cards"
    boot = boot_tag()
    kmsg = Kmsg()
    stop = {"now": False}
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.__setitem__("now", True))

    log = open(a.log, "a", buffering=1)
    if log.tell() == 0:
        cols = ["time", "tripped", "gpu_mem_offset", "gpu_gpc_offset"]
        for i in range(len(handles)):
            cols += [f"g{i}_{k}" for k in ("pstate", "sm_mhz", "mem_mhz", "w", "limit_w", "c", "used_mib", "reasons")]
        if a.gpus:
            for i in tested:
                cols += [f"g{i}_mem_off", f"g{i}_gpc_off"]
        log.write(",".join(cols) + "\n")

    def event(msg, marker=True):
        line = f"{now()} {msg}"
        if marker:  # the marker file exists only after a trip
            with open(a.marker, "a") as m:
                m.write(line + "\n")
        log.write(f"# {line}\n")

    event(f"watchdog start: pid {os.getpid()}, gpu {','.join(map(str, tested))}, {len(handles)} cards, temp-max {a.temp_max} C, "
          f"power-max {a.power_max} W for {a.power_secs} s, xid pattern {a.xid_pattern!r}, interval {a.interval} s"
          + (", per-card xid" if a.per_card_xid else ""), marker=False)

    def reset_card(i, why):
        """Both offsets of card i to 0 and read back; a step down from a non-zero value goes to the ledger."""
        before = (nv.vf(handles[i], "MemClk"), nv.vf(handles[i], "GpcClk"))
        after = nv.reset_offsets(handles[i])
        if a.ledger and any(before):
            with open(a.ledger, "a") as f:
                f.write(f"{boot} {time.strftime('%Y-%m-%d %H:%M:%S')} card {i} mem {before[0]}->{after[0]} "
                        f"core {before[1]}->{after[1]} watchdog: {why}\n")
        return after

    last_card_trip = {}
    over_since = {i: None for i in tested}
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
            if i in over_since and w is not None:
                if w > a.power_max:
                    over_since[i] = over_since[i] or t0
                    if t0 - over_since[i] >= a.power_secs:
                        reasons.append(f"gpu {i} above {a.power_max} W for {t0 - over_since[i]:.1f} s (now {w:.1f} W)")
                else:
                    over_since[i] = None
        card_trips = []
        for line in kmsg.new_lines():
            if a.xid_pattern in line:
                g = xid_card(line, bus_of) if a.per_card_xid else None
                if g is None:
                    reasons.append(f"kernel log: {line}")
                else:
                    card_trips.append((g, line))
        mem_off, gpc_off = nv.vf(ht, "MemClk"), nv.vf(ht, "GpcClk")
        if a.gpus:
            for i in tested:
                row += [nv.vf(handles[i], "MemClk"), nv.vf(handles[i], "GpcClk")]
        log.write(",".join(str(x) if x is not None else "" for x in [now(), trips, mem_off, gpc_off] + row) + "\n")
        for g, line in card_trips:
            if reasons or t0 - last_card_trip.get(g, 0.0) < 5:
                continue  # a global trip below covers it; or this card tripped under 5 s ago
            last_card_trip[g] = t0
            m, c = reset_card(g, f"card trip: {line}")
            killed = kill_pids(card_pids(a.card_pids, g) + ([xid_pid(line)] if xid_pid(line) else [])) \
                if a.card_pids or xid_pid(line) else []
            msg = (f"{now()} CARD {g} TRIP: kernel log: {line} | offsets now gpu {g} mem {m}, gpc {c} | "
                   f"killed {', '.join(f'{cm}[{p}]' for p, cm in killed) or 'nothing'}")
            with open(card_marker, "a") as f:
                f.write(msg + "\n")
            log.write(f"# {msg}\n")
        if reasons and t0 - last_trip >= 5:
            trips += 1
            last_trip = t0
            back = [(i, reset_card(i, "; ".join(reasons))) for i in tested]
            killed = kill_named(kill_names)
            offs = "; ".join(f"gpu {i} mem {m}, gpc {g}" for i, (m, g) in back)
            event(f"TRIP {trips}: {'; '.join(reasons)} | offsets now {offs} | "
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
