#!/usr/bin/env python3
"""One GPU's memory (or core) clock offset through NVML, for the P106 memory-offset test (bench/mem_offset/README.md).

Talks to the driver's libnvidia-ml.so.1 through ctypes: nothing to install. GPUs are numbered as NVML numbers them, in
PCI bus order, the order of nvidia-smi and of CUDA_DEVICE_ORDER=PCI_BUS_ID.

  nvml_memoc.py probe [GPU ...]           read-only: driver, clocks, both offset APIs and their ranges, P0 clock limits,
                                          supported memory clocks, power, temperature, the PCIe link (lspci) and the
                                          CPU frequency settings (default: every GPU)
  nvml_memoc.py get GPU                   the current offset and its allowed range (raw NVML values)
  nvml_memoc.py clocks GPU                one line: P-state, SM and memory clocks, power, temperature, offset
  nvml_memoc.py units GPU -- LOAD...      fixes the offset's units: runs the command LOAD at offset 0 and at a raw
                                          +UNITS_RAW (--units-raw, default 100), samples the memory clock under it, puts
                                          the offset back to 0 and stores the factor in the state file
  nvml_memoc.py set GPU OFFSET            sets the offset in transfer-rate units (MT/s: +800 is +10 % of the P106-090's
                                          8,008 MT/s, which nvidia-smi shows as 4,004 MHz); needs the state file from
                                          `units`, or --raw to pass OFFSET to NVML unchanged
  nvml_memoc.py reset GPU                 offset 0 through every API that accepts it, read back

Options: --core works on the graphics clock instead of memory (offsets in MHz); --raw (set); --cap N refuses memory
offsets above N MT/s (default 1200); --state FILE (default memoc_units_gpu<N>.json next to this script).
set, units and reset need root. Exit status: 0 done, 1 NVML refused or a check failed, 2 usage.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_FUNCTION_NOT_FOUND = 13

CLOCK_GRAPHICS, CLOCK_SM, CLOCK_MEM = 0, 1, 2
PSTATE_0 = 0
STOCK_MEM_MTS = 8008  # the P106-090's GDDR5 rate; nvidia-smi reports half of it as the memory clock

EVENT_REASONS = {  # nvmlClocksEventReason* bits
    0x1: "idle", 0x2: "app-clocks", 0x4: "sw-power-cap", 0x8: "hw-slowdown", 0x10: "sync-boost",
    0x20: "sw-thermal", 0x40: "hw-thermal", 0x80: "hw-power-brake", 0x100: "display-clocks",
}


class ClockOffset(ctypes.Structure):  # nvmlClockOffset_v1_t
    _fields_ = [("version", ctypes.c_uint), ("type", ctypes.c_int), ("pstate", ctypes.c_int),
                ("clockOffsetMHz", ctypes.c_int), ("minClockOffsetMHz", ctypes.c_int),
                ("maxClockOffsetMHz", ctypes.c_int)]


CLOCK_OFFSET_V1 = ctypes.sizeof(ClockOffset) | (1 << 24)  # NVML_STRUCT_VERSION(ClockOffset, 1)


class PciInfo(ctypes.Structure):  # nvmlPciInfo_t
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


class Nvml:
    """Thin wrapper: every call returns (status, values); a symbol missing from the library counts as not found."""

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        r = self.lib.nvmlInit_v2()
        if r != NVML_SUCCESS:
            sys.exit(f"nvmlInit: {self.err(r)}")

    def err(self, r):
        if r == NVML_ERROR_FUNCTION_NOT_FOUND:
            return "not in this libnvidia-ml"
        return self.lib.nvmlErrorString(r).decode()

    def call(self, name, *args):
        f = getattr(self.lib, name, None)
        if f is None:
            return NVML_ERROR_FUNCTION_NOT_FOUND
        return f(*args)

    def handle(self, gpu):
        h = ctypes.c_void_p()
        r = self.call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(gpu), ctypes.byref(h))
        if r != NVML_SUCCESS:
            sys.exit(f"gpu {gpu}: {self.err(r)}")
        return h

    def uint(self, name, h, *args):
        v = ctypes.c_uint()
        r = self.call(name, h, *args, ctypes.byref(v))
        return r, v.value

    def int_pair(self, name, h):
        a, b = ctypes.c_int(), ctypes.c_int()
        r = self.call(name, h, ctypes.byref(a), ctypes.byref(b))
        return r, (a.value, b.value)

    def text(self, name, *args, size=96):
        buf = ctypes.create_string_buffer(size)
        r = self.call(name, *args, buf, ctypes.c_uint(size))
        return r, buf.value.decode(errors="replace")

    # the two offset APIs: the per-domain VF calls (Maxwell+, deprecated) and nvmlDevice{Get,Set}ClockOffsets
    def vf_get(self, h, dom):
        v = ctypes.c_int()
        r = self.call(f"nvmlDeviceGet{dom}VfOffset", h, ctypes.byref(v))
        if r != NVML_SUCCESS:
            return r, None
        r2, rng = self.int_pair(f"nvmlDeviceGet{dom}MinMaxVfOffset", h)
        return NVML_SUCCESS, (v.value, rng if r2 == NVML_SUCCESS else None)

    def vf_set(self, h, dom, raw):
        return self.call(f"nvmlDeviceSet{dom}VfOffset", h, ctypes.c_int(raw))

    def co_get(self, h, clock):
        info = ClockOffset(version=CLOCK_OFFSET_V1, type=clock, pstate=PSTATE_0)
        r = self.call("nvmlDeviceGetClockOffsets", h, ctypes.byref(info))
        if r != NVML_SUCCESS:
            return r, None
        return r, (info.clockOffsetMHz, (info.minClockOffsetMHz, info.maxClockOffsetMHz))

    def co_set(self, h, clock, raw):
        info = ClockOffset(version=CLOCK_OFFSET_V1, type=clock, pstate=PSTATE_0, clockOffsetMHz=raw)
        return self.call("nvmlDeviceSetClockOffsets", h, ctypes.byref(info))


def domain(args):
    return ("GpcClk", CLOCK_GRAPHICS) if args.core else ("MemClk", CLOCK_MEM)


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clocks_line(nv, gpu, h, args):
    dom, clock = domain(args)
    parts = [f"{ts()} gpu {gpu}"]
    p = ctypes.c_int()
    r = nv.call("nvmlDeviceGetPerformanceState", h, ctypes.byref(p))
    parts.append(f"P{p.value}" if r == NVML_SUCCESS else "P?")
    for label, c in (("sm", CLOCK_SM), ("mem", CLOCK_MEM)):
        r, v = nv.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(c))
        parts.append(f"{label} {v} MHz" if r == NVML_SUCCESS else f"{label} ?")
    r, mw = nv.uint("nvmlDeviceGetPowerUsage", h)
    parts.append(f"{mw / 1000:.1f} W" if r == NVML_SUCCESS else "? W")
    r, t = nv.uint("nvmlDeviceGetTemperature", h, ctypes.c_int(0))
    parts.append(f"{t} C" if r == NVML_SUCCESS else "? C")
    r, vf = nv.vf_get(h, dom)
    if r == NVML_SUCCESS:
        parts.append(f"{dom} vf offset {vf[0]}")
    r, co = nv.co_get(h, clock)
    if r == NVML_SUCCESS:
        parts.append(f"clock offsets P0 {co[0]}")
    return ", ".join(parts)


def show(label, r, value, nv):
    print(f"  {label:<34} {value if r == NVML_SUCCESS else '[' + nv.err(r) + ']'}")


def cmd_probe(nv, args):
    r, drv = nv.text("nvmlSystemGetDriverVersion", size=80)
    print(f"{ts()} driver {drv if r == NVML_SUCCESS else nv.err(r)}")
    n = ctypes.c_uint()
    nv.call("nvmlDeviceGetCount_v2", ctypes.byref(n))
    gpus = args.gpus or list(range(n.value))
    for gpu in gpus:
        h = nv.handle(gpu)
        r, name = nv.text("nvmlDeviceGetName", h)
        pci = PciInfo()
        rp = nv.call("nvmlDeviceGetPciInfo_v3", h, ctypes.byref(pci))
        bdf = f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.0" if rp == NVML_SUCCESS else None
        print(f"gpu {gpu}: {name}, {bdf or 'bus ?'}")
        print("  " + clocks_line(nv, gpu, h, args).split(", ", 1)[1])
        for label, c in (("max sm clock", CLOCK_SM), ("max mem clock", CLOCK_MEM)):
            r, v = nv.uint("nvmlDeviceGetMaxClockInfo", h, ctypes.c_int(c))
            show(label, r, f"{v} MHz", nv)
        lo, hi = ctypes.c_uint(), ctypes.c_uint()
        r = nv.call("nvmlDeviceGetMinMaxClockOfPState", h, ctypes.c_int(CLOCK_MEM), ctypes.c_int(PSTATE_0),
                    ctypes.byref(lo), ctypes.byref(hi))
        show("P0 mem clock min..max", r, f"{lo.value}..{hi.value} MHz", nv)
        for dom, clock in (("MemClk", CLOCK_MEM), ("GpcClk", CLOCK_GRAPHICS)):
            r, vf = nv.vf_get(h, dom)
            show(f"{dom} vf offset (range)", r, f"{vf[0]} ({vf[1][0]}..{vf[1][1]})" if vf and vf[1] else vf, nv)
            r, co = nv.co_get(h, clock)
            show(f"{'mem' if clock == CLOCK_MEM else 'graphics'} clock offsets P0 (range)", r,
                 f"{co[0]} ({co[1][0]}..{co[1][1]})" if co else None, nv)
        cnt = ctypes.c_uint(0)
        r = nv.call("nvmlDeviceGetSupportedMemoryClocks", h, ctypes.byref(cnt), None)
        if r in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE) and cnt.value:
            arr = (ctypes.c_uint * cnt.value)()
            r = nv.call("nvmlDeviceGetSupportedMemoryClocks", h, ctypes.byref(cnt), arr)
            show("supported memory clocks", r, ", ".join(f"{v} MHz" for v in arr[:cnt.value]), nv)
        else:
            show("supported memory clocks", r, None, nv)
        r, v = nv.uint("nvmlDeviceGetApplicationsClock", h, ctypes.c_int(CLOCK_MEM))
        show("application mem clock", r, f"{v} MHz", nv)
        r, v = nv.uint("nvmlDeviceGetPersistenceMode", h)
        show("persistence mode", r, "on" if v else "off", nv)
        lo, hi = ctypes.c_uint(), ctypes.c_uint()
        r = nv.call("nvmlDeviceGetPowerManagementLimitConstraints", h, ctypes.byref(lo), ctypes.byref(hi))
        show("power limit range", r, f"{lo.value / 1000:.0f}..{hi.value / 1000:.0f} W", nv)
        ev = ctypes.c_ulonglong()
        r = nv.call("nvmlDeviceGetCurrentClocksEventReasons", h, ctypes.byref(ev))
        show("clock event reasons", r, ", ".join(v for k, v in EVENT_REASONS.items() if ev.value & k) or "none", nv)
        if bdf and not args.no_system:
            out = subprocess.run(["lspci", "-vvv", "-s", bdf], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if any(k in line for k in ("LnkCap:", "LnkCap2:", "LnkSta:", "LnkCtl2:")):
                    print(f"  lspci {line.strip()}")
    print("  memory locked clocks (nvmlDeviceSetMemoryLockedClocks) are Ampere+ and application clocks only pick from")
    print("  the supported list above, so neither can raise the clock: the offset APIs are the only route")
    if not args.no_system:
        cpu = "/sys/devices/system/cpu"
        for f in ("cpu0/cpufreq/scaling_driver", "cpu0/cpufreq/scaling_governor", "cpu1/cpufreq/scaling_governor",
                  "cpu0/cpufreq/energy_performance_preference", "cpu1/cpufreq/energy_performance_preference"):
            try:
                print(f"cpu {f}: {open(os.path.join(cpu, f)).read().strip()}")
            except OSError as e:
                print(f"cpu {f}: [{e.strerror}]")
        i = 0
        while os.path.isdir(f"{cpu}/cpu0/cpuidle/state{i}"):
            d = f"{cpu}/cpu0/cpuidle/state{i}"
            vals = [open(f"{d}/{k}").read().strip() for k in ("name", "latency", "disable")]
            print(f"cpu cpu0 idle state{i}: {vals[0]}, exit latency {vals[1]} us, disabled {vals[2]}")
            i += 1
    return 0


def cmd_get(nv, args):
    h = nv.handle(args.gpu)
    dom, clock = domain(args)
    ok = False
    r, vf = nv.vf_get(h, dom)
    print(f"gpu {args.gpu} {dom} vf offset: " + (f"{vf[0]} (allowed {vf[1]})" if r == NVML_SUCCESS else nv.err(r)))
    ok |= r == NVML_SUCCESS
    r, co = nv.co_get(h, clock)
    print(f"gpu {args.gpu} clock offsets P0: " + (f"{co[0]} (allowed {co[1]})" if r == NVML_SUCCESS else nv.err(r)))
    ok |= r == NVML_SUCCESS
    return 0 if ok else 1


def state_path(args, gpu):
    if args.state:
        return args.state
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"memoc_units_gpu{gpu}.json")


def set_raw(nv, h, args, raw, api=None):
    """Sets a raw offset, the VF call first and nvmlDeviceSetClockOffsets if that is refused; returns the API used."""
    dom, clock = domain(args)
    tried = []
    for name in ([api] if api else ["vf", "offsets"]):
        r = nv.vf_set(h, dom, raw) if name == "vf" else nv.co_set(h, clock, raw)
        if r == NVML_SUCCESS:
            rr, cur = nv.vf_get(h, dom) if name == "vf" else nv.co_get(h, clock)
            if rr == NVML_SUCCESS and cur[0] != raw:
                print(f"  {name}: set {raw} but reads back {cur[0]}")
                return None
            return name
        tried.append(f"{name}: {nv.err(r)}")
    print("  refused: " + "; ".join(tried))
    return None


def cmd_set(nv, args):
    h = nv.handle(args.gpu)
    dom, clock = domain(args)
    if args.raw or args.core:
        raw, api = args.offset, None
        desc = f"raw {raw}" if args.raw else f"{raw} MHz"
    else:
        try:
            st = json.load(open(state_path(args, args.gpu)))
        except OSError:
            print(f"no units for gpu {args.gpu}: run `units` first, or pass --raw", file=sys.stderr)
            return 2
        if args.offset > args.cap:
            print(f"{args.offset} MT/s is above --cap {args.cap}", file=sys.stderr)
            return 2
        raw, api = round(args.offset * st["raw_per_mts"]), st["api"]
        desc = f"{args.offset:+d} MT/s = raw {raw} through {api}"
    print(f"before: {clocks_line(nv, args.gpu, h, args)}")
    used = set_raw(nv, h, args, raw, api)
    print(f"after:  {clocks_line(nv, args.gpu, h, args)}")
    if used is None:
        print(f"gpu {args.gpu} {dom}: setting {desc} failed")
        return 1
    print(f"gpu {args.gpu} {dom}: set {desc} (api {used})")
    return 0


def cmd_reset(nv, args):
    h = nv.handle(args.gpu)
    dom, clock = domain(args)
    print(f"before: {clocks_line(nv, args.gpu, h, args)}")
    bad = False
    for name in ("vf", "offsets"):
        get = (lambda: nv.vf_get(h, dom)) if name == "vf" else (lambda: nv.co_get(h, clock))
        r, cur = get()
        if r != NVML_SUCCESS:
            continue  # this API doesn't work on the card; nothing set through it either
        if cur[0] != 0:
            r = nv.vf_set(h, dom, 0) if name == "vf" else nv.co_set(h, clock, 0)
            r2, cur = get()
            if r != NVML_SUCCESS or r2 != NVML_SUCCESS or cur[0] != 0:
                print(f"  {name}: reset to 0 failed ({nv.err(r)}), reads {cur[0] if cur else '?'}")
                bad = True
    print(f"after:  {clocks_line(nv, args.gpu, h, args)}")
    print(f"gpu {args.gpu} {dom}: " + ("NOT back at 0" if bad else "offset 0"))
    return 1 if bad else 0


def sample_under_load(nv, h, load, interval=0.05):
    """Runs the load command and samples the memory clock and P-state; returns the P0 samples' maximum and count."""
    samples, done = [], threading.Event()

    def sampler():
        while not done.is_set():
            p, c = ctypes.c_int(), ctypes.c_uint()
            if (nv.call("nvmlDeviceGetPerformanceState", h, ctypes.byref(p)) == NVML_SUCCESS
                    and nv.call("nvmlDeviceGetClockInfo", h, ctypes.c_int(CLOCK_MEM), ctypes.byref(c)) == NVML_SUCCESS
                    and p.value == 0):
                samples.append(c.value)
            time.sleep(interval)

    t = threading.Thread(target=sampler)
    t.start()
    status = subprocess.run(load, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    done.set()
    t.join()
    return (max(samples) if samples else None), len(samples), status


def cmd_units(nv, args):
    h = nv.handle(args.gpu)
    if not args.load:
        print("units needs a load command after --", file=sys.stderr)
        return 2
    print(f"before: {clocks_line(nv, args.gpu, h, args)}")
    base, n0, s0 = sample_under_load(nv, h, args.load)
    print(f"  offset 0: memory clock under load {base} MHz ({n0} P0 samples, load exit {s0})")
    api = set_raw(nv, h, args, args.units_raw)
    if api is None:
        print("  the card refuses memory offsets: route B (Coolbits) in hardware_tuning.md 1.3 is what is left")
        return 1
    try:
        high, n1, s1 = sample_under_load(nv, h, args.load)
    finally:
        cmd_reset(nv, args)
    print(f"  raw +{args.units_raw} via {api}: memory clock under load {high} MHz ({n1} P0 samples, load exit {s1})")
    if not base or not high or s0 != 0 or s1 != 0:
        print("  no P0 samples or the load failed: units unknown")
        return 1
    ratio = (high - base) / args.units_raw
    # nvidia-smi's memory MHz are half the transfer rate: a raw unit of 1 MT/s moves the reported clock by 0.5 MHz
    if 0.35 <= ratio <= 0.65:
        raw_per_mts, unit = 1.0, "MT/s (transfer-rate units)"
    elif 0.8 <= ratio <= 1.2:
        raw_per_mts, unit = 0.5, "reported MHz (2 MT/s each)"
    else:
        print(f"  clock moved {high - base} MHz for raw {args.units_raw} (ratio {ratio:.2f}): units unknown, "
              "use --raw and read clocks.mem under load")
        return 1
    st = {"gpu": args.gpu, "api": api, "raw_per_mts": raw_per_mts, "unit": unit, "base_mhz": base,
          "raw": args.units_raw, "loaded_mhz": high, "measured": ts()}
    json.dump(st, open(state_path(args, args.gpu), "w"), indent=1)
    print(f"  NVML raw unit: {unit}; stored in {state_path(args, args.gpu)}")
    return 0


def cmd_clocks(nv, args):
    print(clocks_line(nv, args.gpu, nv.handle(args.gpu), args))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", action="store_true", help="graphics clock instead of memory")
    p.add_argument("--state", help="units state file")
    p.add_argument("--cap", type=int, default=1200, help="highest memory offset accepted, MT/s")
    p.add_argument("--no-system", action="store_true", help="probe: skip lspci and the CPU settings")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probe"); s.add_argument("gpus", type=int, nargs="*")
    for c in ("get", "clocks", "reset"):
        sub.add_parser(c).add_argument("gpu", type=int)
    s = sub.add_parser("set"); s.add_argument("gpu", type=int); s.add_argument("offset", type=int)
    s.add_argument("--raw", action="store_true", help="pass the offset to NVML unchanged")
    s = sub.add_parser("units"); s.add_argument("gpu", type=int)
    s.add_argument("--units-raw", type=int, default=100, help="raw offset for the measurement")
    s.add_argument("load", nargs=argparse.REMAINDER, help="-- command that keeps the card busy for a few seconds")
    args = p.parse_args()
    if getattr(args, "load", None) and args.load[0] == "--":
        args.load = args.load[1:]
    if args.cmd in ("set", "units", "reset") and os.geteuid() != 0:
        print(f"{args.cmd} needs root", file=sys.stderr)
        return 2
    nv = Nvml()
    try:
        return {"probe": cmd_probe, "get": cmd_get, "clocks": cmd_clocks, "set": cmd_set, "reset": cmd_reset,
                "units": cmd_units}[args.cmd](nv, args)
    finally:
        nv.call("nvmlShutdown")


if __name__ == "__main__":
    sys.exit(main())
