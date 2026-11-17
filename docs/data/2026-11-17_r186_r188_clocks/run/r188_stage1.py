#!/usr/bin/env python3
"""r188_stage1.py (R188, Q1553): the driver's three run-time clock controls, tried on one idle card.

Run on the rig as root, with nothing else on the cards:  CARD=0 python3 -I r188_stage1.py > stage1.log
Steps, seconds each, every one undone before the next:
  (a) locked clocks:          nvidia-smi -i CARD -lgc 1911,1911, undone with -rgc
  (b) the CUDA clock override: nvidia-smi -i CARD -cc 1, undone with -cc 0
  (c) the PowerMizer mode through NVML (nvmlDeviceSetPowerMizerMode_v1), no X server: mode 1, then mode 3, each read
      back, undone with mode 2 (the default, auto)
The script holds NVML open for its whole run. With persistence mode off and no X server, a card that no process holds
is torn down, which could drop a setting before it is read; this process holds it. After each command it reads the
card SAMPLES (6) times every SAMPLE_S (0.5) s, through NVML and through nvidia-smi: SM and memory clock, P-state,
power and the active clock-event (throttle) reasons. Ends with every setting at its default and checks it.
Environment: CARD (0), SAMPLES (6), SAMPLE_S (0.5), SMI (nvidia-smi).
"""
import ctypes
import os
import subprocess
import sys
import time

CARD = int(os.environ.get("CARD", "0"))
SAMPLES = int(os.environ.get("SAMPLES", "6"))
SAMPLE_S = float(os.environ.get("SAMPLE_S", "0.5"))
SMI = os.environ.get("SMI", "nvidia-smi")

# NVML's clock-event reason bits (nvml.h, nvmlClocksEventReason*)
REASONS = [(0x1, "GpuIdle"), (0x2, "ApplicationsClocksSetting"), (0x4, "SwPowerCap"), (0x8, "HwSlowdown"),
           (0x10, "SyncBoost"), (0x20, "SwThermalSlowdown"), (0x40, "HwThermalSlowdown"),
           (0x80, "HwPowerBrakeSlowdown"), (0x100, "DisplayClockSetting")]
MODES = {0: "adaptive", 1: "prefer maximum performance", 2: "auto (default)", 3: "prefer consistent performance"}


class PowerMizerModes(ctypes.Structure):
    # nvmlDevicePowerMizerModes_v1_t, as in the nvml.h that nvidia-settings 580.95.05 ships
    _fields_ = [("currentMode", ctypes.c_uint), ("mode", ctypes.c_uint), ("supportedPowerMizerModes", ctypes.c_uint)]


nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvml.nvmlErrorString.restype = ctypes.c_char_p


def err(rc):
    return f"{rc} ({nvml.nvmlErrorString(rc).decode()})"


def log(msg):
    print(f"{time.time():.3f}\t{msg}", flush=True)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().replace("\n", " | ")
    log(f"CMD {' '.join(cmd)} -> rc={p.returncode}: {out}")
    return p.returncode


rc = nvml.nvmlInit_v2()
if rc:
    sys.exit(f"nvmlInit_v2: {err(rc)}")
h = ctypes.c_void_p()
rc = nvml.nvmlDeviceGetHandleByIndex_v2(CARD, ctypes.byref(h))
if rc:
    sys.exit(f"handle {CARD}: {err(rc)}")


def state(tag):
    u, p = ctypes.c_uint(), ctypes.c_int()
    sm = mem = pw = ps = None
    if nvml.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(u)) == 0:
        sm = u.value
    if nvml.nvmlDeviceGetClockInfo(h, 2, ctypes.byref(u)) == 0:
        mem = u.value
    if nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(u)) == 0:
        pw = u.value / 1000
    if nvml.nvmlDeviceGetPerformanceState(h, ctypes.byref(p)) == 0:
        ps = p.value
    r = ctypes.c_ulonglong()
    rr = nvml.nvmlDeviceGetCurrentClocksEventReasons(h, ctypes.byref(r))
    names = ",".join(n for b, n in REASONS if r.value & b) if rr == 0 else f"err {rr}"
    smi = subprocess.run([SMI, "-i", str(CARD), "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,"
                          "clocks_event_reasons.active", "--format=csv,noheader"], capture_output=True, text=True)
    log(f"STATE\t{tag}\tnvml: P{ps} sm={sm} MHz mem={mem} MHz {pw} W reasons=0x{r.value:x}[{names or 'none'}]"
        f"\tsmi: {smi.stdout.strip()}")


def samples(tag):
    for i in range(SAMPLES):
        state(f"{tag}#{i}")
        time.sleep(SAMPLE_S)


def pm_get(tag):
    m = PowerMizerModes()
    rc = nvml.nvmlDeviceGetPowerMizerMode_v1(h, ctypes.byref(m))
    log(f"PM_GET\t{tag}\trc={err(rc)} current={m.currentMode} ({MODES.get(m.currentMode, '?')}) "
        f"supported=0b{m.supportedPowerMizerModes:04b}")
    return rc, m.currentMode


def pm_set(mode, tag):
    m = PowerMizerModes()
    m.mode = mode
    rc = nvml.nvmlDeviceSetPowerMizerMode_v1(h, ctypes.byref(m))
    log(f"PM_SET\t{tag}\tmode={mode} ({MODES[mode]}) rc={err(rc)}")
    return rc


log(f"START card {CARD}; driver query:")
run([SMI, "-i", str(CARD), "--query-gpu=driver_version,name,persistence_mode,vbios_version,clocks.max.sm,"
     "clocks.max.mem,clocks.applications.graphics,clocks.default_applications.graphics", "--format=csv"])
run([SMI, "-i", str(CARD), "-q", "-d", "CLOCK,PERFORMANCE"])
pm_get("before")
samples("before")

log("(a) locked clocks")
run([SMI, "-i", str(CARD), "-lgc", "1911,1911"])
samples("a_set")
run([SMI, "-i", str(CARD), "-q", "-d", "CLOCK,PERFORMANCE"])
run([SMI, "-i", str(CARD), "-rgc"])
samples("a_undone")

log("(b) CUDA clock override")
run([SMI, "-i", str(CARD), "-cc", "1"])
samples("b_set")
run([SMI, "-i", str(CARD), "-q", "-d", "CLOCK,PERFORMANCE"])
run([SMI, "-i", str(CARD), "-cc", "0"])
samples("b_undone")

log("(c) PowerMizer mode through NVML")
for mode in (1, 3):
    pm_set(mode, f"c_mode{mode}")
    pm_get(f"c_mode{mode}_readback")
    samples(f"c_mode{mode}")
    run([SMI, "-i", str(CARD), "-q", "-d", "CLOCK,PERFORMANCE"])
pm_set(2, "c_undo")
pm_get("c_undo_readback")
samples("c_undone")

log("END: settings back to default")
pm_get("end")
samples("end")
nvml.nvmlShutdown()
