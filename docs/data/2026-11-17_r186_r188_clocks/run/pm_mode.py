#!/usr/bin/env python3
"""pm_mode.py (R188, Q1553): read or set each card's PowerMizer mode through NVML, with no X server.

  python3 -I pm_mode.py                 read every card: mode, supported modes, P-state, SM clock
  CARDS=0,3 MODE=0 python3 -I pm_mode.py   set those cards to MODE, read back, then read every card
  HOLD_S=600 ... python3 -I pm_mode.py     after setting, keep NVML open HOLD_S seconds (or until SIGTERM), so that
                                           with persistence mode off the cards are not torn down before a CUDA
                                           process opens them

Modes (nvml.h): 0 adaptive, 1 prefer maximum performance, 2 auto, 3 prefer consistent performance (lock to base).
The struct is nvmlDevicePowerMizerModes_v1_t from the nvml.h that nvidia-settings 580.95.05 ships.
"""
import ctypes
import os
import signal
import sys
import time


class PowerMizerModes(ctypes.Structure):
    _fields_ = [("currentMode", ctypes.c_uint), ("mode", ctypes.c_uint), ("supportedPowerMizerModes", ctypes.c_uint)]


nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvml.nvmlErrorString.restype = ctypes.c_char_p
if nvml.nvmlInit_v2():
    sys.exit("nvmlInit_v2 failed")
n = ctypes.c_uint()
nvml.nvmlDeviceGetCount_v2(ctypes.byref(n))


def handle(i):
    h = ctypes.c_void_p()
    nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
    return h


def read(i):
    h, m, u, p = handle(i), PowerMizerModes(), ctypes.c_uint(), ctypes.c_int()
    rc = nvml.nvmlDeviceGetPowerMizerMode_v1(h, ctypes.byref(m))
    nvml.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(u))
    nvml.nvmlDeviceGetPerformanceState(h, ctypes.byref(p))
    print(f"{time.time():.3f}\tcard {i}\tmode={m.currentMode if rc == 0 else 'rc' + str(rc)}"
          f"\tsupported=0b{m.supportedPowerMizerModes:04b}\tP{p.value}\tsm={u.value} MHz", flush=True)


if "MODE" in os.environ:
    mode = int(os.environ["MODE"])
    for i in [int(c) for c in os.environ["CARDS"].split(",")]:
        m = PowerMizerModes()
        m.mode = mode
        rc = nvml.nvmlDeviceSetPowerMizerMode_v1(handle(i), ctypes.byref(m))
        print(f"{time.time():.3f}\tcard {i}\tset mode {mode}: {nvml.nvmlErrorString(rc).decode()}", flush=True)
for i in range(n.value):
    read(i)
hold = float(os.environ.get("HOLD_S", "0"))
if hold > 0:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"{time.time():.3f}\tholding NVML open for {hold:.0f} s", flush=True)
    time.sleep(hold)
nvml.nvmlShutdown()
