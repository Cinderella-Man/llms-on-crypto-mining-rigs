#!/usr/bin/env python3
"""card_log_standin.py: runs scripts/card_log.py (or any byte-for-byte copy of it) on a stand-in for NVML, so that a
driver that starts the card log can be dry-run on a machine without NVIDIA cards: no card and no NVIDIA library is
touched. The stand-in answers the Get calls card_log.py makes, the same way scripts/card_log_test.sh's does:
  - STANDIN_CARDS cards (2), named P106-090, driver 580.178.04, limit 75 W, both clock offsets 0;
  - every card busy (P0, utilization 37 %, 31.64 W, 33 C, 4,692 MiB used) at 1,911 MHz, except card 1 from its 5th to
    its 16th sample: 1,354 MHz with the power-cap reason (a slow stretch scripts/card_table.py finds);
  - the memory clock STANDIN_MEM_MHZ (4006), or STANDIN_MEM_MHZ_SET (4304) while the file STANDIN_OFFSET_FILE exists
    (a stand-in for a memory offset that a driver sets);
  - STANDIN_FAIL=1: NVML does not load (card_log.py reports it and exits with status 2);
  - STANDIN_CALLS=FILE: the NVML functions called are written to FILE (JSON) at the end.

  python3 -I card_log_standin.py CARD_LOG_PY [card_log.py's arguments]
  python3 -I card_log_standin.py --shim DIR    # writes DIR/python3: a python3 for PATH that runs every card_log.py
                                               # it is asked to run on this stand-in, and anything else on the
                                               # python3 that wrote it

The exit status is card_log.py's.
"""
import ctypes
import importlib.util
import json
import os
import stat
import sys

calls = set()
samples = {}
CARDS = int(os.environ.get("STANDIN_CARDS", "2"))


class Lib:
    """A stand-in for libnvidia-ml.so.1: only Get calls, each recorded."""

    def __getattr__(self, name):
        raise AttributeError(name)  # a function the stand-in lacks: card_log.py must treat it as missing

    @staticmethod
    def nvmlErrorString(r):
        return b"stand-in error"  # a plain function: card_log.py sets its restype

    def _rec(self, name):
        calls.add(name)

    def nvmlInit_v2(self):
        self._rec("nvmlInit_v2")
        return 1 if os.environ.get("STANDIN_FAIL") == "1" else 0

    def nvmlShutdown(self):
        self._rec("nvmlShutdown")
        return 0

    def nvmlDeviceGetCount_v2(self, ref):
        self._rec("nvmlDeviceGetCount_v2")
        ref._obj.value = CARDS
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, i, ref):
        self._rec("nvmlDeviceGetHandleByIndex_v2")
        ref._obj.value = i.value + 1
        return 0

    def nvmlSystemGetDriverVersion(self, buf, size):
        self._rec("nvmlSystemGetDriverVersion")
        buf.value = b"580.178.04"
        return 0

    def nvmlDeviceGetName(self, h, buf, size):
        self._rec("nvmlDeviceGetName")
        buf.value = b"P106-090"
        return 0

    def nvmlDeviceGetPerformanceState(self, h, ref):
        self._rec("nvmlDeviceGetPerformanceState")
        ref._obj.value = 0
        return 0

    def nvmlDeviceGetClockInfo(self, h, clk, ref):
        self._rec("nvmlDeviceGetClockInfo")
        g = h.value - 1
        if clk.value == 1:  # the SM clock is asked first: one sample of this card
            samples[g] = samples.get(g, 0) + 1
            ref._obj.value = 1354 if g == 1 and 5 <= samples[g] <= 16 else 1911
        else:
            f = os.environ.get("STANDIN_OFFSET_FILE")
            on = bool(f) and os.path.exists(f)
            ref._obj.value = int(os.environ.get("STANDIN_MEM_MHZ_SET" if on else "STANDIN_MEM_MHZ", "4304" if on else "4006"))
        return 0

    def nvmlDeviceGetPowerUsage(self, h, ref):
        self._rec("nvmlDeviceGetPowerUsage")
        ref._obj.value = 31640
        return 0

    def nvmlDeviceGetTemperature(self, h, which, ref):
        self._rec("nvmlDeviceGetTemperature")
        ref._obj.value = 33
        return 0

    def nvmlDeviceGetMemoryInfo(self, h, ref):
        self._rec("nvmlDeviceGetMemoryInfo")
        ref._obj.used = 4692 << 20
        return 0

    def nvmlDeviceGetUtilizationRates(self, h, ref):
        self._rec("nvmlDeviceGetUtilizationRates")
        ref._obj.gpu = 37
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, h, ref):
        self._rec("nvmlDeviceGetCurrentClocksEventReasons")
        g = h.value - 1
        ref._obj.value = 0x4 if g == 1 and 5 <= samples.get(g, 0) <= 16 else 0
        return 0

    def nvmlDeviceGetEnforcedPowerLimit(self, h, ref):
        self._rec("nvmlDeviceGetEnforcedPowerLimit")
        ref._obj.value = 75000
        return 0

    def nvmlDeviceGetMemClkVfOffset(self, h, ref):
        self._rec("nvmlDeviceGetMemClkVfOffset")
        ref._obj.value = 0
        return 0

    def nvmlDeviceGetGpcClkVfOffset(self, h, ref):
        self._rec("nvmlDeviceGetGpcClkVfOffset")
        ref._obj.value = 0
        return 0


SHIM = """#!/usr/bin/env bash
# python3 for dry runs (scripts/card_log_standin.py --shim): card_log.py on the stand-in for NVML, anything else on
# the real python3
for ((i = 1; i <= $#; i++)); do
    case ${!i} in *card_log.py) exec %(py)s -I %(me)s "${@:i}" ;; esac
done
exec %(py)s "$@"
"""


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--shim":
        path = os.path.join(sys.argv[2], "python3")
        with open(path, "w") as f:
            f.write(SHIM % {"py": sys.executable, "me": os.path.abspath(__file__)})
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return 0
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    ctypes.CDLL = lambda *a, **k: Lib()
    sys.dont_write_bytecode = True  # no __pycache__ beside the card_log.py it runs
    spec = importlib.util.spec_from_file_location("card_log", sys.argv[1])
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    try:
        return cl.main()
    finally:
        if os.environ.get("STANDIN_CALLS"):
            with open(os.environ["STANDIN_CALLS"], "w") as f:
                json.dump(sorted(calls), f)


if __name__ == "__main__":
    sys.exit(main())
