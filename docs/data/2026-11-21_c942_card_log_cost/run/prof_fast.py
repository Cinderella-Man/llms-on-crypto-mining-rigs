"""Today's sample against --fast and --slow-every 4, Python side only: NVML's Get calls are real ctypes foreign calls
into libc's abs() (handle NULL, so 0 = success), as in prof_today.py. One whole sample: 9 cards' fields, the line,
the write (line-buffered for today's, a 64 KiB buffer for --fast). Args: card_log.py cards samples"""
import ctypes, importlib.util, os, sys, time
sys.dont_write_bytecode = True
path, CARDS, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
libc = ctypes.CDLL(None)
GETS = ["nvmlDeviceGetPerformanceState", "nvmlDeviceGetClockInfo", "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetTemperature", "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
        "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetEnforcedPowerLimit",
        "nvmlDeviceGetMemClkVfOffset", "nvmlDeviceGetGpcClkVfOffset"]
class Lib:
    def __init__(self):
        for g in GETS:
            setattr(self, g, libc._FuncPtr(("abs", libc)))
        self.nvmlErrorString = libc._FuncPtr(("abs", libc))
    def nvmlInit_v2(self): return 0
    def nvmlShutdown(self): return 0
    def nvmlDeviceGetCount_v2(self, ref): ref._obj.value = CARDS; return 0
    def nvmlDeviceGetHandleByIndex_v2(self, i, ref): return 0
ctypes.CDLL = lambda *a, **k: Lib()
spec = importlib.util.spec_from_file_location("card_log", path)
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
nv = cl.Nvml()
hs = [nv.handle(i) for i in range(nv.count())]
def bench(name, sample, log):
    def whole():
        m0 = time.monotonic(); t0 = time.time()
        row = sample()
        ms = (time.monotonic() - m0) * 1000
        log.write(f"{t0:.3f},{cl.stamp(t0)},{ms:.1f}," + ",".join(row) + "\n")
        cl.alive(os.getpid()); os.path.exists("no.such.stop")
    whole()
    c0 = time.process_time()
    for _ in range(N):
        whole()
    print(f"{name:40s} {(time.process_time() - c0) / N * 1e6:7.1f} us CPU a sample")
bench("today's sample, line-buffered", lambda: sum((nv.sample(h) for h in hs), []), open("pf_today.csv", "w", buffering=1))
bench("--fast", cl.FastSample(nv, hs, 1).row, open("pf_fast.csv", "w", buffering=1 << 16))
bench("--slow-every 4", cl.FastSample(nv, hs, 4).row, open("pf_slow.csv", "w", buffering=1 << 16))
