"""Where today's card_log.py sample spends the host's CPU, outside NVML itself.
NVML's Get calls are replaced by real ctypes foreign calls into libc's abs() (a do-nothing C function: the handle is
NULL, so it returns 0 = NVML_SUCCESS), so every Python and ctypes cost of a real call is paid and NVML's own is not.
Args: card_log.py cards samples"""
import ctypes, importlib.util, io, os, sys, time
sys.dont_write_bytecode = True
path, CARDS, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
real_cdll = ctypes.CDLL
libc = real_cdll(None)
GETS = ["nvmlDeviceGetPerformanceState", "nvmlDeviceGetClockInfo", "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetTemperature", "nvmlDeviceGetMemoryInfo", "nvmlDeviceGetUtilizationRates",
        "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetEnforcedPowerLimit",
        "nvmlDeviceGetMemClkVfOffset", "nvmlDeviceGetGpcClkVfOffset"]
class Lib:
    def __init__(self):
        for g in GETS:  # a fresh foreign function object per name, as CDLL makes one per name
            setattr(self, g, libc._FuncPtr(("abs", libc)))
        self.nvmlErrorString = libc._FuncPtr(("abs", libc))
    def nvmlInit_v2(self): return 0
    def nvmlShutdown(self): return 0
    def nvmlDeviceGetCount_v2(self, ref): ref._obj.value = CARDS; return 0
    def nvmlDeviceGetHandleByIndex_v2(self, i, ref): return 0  # handle stays NULL: abs(NULL) = 0
    def nvmlSystemGetDriverVersion(self, buf, n): buf.value = b"580.178.04"; return 0
    def nvmlDeviceGetName(self, h, buf, n): buf.value = b"P106-090"; return 0
ctypes.CDLL = lambda *a, **k: Lib()
spec = importlib.util.spec_from_file_location("card_log", path)
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
nv = cl.Nvml()
hs = [nv.handle(i) for i in range(nv.count())]
clock = time.process_time
def per(fn, n=N):
    fn()
    c0 = clock(); w0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (clock() - c0) / n * 1e6, (time.perf_counter() - w0) / n * 1e6
res = {}
# 1. one whole sample as run() does it: 8 calls x cards, the row, the line, a line-buffered write to a file
log = open(os.devnull if os.environ.get("DEVNULL") else "prof_today.csv", "w", buffering=1)
def whole():
    m0 = time.monotonic(); t0 = time.time(); row = []
    for h in hs:
        row += nv.sample(h)
    ms = (time.monotonic() - m0) * 1000
    log.write(f"{t0:.3f},{cl.stamp(t0)},{ms:.1f}," + ",".join(row) + "\n")
    cl.alive(os.getpid()); os.path.exists("no.such.stop")
res["whole sample (today's run loop body)"] = per(whole)
res["  the 9 cards' nv.sample() alone"] = per(lambda: [nv.sample(h) for h in hs])
row = sum((nv.sample(h) for h in hs), [])
t0 = time.time()
res["  the line: stamp + format + join"] = per(lambda: f"{t0:.3f},{cl.stamp(t0)},{1.0:.1f}," + ",".join(row) + "\n")
res["    stamp() alone"] = per(lambda: cl.stamp(t0))
line = f"{t0:.3f},{cl.stamp(t0)},{1.0:.1f}," + ",".join(row) + "\n"
res["  write, line-buffered (one write() a row)"] = per(lambda: log.write(line))
blog = open("prof_today_b.csv", "w", buffering=1 << 16)
res["  write, 64 KiB buffer"] = per(lambda: blog.write(line))
res["  parent check + stop-file check"] = per(lambda: (cl.alive(os.getpid()), os.path.exists("no.such.stop")))
# 2. one call of each kind, as Nvml.sample makes it (getattr per call, fresh ctypes objects, byref)
h = hs[0]
res["  per call: sint(PerformanceState)"] = per(lambda: nv.sint("nvmlDeviceGetPerformanceState", h), N * 9)
res["  per call: uint(ClockInfo, c_int(1))"] = per(lambda: nv.uint("nvmlDeviceGetClockInfo", h, ctypes.c_int(1)), N * 9)
res["  per call: uint(PowerUsage)"] = per(lambda: nv.uint("nvmlDeviceGetPowerUsage", h), N * 9)
def meminfo():
    mi = cl.MemInfo()
    return mi.used >> 20 if nv.call("nvmlDeviceGetMemoryInfo", h, ctypes.byref(mi)) == 0 else None
res["  per call: MemoryInfo struct"] = per(meminfo, N * 9)
f = nv.lib.nvmlDeviceGetPowerUsage; v = ctypes.c_uint(); bv = ctypes.byref(v)
res["  bare foreign call f(h, byref) with prebuilt args"] = per(lambda: f(h, bv), N * 9)
res["  python3 function call overhead (lambda: None)"] = per(lambda: None, N * 9)
for k, (c, w) in res.items():
    print(f"{k:52s} {c:9.1f} us CPU  {w:9.1f} us wall")
