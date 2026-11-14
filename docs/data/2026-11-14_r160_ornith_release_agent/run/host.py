# host.py RIGDIR RUN ARM: lowest MemAvailable (MiB, time), swap start/max (MiB, time), MemAvailable at the end, and each
# card's peak memory (MiB) from rig_server.sh's meminfo.log (epoch, MemAvailable kB, swap used kB) and gpu.csv.
import sys, time, collections
d, run, arm = sys.argv[1:4]
rows = [l.split() for l in open(d + "/meminfo.log") if l.strip() and l.split()[0].isdigit()]
rows = [(int(e), int(a) // 1024, int(s) // 1024) for e, a, s in rows]
t = lambda e: time.strftime("%H:%M:%S", time.gmtime(e + 3600))  # the rig's clock, CET
lo = min(rows, key=lambda r: r[1]); sw = max(rows, key=lambda r: r[2])
peak = collections.defaultdict(int)
for l in open(d + "/gpu.csv"):
    f = [x.strip() for x in l.split(",")]
    if len(f) >= 6 and f[5].endswith("MiB"):
        peak[int(f[1])] = max(peak[int(f[1])], int(f[5].split()[0]))
print("run\tarm\tlowest_memavailable_mib\tat\tswap_start_mib\tswap_max_mib\tswap_max_at\tmemavailable_end_mib\tcard_peak_mib_0_to_8")
print("\t".join([run, arm, str(lo[1]), t(lo[0]), str(rows[0][2]), str(sw[2]), t(sw[0]), str(rows[-1][1]),
                 ",".join(str(peak[i]) for i in sorted(peak))]))
