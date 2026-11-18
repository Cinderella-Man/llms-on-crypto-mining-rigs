# host_cards.py RIGDIR RUN ARM: R166's run/host.py for a run whose card log is scripts/card_log.py's cards.csv (one row
# a sample, g<i>_used_mib per card) instead of gpu.csv: lowest MemAvailable (MiB, time), swap start/max (MiB, time),
# MemAvailable at the end, and each card's peak memory (MiB), from rig_server.sh's meminfo.log and cards.csv.
import csv, sys, time, collections
d, run, arm = sys.argv[1:4]
rows = [l.split() for l in open(d + "/meminfo.log") if l.strip() and l.split()[0].isdigit()]
rows = [(int(e), int(a) // 1024, int(s) // 1024) for e, a, s in rows]
t = lambda e: time.strftime("%H:%M:%S", time.gmtime(e + 3600))  # the rig's clock, CET
lo = min(rows, key=lambda r: r[1]); sw = max(rows, key=lambda r: r[2])
peak = collections.defaultdict(int)
for r in csv.DictReader(l for l in open(d + "/cards.csv") if not l.startswith("#")):
    for k, v in r.items():
        if k.startswith("g") and k.endswith("_used_mib") and v not in ("", None):
            i = int(k[1:].split("_")[0])
            peak[i] = max(peak[i], int(float(v)))
print("run\tarm\tlowest_memavailable_mib\tat\tswap_start_mib\tswap_max_mib\tswap_max_at\tmemavailable_end_mib\tcard_peak_mib_0_to_8")
print("\t".join([run, arm, str(lo[1]), t(lo[0]), str(rows[0][2]), str(sw[2]), t(sw[0]), str(rows[-1][1]),
                 ",".join(str(peak[i]) for i in sorted(peak))]))
