#!/usr/bin/env python3
"""host_summary.py RIGDIR RUN ARM: one host.tsv row from the agent check's rig logs (release_check_s12.sh's collect).

RIGDIR holds meminfo.log (epoch, MemAvailable kB, swap used kB a line, every 2 s, from the server's start to its stop)
and gpu.csv (nvidia-smi: local time, index, temperature, power, SM clock, memory used "N MiB", utilization). When
several runs share one client folder, the collect step appends each run's logs to the same files: only the last run
is read (the lines after the last gap of more than 300 s in meminfo.log, and the gpu.csv lines from that time on).
Prints the columns of host.tsv: run, arm, lowest MemAvailable MiB and its local time, swap used at the start, its
highest and when, MemAvailable at the end, and each card's peak memory MiB (index order). UTC_OFFSET_H is the rig's
offset from UTC (default 1, CET). Runs anywhere with Python 3 (no packages).
"""
import csv
import os
import sys
import time

rigdir, run, arm = sys.argv[1:4]
OFF = float(os.environ.get("UTC_OFFSET_H", "1"))
loc = lambda e, f="%H:%M:%S": time.strftime(f, time.gmtime(e + OFF * 3600))
mib = lambda kb: round(kb / 1024)
rows = []
for line in open(os.path.join(rigdir, "meminfo.log")):
    p = line.split()
    if len(p) >= 3 and p[0].isdigit():
        e = int(p[0])
        if rows and e - rows[-1][0] > 300:
            rows = []  # a new run starts: keep only the last one
        rows.append((e, int(p[1]), int(p[2])))
start = loc(rows[0][0] - 2, "%Y/%m/%d %H:%M:%S")
low = min(rows, key=lambda r: r[1])
smax = max(rows, key=lambda r: r[2])
peak = {}
for r in csv.reader(open(os.path.join(rigdir, "gpu.csv"))):
    if len(r) < 6 or r[0].strip()[:19] < start:
        continue
    try:
        i, m = int(r[1]), int(r[5].strip().split()[0])
    except ValueError:
        continue
    peak[i] = max(peak.get(i, 0), m)
print("\t".join(map(str, [run, arm, mib(low[1]), loc(low[0]), mib(rows[0][2]), mib(smax[2]), loc(smax[0]),
                          mib(rows[-1][1]), ",".join(str(peak[i]) for i in sorted(peak))])))
print(f"# {len(rows)} meminfo lines from {loc(rows[0][0])} to {loc(rows[-1][0])}; gpu.csv lines from {start}",
      file=sys.stderr)
