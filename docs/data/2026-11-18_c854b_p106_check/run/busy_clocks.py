#!/usr/bin/env python3
"""busy_clocks.py LABEL=LOG FROM TO [LABEL=LOG FROM TO ...]: each card's SM clock while busy, per run, in one window.

LOG is a rig's card log: scripts/card_log.py's cards.csv (one row a sample, g<i>_sm_mhz and g<i>_util per card), or
rig_server.sh's older gpu.csv (nvidia-smi rows: time, index, temperature, power, SM clock, memory used, utilization).
FROM and TO are the rig's local times "YYYY-MM-DD HH:MM:SS" (the agent check's first task start and last task end).
Busy is utilization above 0 %. Prints a TSV row a run: the busy samples' median SM clock of each card, their mean over
the cards, and each card's share of busy samples below 1,800 MHz. Standard library only.
"""
import csv
import statistics
import sys


def samples(path, t0, t1):
    """(card, SM MHz) for every busy sample in [t0, t1]."""
    with open(path) as f:
        first = f.readline()
    if first.startswith("epoch"):
        rows = csv.DictReader(l for l in open(path) if not l.startswith("#"))
        for r in rows:
            t = r["time"][:19]
            if not t0 <= t <= t1:
                continue
            for k in r:
                if k.startswith("g") and k.endswith("_util") and r[k] not in ("", None) and float(r[k]) > 0:
                    c = int(k[1:].split("_")[0])
                    yield c, int(float(r[f"g{c}_sm_mhz"]))
    else:
        for l in open(path):
            f = [x.strip() for x in l.split(",")]
            if len(f) < 7 or not f[0][:4].isdigit():
                continue
            t = f[0][:19].replace("/", "-")
            if t0 <= t <= t1 and int(f[6].split()[0]) > 0:
                yield int(f[1]), int(f[4].split()[0])


a = sys.argv[1:]
print("run\tfrom\tto\tbusy_sm_mhz_median_0_to_8\tmean_of_cards\tbusy_below_1800_pct_0_to_8")
for i in range(0, len(a), 3):
    (label, path), t0, t1 = a[i].split("=", 1), a[i + 1], a[i + 2]
    clk = {}
    for c, m in samples(path, t0, t1):
        clk.setdefault(c, []).append(m)
    med = [statistics.median(clk[c]) for c in sorted(clk)]
    low = [100 * sum(m < 1800 for m in clk[c]) / len(clk[c]) for c in sorted(clk)]
    print("\t".join([label, t0, t1, ",".join(f"{m:g}" for m in med), f"{statistics.mean(med):.1f}",
                     ",".join(f"{x:.2f}" for x in low)]))
