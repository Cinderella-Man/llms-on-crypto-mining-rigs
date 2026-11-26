#!/usr/bin/env python3
"""R243 (Q1921): each card's memory in an agent run, from the card log (scripts/card_log.py's cards.csv): after loading
(the last sample before the run's first task, when the server is up and idle) and at its highest over the tasks, and
what the highest leaves of what a card offers a process.

    python3 -I card_memory.py USABLE_MIB RUN=CARDS_CSV=FIRST=LAST ...

NVML, which the card log reads, counts the driver's reserve as used (75 MiB a card on an idle P106-090 under driver
580, where nvidia-smi shows 0), so each card's reading in the log's first sample (taken when the server starts, with
nothing loaded) is subtracted, as R124's run/card_fit.py does (docs/data/2026-11-26_r124_flash_131k_rig2/): the columns
are on nvidia-smi's scale. USABLE_MIB is what one card offers a process, the card's total less that reserve
(6,144 - 76 = 6,068 MiB on a P106-090 under driver 580, the total llama.cpp prints for the card). FIRST and LAST are the
run's first task's start and last task's end, "YYYY-MM-DD HH:MM:SS" in the rig's local time (tables_agent.sh's
window_<run>). Prints one row a run and card. Standard library only.
"""
import csv
import sys

usable = int(sys.argv[1])
print("run\tcard\tidle_reading_mib\tafter_load_mib\tpeak_mib\tpeak_at\tleft_at_peak_mib")
for arg in sys.argv[2:]:
    run, path, first, last = arg.split("=")
    rows = [r for r in csv.DictReader(line for line in open(path) if not line.startswith("#"))]
    before = [r for r in rows if r["time"][:19] < first]
    during = [r for r in rows if first <= r["time"][:19] <= last]
    cards = sorted({int(k[1:].split("_")[0]) for k in rows[0] if k.startswith("g") and k.endswith("_used_mib")})
    for c in cards:
        col = f"g{c}_used_mib"
        idle = int(float(rows[0][col]))
        load = int(float(before[-1][col])) - idle if before else -1
        top = max(during, key=lambda r: float(r[col]))
        peak = int(float(top[col])) - idle
        print(f"{run}\t{c}\t{idle}\t{load}\t{peak}\t{top['time'][11:19]}\t{usable - peak}")
