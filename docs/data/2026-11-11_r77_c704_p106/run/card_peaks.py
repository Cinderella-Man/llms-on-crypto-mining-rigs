#!/usr/bin/env python3
"""card_peaks.py: each card's highest memory in use per load of a deep-lane run, from its MONITOR samples.

    python3 -I card_peaks.py OUT > card_peaks.tsv

OUT is a deep-lane results folder (bench/deep_lane): loads.tsv gives each load's start; a load's samples are those of
gpu.csv (one line per card every 10 s: time, index, temperature, power, clock, memory, utilization) from its start to
the next load's start (the last load: to the end of the file). Writes one row per load: the load, its arm, its samples
per card and each card's highest MiB. Standard library only.
"""
import csv
import datetime
import os
import sys

out = sys.argv[1]
with open(os.path.join(out, 'loads.tsv')) as f:
    loads = [(r['load'], r['arm'], datetime.datetime.fromisoformat(r['started']).replace(tzinfo=None))
             for r in csv.DictReader(f, delimiter='\t')]
samples = []
with open(os.path.join(out, 'gpu.csv')) as f:
    for row in csv.reader(f, skipinitialspace=True):
        if len(row) < 6 or not row[1].strip().isdigit():
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
            mib = int(row[5].split()[0])
        except ValueError:
            continue
        samples.append((t, int(row[1]), mib))
cards = sorted({c for _, c, _ in samples})
print('\t'.join(['load', 'arm', 'samples_a_card'] + ['card%d_mib' % c for c in cards]))
for i, (name, arm, t0) in enumerate(loads):
    t1 = loads[i + 1][2] if i + 1 < len(loads) else datetime.datetime.max
    peak = {c: 0 for c in cards}
    n = 0
    for t, c, mib in samples:
        if t0 <= t < t1:
            peak[c] = max(peak[c], mib)
            n += c == cards[0]
    print('\t'.join([name, arm, str(n)] + [str(peak[c]) for c in cards]))
