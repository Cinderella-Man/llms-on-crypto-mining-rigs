#!/usr/bin/env python3
"""card_max.py: the highest memory in use per card per load of one deep-lane run.

    python3 -I card_max.py OUT > card_max.tsv

OUT is a deep-lane results folder (bench/deep_lane: loads.tsv, gpu.csv, nvidia-smi's samples every 10 s). A load's
samples are those from its start (loads.tsv, started) to the next load's start; a load listed twice (a resumed run)
counts from its last start. Prints per load the highest MiB of each card, the fullest card and card 8's value.
Standard library only.
"""
import csv
import os
import sys
from datetime import datetime

src = sys.argv[1]
with open(os.path.join(src, 'loads.tsv'), newline='') as f:  # both clocks read as the rig's local time
    starts = sorted((datetime.fromisoformat(r['started'][:19]).timestamp(), r['load'])
                    for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))
samples = []
with open(os.path.join(src, 'gpu.csv')) as f:
    for line in f:
        p = [x.strip() for x in line.split(',')]
        if len(p) < 6 or not p[1].isdigit():
            continue
        t = datetime.strptime(p[0].split('.')[0], '%Y/%m/%d %H:%M:%S').timestamp()
        samples.append((t, int(p[1]), int(p[5].split()[0])))
cards = sorted({c for _, c, _ in samples})
rows = {}  # a load started twice keeps its last start's window; every start ends the window before it
for i, (t0, name) in enumerate(starts):
    t1 = starts[i + 1][0] if i + 1 < len(starts) else float('inf')
    mx, n = {}, 0
    for t, c, m in samples:
        if t0 <= t < t1:
            mx[c] = max(mx.get(c, 0), m)
            n += 1
    if mx:
        top = max(mx, key=mx.get)
        rows[name] = [name, str(n)] + [str(mx.get(c, '')) for c in cards] + [f'card {top}', str(mx[top])]
print('\t'.join(['load', 'samples'] + [f'card{c}_max_mib' for c in cards] + ['fullest', 'fullest_mib']))
for name in dict.fromkeys(n for _, n in starts):
    if name in rows:
        print('\t'.join(rows[name]))
