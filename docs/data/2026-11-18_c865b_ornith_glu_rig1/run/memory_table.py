#!/usr/bin/env python3
"""memory_table.py: MiB on every card and the host's free memory per load of a deep-lane run, standard library only.

    python3 -I memory_table.py OUT memory.tsv

Reads OUT/loads.tsv (the load and its arm), OUT/fit_<load>.txt (deep_lane.sh's `nvidia-smi` lines "index, used MiB,
total MiB" and /proc/meminfo's MemAvailable once the model is up) and OUT/ram_watchdog-<load>.tsv (the RAM watchdog's
samples; column 2 is MemAvailable in MiB) and writes one line per load: the MiB on each card, MemAvailable at the fit
check in KiB, and the lowest MemAvailable the watchdog saw during the load in MiB.
"""
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
order, arms = [], {}
for line in open(os.path.join(src, 'loads.tsv')):
    p = line.rstrip('\n').split('\t')
    if p[0] != 'load':
        order.append(p[0])
        arms[p[0]] = p[1]
n_cards = None
rows = []
for load in order:
    lines = open(os.path.join(src, 'fit_%s.txt' % load)).read().splitlines()
    mib = [l.split(',')[1].strip() for l in lines if l[:1].isdigit() and l.count(',') == 2]
    n_cards = n_cards or len(mib)
    avail = [l.split()[1] for l in lines if l.startswith('MemAvailable')][0]
    low = min(int(l.split('\t')[1]) for l in open(os.path.join(src, 'ram_watchdog-%s.tsv' % load))
              if l.strip() and not l.startswith('#'))
    rows.append('\t'.join([load, arms[load]] + mib + [avail, str(low)]))
head = ['load', 'arm'] + ['card%d_mib' % i for i in range(n_cards)] + ['memavailable_kib_at_fit', 'lowest_memavailable_mib']
with open(dst, 'w') as f:
    f.write('\t'.join(head) + '\n' + '\n'.join(rows) + '\n')
