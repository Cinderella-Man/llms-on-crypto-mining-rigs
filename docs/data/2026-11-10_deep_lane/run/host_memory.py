#!/usr/bin/env python3
"""host_memory.py: the host's memory during a deep-lane run, from the RAM watchdog's samples (one a second).

    python3 -I host_memory.py OUT > host_memory.tsv

OUT is a deep-lane results folder (requests.csv, ram_watchdog-<load>.tsv). One line per load (the whole load) and one
per request (the samples between the request's start and end): the lowest MemAvailable and usable memory, the most
swap in use and its change over the window, the highest pressure (psi some and full, 10 s averages).
"""
import csv
import datetime
import glob
import os
import sys

src = sys.argv[1]
samples = {}
for path in sorted(glob.glob(os.path.join(src, 'ram_watchdog-*.tsv'))):
    load = os.path.basename(path)[len('ram_watchdog-'):-len('.tsv')]
    with open(path) as f:
        samples[load] = [[float(x) for x in line.split('\t')[:8]] for line in f if line[:1].isdigit()]


def line(name, rows):
    if not rows:
        return f'{name}\t0'
    return '\t'.join([name, str(len(rows)), f'{min(r[1] for r in rows):.0f}', f'{min(r[3] for r in rows):.0f}',
                      f'{max(r[4] for r in rows):.0f}', f'{rows[-1][4] - rows[0][4]:+.0f}',
                      f'{max(r[6] for r in rows):.2f}', f'{max(r[7] for r in rows):.2f}'])


print('window\tsamples\tmin_avail_mib\tmin_usable_mib\tmax_swap_used_mib\tswap_used_change_mib\tmax_psi_some\tmax_psi_full')
for load, rows in samples.items():
    print(line(f'{load} (whole load)', rows))
with open(os.path.join(src, 'requests.csv'), newline='') as f:
    for r in csv.DictReader(f):
        if r['load'] not in samples or r['kind'] == 'warmup':
            continue
        end = datetime.datetime.strptime(r['time'], '%Y-%m-%dT%H:%M:%S%z').timestamp()
        start = end - float(r['wall_s'])
        rows = [s for s in samples[r['load']] if start - 1 <= s[0] <= end + 1]
        print(line(f'{r["load"]} {r["kind"]} pass {r["pass"]} depth {r["depth"]}', rows))
