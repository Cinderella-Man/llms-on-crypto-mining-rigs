#!/usr/bin/env python3
"""Summarises an nvidia-smi sample log (smi.csv: timestamp, index, pstate, clocks.current.sm, clocks.current.memory,
power.draw, temperature.gpu, utilization.gpu, memory.used, clocks_event_reasons.active) by phase, to see whether the
cards run at a lower SM clock after a heavy prompt phase (Q117's order-effect question).

Usage: smi_phases.py smi.csv [gpus]     gpus: comma list (default: every index with memory in use)

Per second it averages the selected GPUs and labels the second by GPU utilisation and power: idle (util < 10 %),
busy. Consecutive busy seconds form a phase; for each phase it prints duration, mean/min SM clock, mean power, max
temperature and the union of throttle reasons (bit names below). Throttle bits: 0x1 idle, 0x4 SW power cap,
0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake."""
import csv
import sys
from collections import defaultdict
from datetime import datetime

BITS = {0x1: 'idle', 0x2: 'appclk', 0x4: 'pwrcap', 0x8: 'hwslow', 0x10: 'sync', 0x20: 'swtherm', 0x40: 'hwtherm',
        0x80: 'pwrbrake', 0x100: 'display'}


def num(s):
    return float(s.strip().split()[0])


rows = []
with open(sys.argv[1]) as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        if len(line) < 10:
            continue
        t = datetime.strptime(line[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
        rows.append(dict(t=t.replace(microsecond=0), gpu=int(line[1]), sm=num(line[3]), pw=num(line[5]),
                         temp=num(line[6]), util=num(line[7]), mem=num(line[8]), reasons=int(line[9].strip(), 16)))

if len(sys.argv) > 2:
    gpus = {int(g) for g in sys.argv[2].split(',')}
else:
    gpus = {r['gpu'] for r in rows if r['mem'] > 500}

sec = defaultdict(list)
for r in rows:
    if r['gpu'] in gpus:
        sec[r['t']].append(r)

phases = []
cur = None
for t in sorted(sec):
    rs = sec[t]
    util = sum(r['util'] for r in rs) / len(rs)
    busy = util >= 10
    if busy:
        if cur is None:
            cur = dict(t0=t, rs=[])
            phases.append(cur)
        cur['t1'] = t
        cur['rs'].extend(rs)
    else:
        cur = None

print(f'{sys.argv[1]}: GPUs {sorted(gpus)}, {len(sec)} s sampled')
print(f'{"start":>8} {"dur_s":>5} {"util%":>5} {"sm_mean":>7} {"sm_min":>6} {"pw_mean":>7} {"temp_max":>8}  reasons')
for p in phases:
    rs = p['rs']
    dur = (p['t1'] - p['t0']).seconds + 1
    if dur < 2:
        continue
    reasons = 0
    for r in rs:
        reasons |= r['reasons']
    names = ','.join(n for b, n in BITS.items() if reasons & b) or '-'
    print(f'{p["t0"].strftime("%H:%M:%S"):>8} {dur:>5} {sum(r["util"] for r in rs)/len(rs):>5.0f} '
          f'{sum(r["sm"] for r in rs)/len(rs):>7.0f} {min(r["sm"] for r in rs):>6.0f} '
          f'{sum(r["pw"] for r in rs)/len(rs):>7.1f} {max(r["temp"] for r in rs):>8.0f}  {names}')
