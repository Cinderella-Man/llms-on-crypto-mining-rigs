#!/usr/bin/env python3
"""How often the software power cap (clock event reason 0x4) shows in nvidia-smi sample logs, counted two ways: as a
share of the samples, and as the busy phases in which at least one sample of one card has it.

Q108 decode question (2026-10-24): power_and_tuning.md counts the cap in 1 % of all samples, while
scripts/smi_phases.py lists it in nearly every busy phase. This prints both counts per log, so the two can be read
side by side.

Usage: cap_share.py smi.csv [smi.csv ...] > cap_share.csv
Input: nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,
memory.used,clocks_event_reasons.active --format=csv -lms <ms> (the format scripts/smi_phases.py reads).
Cards in use: every index with more than 500 MiB of memory in use. Phases are cut as smi_phases.py cuts them:
consecutive seconds whose mean utilization over the cards in use is at least 10 %, at least 2 s long. A compute phase
is one whose mean SM clock is at least 1,300 MHz (the model-loading phases run at the 139 MHz idle clock).
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime

CAP = 0x4
SLOW = {0x8: 'hwslow', 0x20: 'swtherm', 0x40: 'hwtherm', 0x80: 'pwrbrake'}

w = csv.writer(sys.stdout, lineterminator='\n')
w.writerow(['log', 'cards_in_use', 'samples', 'capped_samples', 'capped_pct', 'busy_samples', 'capped_busy_pct',
            'compute_phases', 'compute_phases_ge_10s', 'compute_phases_ge_10s_with_cap', 'compute_phase_samples',
            'capped_pct_in_compute_phases', 'sm_clock_mean_compute_mhz', 'power_mean_compute_w', 'slowdown_bits_seen'])
for path in sys.argv[1:]:
    recs = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for l in r:
            if len(l) < 10:
                continue
            recs.append(dict(t=datetime.strptime(l[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(microsecond=0),
                             gpu=int(l[1]), sm=float(l[3].split()[0]), pw=float(l[5].split()[0]),
                             util=float(l[7].split()[0]), mem=float(l[8].split()[0]), reasons=int(l[9].strip(), 16)))
    gpus = {x['gpu'] for x in recs if x['mem'] > 500}
    sel = [x for x in recs if x['gpu'] in gpus]
    sec = defaultdict(list)
    for x in sel:
        sec[x['t']].append(x)
    phases, cur = [], None
    for t in sorted(sec):
        xs = sec[t]
        if sum(x['util'] for x in xs) / len(xs) >= 10:
            if cur is None:
                cur = dict(t0=t, xs=[])
                phases.append(cur)
            cur['t1'] = t
            cur['xs'].extend(xs)
        else:
            cur = None
    phases = [p for p in phases if (p['t1'] - p['t0']).seconds + 1 >= 2]
    comp = [p for p in phases if st.mean(x['sm'] for x in p['xs']) >= 1300]
    long_ = [p for p in comp if (p['t1'] - p['t0']).seconds + 1 >= 10]
    capped = lambda xs: sum(1 for x in xs if x['reasons'] & CAP)
    in_comp = [x for p in comp for x in p['xs']]
    busy = [x for x in sel if x['util'] >= 10]
    slow = 0
    for x in recs:
        slow |= x['reasons'] & sum(SLOW)
    w.writerow([path, len(gpus), len(sel), capped(sel), round(100 * capped(sel) / len(sel), 2), len(busy),
                round(100 * capped(busy) / len(busy), 1) if busy else '', len(comp), len(long_),
                sum(1 for p in long_ if capped(p['xs'])), len(in_comp),
                round(100 * capped(in_comp) / len(in_comp), 1) if in_comp else '',
                round(st.mean(x['sm'] for x in in_comp)) if in_comp else '',
                round(st.mean(x['pw'] for x in in_comp), 1) if in_comp else '',
                ','.join(n for b, n in SLOW.items() if slow & b) or '-'])
