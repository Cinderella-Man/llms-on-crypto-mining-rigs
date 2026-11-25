#!/usr/bin/env python3
"""rig_tail.py: the rig check's prompt tables (C1024, Q1757), standard library only.

    python3 -I rig_tail.py --lane NAME=OUT [--lane NAME=OUT ...] [--control unset] [--tsv tail.tsv]

Each OUT is one deep-lane run of run/rig_lane.sh (its requests.csv and loads.tsv). Per lane, depth and arm, every timed
load by value (the warm-up load is left out):
- the tail: the first decode pass at each depth (`kind` decode, `pass` 1), which reads the prompt's last tokens in one
  micro-batch on top of the prime's cache, its prompt_n and its prompt_ms;
- the prime: the prompt without its tail (`kind` prime), its prompt_n and its prompt tok/s.
The class against the control arm is deep_table.py's rule: the spread is the larger of the two arms' differences
between their slowest and fastest load; gain when the arm's mean is better than the control's by more than the spread
(a time lower, a speed higher), loss when worse by more than it, else negligible. Prints Markdown; --tsv writes one
row per lane, metric, depth and arm.
"""
import argparse
import csv
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--lane', action='append', required=True, help='NAME=OUT')
ap.add_argument('--control', default='unset')
ap.add_argument('--tsv', default='')
A = ap.parse_args()


def timed_loads(src):
    path = os.path.join(src, 'loads.tsv')
    if not os.path.exists(path):
        return None
    keep = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r.get('timed') == 'stopped':
                continue
            keep.pop(r['load'], None)
            keep[r['load']] = r
    return [k for k, r in keep.items() if r['timed'] == '1']


def cls(ctl, arm, lower_better):
    spread = max(max(ctl) - min(ctl), max(arm) - min(arm))
    mc, ma = sum(ctl) / len(ctl), sum(arm) / len(arm)
    delta = ma - mc
    better = -delta if lower_better else delta
    c = 'gain' if better > spread else 'loss' if -better > spread else 'negligible'
    return mc, ma, delta, 100 * delta / mc, spread, c


def fmt(v, nd):
    return f'{v:,.{nd}f}'


out_rows = []
for lane in A.lane:
    name, src = lane.split('=', 1)
    with open(os.path.join(src, 'requests.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    loads = timed_loads(src)
    if loads is None:
        loads = sorted({r['load'] for r in rows if r['load'] != 'warm'})
    arms = []
    for r in rows:
        if r['load'] in loads and r['arm'] not in arms:
            arms.append(r['arm'])
    others = [a for a in arms if a != A.control]
    depths = sorted({int(r['depth']) for r in rows if r['kind'] in ('decode', 'prime')})
    for metric, kind, pas, lower in (('tail prompt ms', 'decode', '1', True), ('prime prompt tok/s', 'prime', '', False)):
        print(f'\n### {name}: {metric}\n')
        print('| Depth | Arm | Tokens read | Loads (by value) | Mean | Spread | Against control | Class |')
        print('|---|---|---|---|---|---|---|---|')
        for d in depths:
            vals, ns = {}, {}
            for a in arms:
                sel = [r for r in rows if r['arm'] == a and r['load'] in loads and r['kind'] == kind
                       and int(r['depth']) == d and (not pas or r['pass'] == pas)]
                sel.sort(key=lambda r: loads.index(r['load']))
                if not sel:
                    continue
                if kind == 'prime':
                    sel = [r for r in sel if float(r['prompt_ms'] or 0) > 0 and int(r['prompt_n'] or 0) >= 32]
                    if not sel:
                        continue
                    vals[a] = [(r['load'], int(r['prompt_n']) * 1000 / float(r['prompt_ms'])) for r in sel]
                else:
                    vals[a] = [(r['load'], float(r['prompt_ms'])) for r in sel]
                ns[a] = sorted({int(r['prompt_n'] or 0) for r in sel})
            if A.control not in vals:
                continue
            nd = 3 if kind == 'decode' else 2
            for a in [A.control] + [o for o in others if o in vals]:
                v = [x for _, x in vals[a]]
                byval = ', '.join(f'{fmt(x, nd)} ({ld})' for ld, x in vals[a])
                if a == A.control:
                    print(f'| {d:,} | {a} | {"/".join(str(n) for n in ns[a])} | {byval} | {fmt(sum(v) / len(v), nd)} | '
                          f'{fmt(max(v) - min(v), nd)} | control | |')
                    out_rows.append([name, metric, d, a, '/'.join(str(n) for n in ns[a]), ' '.join(f'{x:.3f}' for x in v),
                                     f'{sum(v) / len(v):.3f}', f'{max(v) - min(v):.3f}', '', '', '', 'control'])
                    continue
                c = [x for _, x in vals[A.control]]
                mc, ma, delta, pct, spread, k = cls(c, v, lower)
                print(f'| {d:,} | {a} | {"/".join(str(n) for n in ns[a])} | {byval} | {fmt(ma, nd)} | {fmt(spread, nd)} | '
                      f'{delta:+,.{nd}f} ({pct:+.2f} %) | **{k}** |')
                out_rows.append([name, metric, d, a, '/'.join(str(n) for n in ns[a]), ' '.join(f'{x:.3f}' for x in v),
                                 f'{ma:.3f}', f'{spread:.3f}', f'{delta:.3f}', f'{pct:.2f}', f'{mc:.3f}', k])

if A.tsv:
    with open(A.tsv, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['lane', 'metric', 'depth', 'arm', 'tokens_read', 'loads', 'mean', 'spread', 'delta', 'percent',
                    'control_mean', 'class'])
        w.writerows(out_rows)
    print(f'\n{len(out_rows)} rows to {A.tsv}', file=sys.stderr)
