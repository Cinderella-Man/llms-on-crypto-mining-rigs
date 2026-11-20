#!/usr/bin/env python3
"""steps.py: R175's verdict table: the step of every load per depth and pass type, and the draft shortlist's counts.

    python3 -I steps.py OUT [--control c] [--arm fb1] [--tsv steps.tsv] > steps.md

OUT is a deep lane's results folder (bench/deep_lane). From OUT/requests.csv, for every depth and pass type (the greedy
`decode` requests' pass 1 and pass 2, and the `sample` request with the preset's sampler), each timed load's step in ms
(generation time over steps; a request's steps are its generated tokens minus the draft tokens it accepted, as
deep_table.py counts them), its steps and its accepted of drafted tokens. The arm is classed against the control two
ways:
- `ctl`: the spread is the control's two loads' difference alone;
- `lane`: the spread is the larger of the two arms' differences (deep_table.py's rule).
Gain when the arm's mean is below the control's by more than the spread, loss when above by more than it, else
negligible. "all" rows add every depth's generation time and steps of that pass type. From OUT/server-<load>.log, the
draft shortlist's last count ("draft head shortlist: N drafts with it, M with the full head, K target tokens outside
it") and its set-up line ("full head for R drafts after a target token outside it"). Standard library only.
"""
import argparse
import csv
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out')
ap.add_argument('--control', default='c')
ap.add_argument('--arm', default='fb1')
ap.add_argument('--tsv', default='')
A = ap.parse_args()

with open(os.path.join(A.out, 'requests.csv'), newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['load'] != 'warm' and r['kind'] in ('decode', 'sample')
            and r['arm'] in (A.control, A.arm)]
loads = {A.control: [], A.arm: []}
for r in rows:
    if r['load'] not in loads[r['arm']]:
        loads[r['arm']].append(r['load'])
KINDS = [('decode', '1', 'greedy, pass 1'), ('decode', '2', 'greedy, pass 2'), ('sample', None, 'sampled')]
depths = sorted({int(r['depth']) for r in rows})


def pick(load, kind, ps, depth):
    out = [r for r in rows if r['load'] == load and r['kind'] == kind and (ps is None or r['pass'] == ps)
           and (depth is None or int(r['depth']) == depth)]
    ms = sum(float(r['gen_ms']) for r in out)
    steps = sum(int(r['gen_n']) - int(r['draft_accepted'] or 0) for r in out)
    acc = sum(int(r['draft_accepted'] or 0) for r in out)
    dn = sum(int(r['draft_n'] or 0) for r in out)
    return ms / steps if steps else float('nan'), steps, acc, dn, {r['ids_md5'] for r in out}


def klass(c, a, spread):
    d = sum(a) / len(a) - sum(c) / len(c)
    return d, ('gain' if d < -spread else 'loss' if d > spread else 'negligible')


md = [f'| Pass type | Depth | `{A.control}` loads, ms (steps; accepted of drafted) | `{A.arm}` loads | Arm − control |'
      ' Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |', '|---|---:|---|---|---:|---:|---|---:|---|']
tsv = ['pass_type\tdepth\tload\tarm\tstep_ms\tsteps\taccepted\tdrafted\tids_md5']
for kind, ps, name in KINDS:
    for depth in depths + [None]:
        vals = {}
        for arm in (A.control, A.arm):
            for ld in loads[arm]:
                st, n, acc, dn, ids = pick(ld, kind, ps, depth)
                vals.setdefault(arm, []).append((ld, st, n, acc, dn))
                tsv.append(f"{name}\t{'all' if depth is None else depth}\t{ld}\t{arm}\t{st:.3f}\t{n}\t{acc}\t{dn}\t"
                           f"{','.join(sorted(i[:8] for i in ids))}")
        c = [v[1] for v in vals[A.control]]
        a = [v[1] for v in vals[A.arm]]
        sc, sa = max(c) - min(c), max(a) - min(a)
        d, k1 = klass(c, a, sc)
        _, k2 = klass(c, a, max(sc, sa))
        fmt = lambda vs: '; '.join(f'{ld} {st:.2f} ({n}; {acc}/{dn})' for ld, st, n, acc, dn in vs)
        md.append(f"| {name} | {'all' if depth is None else f'{depth:,}'} | {fmt(vals[A.control])} | {fmt(vals[A.arm])} "
                  f"| {d:+.2f} | {sc:.2f} | {k1} | {max(sc, sa):.2f} | {k2} |")

md += ['', '| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head |'
       ' Target tokens outside it | Ids in it at the end |', '|---|---|---:|---:|---:|---:|---:|']
for arm in (A.control, A.arm):
    for ld in loads[arm]:
        rounds = rep = None
        with open(os.path.join(A.out, f'server-{ld}.log'), errors='replace') as f:
            for line in f:
                m = re.search(r'full head for (\d+) drafts after a target token outside it', line)
                if m:
                    rounds = m.group(1)
                m = re.search(r'draft head shortlist: (\d+) drafts with it, (\d+) with the full head, (\d+) target '
                              r'tokens outside it; (\d+) ids in it', line)
                if m:
                    rep = m.groups()
        md.append(f"| {ld} | {arm} | {rounds} | " + (' | '.join(rep) if rep else 'none | none | none | none') + ' |')
print('\n'.join(md))
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write('\n'.join(tsv) + '\n')
