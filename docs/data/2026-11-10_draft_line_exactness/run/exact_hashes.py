#!/usr/bin/env python3
"""exact_hashes.py: every load's greedy reply per depth as a hash, one row per load (registry Q1175).

  python3 exact_hashes.py --out hashes.tsv [--ref-csv <deep_lane requests.csv>] <label>=<replies_<arm>.json> ...

A cell is the first eight characters of the md5 of the reply's token ids (the first request at that depth), as
bench/deep_lane writes it. --ref-csv adds the arms of an earlier deep_lane run (its first decode pass per depth).
"""
import argparse
import csv
import json

ap = argparse.ArgumentParser()
ap.add_argument('loads', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--ref-csv', default='')
ap.add_argument('--ref-label', default='the earlier run')
A = ap.parse_args()

rows, depths = [], set()
if A.ref_csv:
    ref = {}
    with open(A.ref_csv) as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'decode' and r['pass'] == '1':
                ref.setdefault(r['arm'], {})[int(r['depth'])] = r['ids_md5'][:8]
    for arm, cells in ref.items():
        rows.append((f'{A.ref_label}, {arm}', cells))
        depths |= set(cells)
for spec in A.loads:
    label, path = spec.split('=', 1)
    with open(path) as f:
        cells = {int(d): r['ids_md5'][:8] for d, r in json.load(f)['replies'].items()}
    rows.append((label, cells))
    depths |= set(cells)
depths = sorted(depths)
with open(A.out, 'w') as f:
    f.write('load\t' + '\t'.join(str(d) for d in depths) + '\n')
    for label, cells in rows:
        f.write(label + '\t' + '\t'.join(cells.get(d, '') for d in depths) + '\n')
    f.write('distinct replies\t' + '\t'.join(str(len({c[d] for _, c in rows if d in c})) for d in depths) + '\n')
print(open(A.out).read(), end='')
