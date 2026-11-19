#!/usr/bin/env python3
"""relabel.py SRC DST old=new,...: a deep-lane run's requests.csv and loads.tsv copied from SRC to DST with arms renamed.

FN115's loads are each their own arm in the lane (own1, own2, dist1, dist2), so that each writes its own step trace;
bench/deep_lane/deep_table.py classes an arm against the control on two loads of each arm, so the tables read a copy
with the arms named own and dist again:
    python3 -I relabel.py OUT REL own0=own,own1=own,own2=own,dist1=dist,dist2=dist
Every other column is copied as it is. Standard library only.
"""
import csv
import os
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
src, dst, spec = sys.argv[1:]
amap = dict(x.split('=', 1) for x in spec.split(',') if '=' in x)
os.makedirs(dst, exist_ok=True)
for name, delim in (('requests.csv', ','), ('loads.tsv', '\t')):
    p = os.path.join(src, name)
    if not os.path.exists(p):
        continue
    with open(p, newline='') as f:
        rd = csv.DictReader(f, delimiter=delim, quoting=csv.QUOTE_NONE if delim == '\t' else csv.QUOTE_MINIMAL)
        rows, fields = list(rd), rd.fieldnames
    for r in rows:
        r['arm'] = amap.get(r['arm'], r['arm'])
    with open(os.path.join(dst, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=delim, lineterminator='\n',
                           quoting=csv.QUOTE_NONE if delim == '\t' else csv.QUOTE_MINIMAL, escapechar='\\' if delim == '\t' else None)
        w.writeheader()
        w.writerows(rows)
    print(f'{name}: {len(rows)} rows, arms {sorted(set(r["arm"] for r in rows))}')
