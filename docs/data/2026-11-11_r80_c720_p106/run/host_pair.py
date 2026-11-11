#!/usr/bin/env python3
"""host_pair.py: two host-profiled loads' parts of the step side by side, from host_depth.py's summary.tsv of each
(R80: hostdev, C100's tables on the cards, against hoststage, the same with C720's small inputs in the update).
Standard library only.

    python3 -I host_pair.py A/summary.tsv B/summary.tsv --names hostdev,hoststage [--pass p2] [--tsv OUT.tsv] > pair.md

Per part and depth of the chosen pass: A's mean ms a step, B's, B − A; then each load's growth from 16,384 to
49,152 tokens and B's growth less A's. Rows in host_depth.py's order."""
import argparse
import csv

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('a')
ap.add_argument('b')
ap.add_argument('--names', default='A,B')
ap.add_argument('--pass', dest='pas', default='p2')
ap.add_argument('--tsv', default='')
A = ap.parse_args()
NA, NB = A.names.split(',')


def rd(path):
    out, order = {}, []
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            req = r['request']
            if not req.endswith(' ' + A.pas):
                continue
            d = int(req.split()[1])
            if r['part'] not in order:
                order.append(r['part'])
            try:
                out[(r['part'], d)] = float(r['mean_ms'])
            except ValueError:
                pass
    return out, order


a, order = rd(A.a)
b, _ = rd(A.b)
depths = sorted({d for (_, d) in a})
lo, hi = 16384, 49152
head = ['part']
for d in depths:
    head += ['%s %d' % (NA, d), '%s %d' % (NB, d), 'diff %d' % d]
head += ['%s growth' % NA, '%s growth' % NB, 'growth diff']
rows = []
for p in order:
    row = [p]
    for d in depths:
        x, y = a.get((p, d)), b.get((p, d))
        row += ['%.2f' % x if x is not None and x == x else '', '%.2f' % y if y is not None and y == y else '',
                '%+.2f' % (y - x) if None not in (x, y) and x == x and y == y else '']
    ga = a.get((p, hi), float('nan')) - a.get((p, lo), float('nan'))
    gb = b.get((p, hi), float('nan')) - b.get((p, lo), float('nan'))
    row += ['%+.2f' % ga if ga == ga else '', '%+.2f' % gb if gb == gb else '',
            '%+.2f' % (gb - ga) if ga == ga and gb == gb else '']
    rows.append(row)
print('| ' + ' | '.join(head) + ' |')
print('|---|' + '---:|' * (len(head) - 1))
for r in rows:
    print('| ' + ' | '.join(r) + ' |')
if A.tsv:
    with open(A.tsv, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(head)
        w.writerows(rows)
