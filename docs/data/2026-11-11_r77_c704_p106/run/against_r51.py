#!/usr/bin/env python3
"""against_r51.py: the profiled load's buckets and step totals beside R51's, per depth and pass, summed over the main
pass's cards and the draft.

    python3 -I against_r51.py R51_DIR R77_DIR > against_r51.tsv

Each folder holds depth_parts.py's buckets.tsv and totals.tsv (R51: docs/data/2026-11-10_step_parts_depth/; R77: this
folder). Rows: the deep trace's decodes found in both, each bucket summed over its three groups (main, cards 0 to 7;
main, card 8; draft), then the step's GPU time and the server's unprofiled step. Standard library only.
"""
import csv
import os
import sys


def read(d):
    b, t = {}, {}
    with open(os.path.join(d, 'buckets.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['trace'] != 'deep' or not r['group'] or not r['ms_a_step']:
                continue
            k = (r['request'], r['where'])
            b[k] = b.get(k, 0.0) + float(r['ms_a_step'])
    with open(os.path.join(d, 'totals.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['trace'] == 'deep' and r['ms_a_step']:
                t[(r['request'], r['what'])] = float(r['ms_a_step'])
    return b, t


(b51, t51), (b77, t77) = read(sys.argv[1]), read(sys.argv[2])
reqs = [q for q in dict.fromkeys(k[0] for k in b77) if any(k[0] == q for k in b51)]
wheres = list(dict.fromkeys(k[1] for k in b77))
print('request\twhat\tR51_ms\tR77_ms\tR77_less_R51_ms')
for q in reqs:
    for w in wheres:
        a, c = b51.get((q, w)), b77.get((q, w))
        if a is not None and c is not None:
            print(f'{q}\t{w}\t{a:.3f}\t{c:.3f}\t{c - a:+.3f}')
    for w in ('GPU time of the step', "server's ms a step, unprofiled load"):
        a, c = t51.get((q, w)), t77.get((q, w))
        if a is not None and c is not None:
            print(f'{q}\t{w}\t{a:.2f}\t{c:.2f}\t{c - a:+.2f}')
