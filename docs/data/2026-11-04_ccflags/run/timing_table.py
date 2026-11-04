#!/usr/bin/env python3
# Q671: the tables of one p106_timing.sh run: per case and arm the median over the passes of the time per call, the change
# against the base arm, and the base arm's spread between passes (a change smaller than the spread is not one).
# Reads $OUT/<arm>/<group>.<pass>.txt (test-backend-ops perf) and $OUT/<arm>/<group>.<pass>.csv (bench/mmvq_pascal --csv).
# Usage: timing_table.py <OUT dir> [<base arm, default base>]
# Writes <OUT>/summary_tbo.tsv and <OUT>/summary_bench.tsv (and prints them).
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

ANSI = re.compile(r'\x1b\[[0-9;]*m')
LINE = re.compile(r'^\s*([A-Z_0-9]+)\((.*)\):\s+(\d+) runs - +([\d.]+) us/run')
KEYS = ['shape', 'type', 'K', 'N', 'experts', 'used', 'route', 'tokens', 'variant', 'G', 'R', 'gcap', 'rb', 'staging', 'L',
        'itt', 'gen']


def split_name(path):
    arm = os.path.basename(os.path.dirname(path))
    group, rep, _ = os.path.basename(path).rsplit('.', 2)
    return arm, group, int(rep)


def summarize(samples, base, out_path, key_names):
    """samples: {(group, case): {arm: [us per pass]}}"""
    arms = sorted({a for v in samples.values() for a in v}, key=lambda a: (a != base, a))
    with open(out_path, 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        head = ['group', 'case', 'base_noise_pct']
        for a in arms:
            head += [f'{a}_us'] + ([f'{a}_vs_{base}_pct'] if a != base else [])
        w.writerow(head)
        for (group, case), per in sorted(samples.items()):
            b = per.get(base)
            if not b:
                continue
            bm = statistics.median(b)
            noise = 100 * (max(b) - min(b)) / bm if len(b) > 1 and bm else ''
            row = [group, case, f'{noise:.1f}' if noise != '' else '']
            for a in arms:
                v = per.get(a)
                m = statistics.median(v) if v else None
                row.append(f'{m:.2f}' if m is not None else '')
                if a != base:
                    row.append(f'{100 * (m - bm) / bm:+.1f}' if m is not None and bm else '')
            w.writerow(row)
    print(open(out_path).read())


def main():
    out = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else 'base'
    tbo = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(out, '*', '*.*.txt'))):
        arm, group, rep = split_name(p)
        for line in open(p, errors='replace'):
            m = LINE.match(ANSI.sub('', line))
            if m:
                tbo[(group, f'{m.group(1)}({m.group(2)})')][arm].append(float(m.group(4)))
    summarize(tbo, base, os.path.join(out, 'summary_tbo.tsv'), None)
    bench = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(out, '*', '*.*.csv'))):
        arm, group, rep = split_name(p)
        for r in csv.DictReader(open(p)):
            col = 'us_median' if 'us_median' in r else 'us_per_step' if 'us_per_step' in r else None
            if col is None or not r.get(col):
                continue
            case = ','.join(f'{k}={r[k]}' for k in KEYS if k in r)
            bench[(group, case)][arm].append(float(r[col]))
    if bench:
        summarize(bench, base, os.path.join(out, 'summary_bench.tsv'), None)


main()
