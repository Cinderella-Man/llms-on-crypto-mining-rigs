#!/usr/bin/env python3
"""run/variants.sh's results as one table: per variant, the makers'-chain acceptance and top-1 agreement on the code and
agent slices, and their change against the ref variant (in points).

  python3 -I variant_table.py LOGS NAME VARIANT ... > variants-NAME.tsv

LOGS holds var-NAME-<variant>-{code,agent}.tsv (run/mtp_variant.cpp's tables). Standard library only.
"""
import sys


def all_row(path):
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            r = line.rstrip('\n').split('\t')
            if r[0] == 'all':
                return dict(zip(head, r))
    return None


def main():
    logs, name, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
    cols = ['variant']
    for t in ('code', 'agent'):
        cols += [f'{t}_positions', f'{t}_acc_chain', f'{t}_top1_agree', f'{t}_acc_chain_vs_ref_pts']
    print('\t'.join(cols))
    ref = {}
    for v in ['ref'] + [v for v in variants if v != 'ref']:
        row = [v]
        for t in ('code', 'agent'):
            r = all_row(f'{logs}/var-{name}-{v}-{t}.tsv')
            if not r:
                row += [''] * 4
                continue
            a = float(r['acc_chain'])
            if v == 'ref':
                ref[t] = a
            row += [r['positions'], f'{a:.4f}', f'{float(r["top1_agree"]):.4f}',
                    f'{100 * (a - ref[t]):+.1f}' if t in ref else '']
        print('\t'.join(row))


if __name__ == '__main__':
    main()
