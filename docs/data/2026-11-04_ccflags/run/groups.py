#!/usr/bin/env python3
# Q671: the census of analyze.py per kernel group instead of per file: the groups the P106 timing candidates are chosen
# from (the grid types' 2-4 token kernels, the multi-token Q2_0 kernels by token count, the k-column kernels by type, ...),
# for every arm: kernels, mean and maximum registers, local-memory instructions (STL + LDL), mean resident blocks per SM
# (5-SM GP106, 128-thread blocks unless the file's threads say otherwise), instructions, and how many kernels' machine code
# differs from the base arm's.
# Usage: groups.py <kernels.tsv.gz> <out.tsv>
import csv
import gzip
import re
import sys
from collections import defaultdict

TYPES = {'2': 'Q4_0', '3': 'Q4_1', '6': 'Q5_0', '8': 'Q8_0', '12': 'Q4_K', '13': 'Q5_K', '14': 'Q6_K', '18': 'IQ3_XXS',
         '22': 'IQ2_S', '23': 'IQ4_XS'}


def group(r):
    n, f = r['name'], r['file']
    m = re.search(r'mul_mat_vec_q_pascal(_mc|_qkv)?<\(ggml_type\)(\d+)', n)
    if m:
        t = TYPES.get(m.group(2), 'type ' + m.group(2))
        kind = {'_mc': '2-4 tokens', '_qkv': 'Q, K, V in one launch', None: 'one token'}[m.group(1)]
        if m.group(2) in ('18', '22'):
            return f'{t} (grid type), {kind}'
        return f'other types, {kind}'
    m = re.search(r'mul_mat_vec_q2_0_pascal_mtok<\(bool\)(\d+), \(int\)(\d+), \(int\)(\d+)>', n)
    if m:
        return f'Q2_0 experts, {m.group(3)} tokens' + (', fused gate/up' if m.group(1) == '1' else '')
    if f.startswith('mmvq-pascal-kcol-') and 'mul_mat_kcol' in n:
        return 'k-column ' + f[len('mmvq-pascal-kcol-'):-3]
    if f.startswith('mmvq-pascal-kcol-'):
        return 'k-column, activation records'
    if f == 'mmvf-pascal.cu':
        return 'BF16'
    if f == 'fattn-pascal.cu':
        return 'decode attention (fattn-pascal)'
    if f == 'fattn-tile-pascal.cu':
        return 'prompt attention tile (fattn-tile-pascal)'
    if f == 'gdn-fused.cu':
        return 'GDN fusions'
    if f == 'moe-persistent.cu':
        return 'persistent MoE'
    return 'other'


def main():
    rows = list(csv.DictReader(gzip.open(sys.argv[1], 'rt'), delimiter='\t'))
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[group(r)][r['arm']].append(r)
    with open(sys.argv[2], 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['group', 'arm', 'kernels', 'regs_mean', 'regs_max', 'local_insns', 'blocks_sm_mean', 'insns', 'kernels_differing_from_base'])
        for g in sorted(agg):
            for a in arms:
                rs = agg[g].get(a)
                if not rs:
                    continue
                w.writerow([g, a, len(rs), round(sum(int(r['regs']) for r in rs) / len(rs), 1), max(int(r['regs']) for r in rs),
                            sum(int(r['stl']) + int(r['ldl']) for r in rs), round(sum(int(r['blocks_sm']) for r in rs) / len(rs), 2),
                            sum(int(r['insns']) for r in rs), sum(1 for r in rs if r['same_as_base'] == '0')])


main()
