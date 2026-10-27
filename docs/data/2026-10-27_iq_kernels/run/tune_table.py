#!/usr/bin/env python3
"""Tables from round 2's two rig passes: run/tune_rig.sh's log (test-backend-ops perf: expert down and dense products)
and run/tune_rig_fused.sh's log (the fused expert gate/up under GGML_CUDA_PROFILE).

  python3 -I run/tune_table.py tune_rig.log tune_rig_fused.log OUT_PREFIX

Writes OUT_PREFIX.csv (one row per variant, product, shape, type and token count: microseconds per call, and where the
weights read per call are known, at one token or for a dense product, GB/s and picoseconds per weight) and
OUT_PREFIX.md (per variant: the one-token GPU time of a decoded token's grid-type expert products in Unsloth's
UD-Q2_K_XL of Ornith-1.5-122B, from the layer counts below, against llama.cpp's kernels; then every case's speedup).
The perf log's MUL_MAT_VEC_FUSION lines are skipped: perf mode times only a graph's last node, the GLU. Standard
library only.
"""
import csv
import re
import sys

BYTES_PER_WEIGHT = {'iq2_s': 82 / 256, 'iq3_xxs': 98 / 256}
# UD-Q2_K_XL's layers per expert product and type (docs/data/2026-10-27_iq_kernels/kernel_profile_ud.csv)
UD_LAYERS = {('gate_up', 'iq2_s'): 18, ('gate_up', 'iq3_xxs'): 14, ('down', 'iq2_s'): 7, ('down', 'iq3_xxs'): 25}
REF = 'llama.cpp'
GATE_UP_WEIGHTS = 2 * 8 * 1024 * 3072  # tune_rig_fused.sh's cases: gate and up, 8 experts of 1024 x 3072


def params(s):
    """key=value pairs of a test name, values may hold bracketed lists with commas"""
    out, depth, cur = {}, 0, ''
    for ch in s + ',':
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            k, _, v = cur.partition('=')
            out[k] = v
            cur = ''
        else:
            cur += ch
    return out


def row(variant, product, shape, typ, tokens, us, weights):
    r = dict(variant=variant, product=product, shape=shape, type=typ, tokens=tokens, us_per_call=us, gb_per_s='',
             ps_per_weight='')
    if weights:
        r['gb_per_s'] = round(weights * BYTES_PER_WEIGHT[typ] / (us * 1e-6) / 1e9, 1)
        r['ps_per_weight'] = round(us * 1e6 / weights, 2)
    return r


def main():
    perf_log, fused_log, prefix = sys.argv[1:4]
    rows, variant = [], None
    line_re = re.compile(r'^([A-Z_]+)\((.*)\): (\d+) runs - ([0-9.]+) us/run')
    for line in open(perf_log):
        line = line.rstrip('\n')
        if line.startswith('-- '):
            variant = REF if line[3:].startswith('llama.cpp') else line[3:]
            continue
        m = line_re.match(line)
        if not m or variant is None or m.group(1) == 'MUL_MAT_VEC_FUSION':
            continue
        op, p, us = m.group(1), params(m.group(2)), float(m.group(4))
        mm, n, k = int(p['m']), int(p['n']), int(p['k'])
        if op == 'MUL_MAT_ID':
            # distinct experts per call are random beyond one token, so the weights read are known at one token only
            w = int(p['n_used']) * mm * k if n == 1 else 0
            rows.append(row(variant, 'down', f'{mm}x{k}', p['type_a'], n, us, w))
        else:
            rows.append(row(variant, 'dense', f'{mm}x{k}', p['type_a'], n, us, mm * k))
    for line in open(fused_log):
        f = line.split()
        if len(f) < 5 or line.startswith('=='):
            continue
        v, typ, n, us = f[0], f[1], int(f[2]), float(f[3])
        rows.append(row(v, 'gate_up', '2x1024x3072', typ, n, us, GATE_UP_WEIGHTS if n == 1 else 0))
    rows.sort(key=lambda r: (r['variant'] == REF, r['product'], r['shape'], r['type'], r['tokens']))
    with open(prefix + '.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    key = lambda r: (r['product'], r['shape'], r['type'], r['tokens'])
    t = {(r['variant'],) + key(r): r['us_per_call'] for r in rows}
    one = {(r['variant'], r['product'], r['type']): r['us_per_call'] for r in rows
           if r['tokens'] == 1 and r['product'] in ('gate_up', 'down')}
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    cases = list(dict.fromkeys(key(r) for r in rows if r['shape'] != '4096x14336'))
    ud = {v: sum(n * one.get((v, prod, typ), float('nan')) for (prod, typ), n in UD_LAYERS.items()) / 1000
          for v in variants}
    out = ['| Variant | UD-Q2_K_XL, ms per decoded token (grid-type expert products, summed over layers) '
           '| vs llama.cpp |', '|---|---|---|']
    for v in sorted(variants, key=lambda v: ud[v]):
        out.append(f'| {v} | {ud[v]:.2f} | {ud[REF] / ud[v]:.2f}× |')
    out += ['', "Speedup over llama.cpp's kernels per case (product, type, tokens):", '',
            '| Variant | ' + ' | '.join(f'{p} {ty} {n}' for p, _, ty, n in cases) + ' |',
            '|---|' + '---|' * len(cases)]
    for v in variants:
        if v == REF:
            continue
        cells = []
        for c in cases:
            a, b = t.get((REF,) + c), t.get((v,) + c)
            cells.append(f'{a / b:.2f}' if a and b else '')
        out.append(f'| {v} | ' + ' | '.join(cells) + ' |')
    with open(prefix + '.md', 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
