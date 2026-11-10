#!/usr/bin/env python3
"""exact_repeat.py: two loads of one line on the same prompts, token by token (registry Q1175).

  python3 exact_repeat.py <replies of load A>.json <replies of load B>.json --label "<what the two loads are>" \
      [--append] --out repeat.tsv

Both files are exact_client.py's replies_<arm>.json. Per depth: whether the two loads wrote the same tokens, the first
token where they part, how far the log-probability of the same token moves between the loads over the tokens before
that (from each load's request with probabilities: how many are exactly equal, the median and the largest shift), and
at the parting token each load's lead of its own token over the other load's.
"""
import argparse
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('a')
ap.add_argument('b')
ap.add_argument('--label', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--append', action='store_true')
A = ap.parse_args()


def load(path):
    with open(path) as f:
        return {int(d): r for d, r in json.load(f)['replies'].items()}


def aligned(rep):  # the probabilities of each token of the request that listed them, None where the server lists none
    out, j = [], 0
    for t in rep['genp_ids']:
        if j < len(rep['probs']) and rep['probs'][j]['id'] == t:
            out.append(rep['probs'][j])
            j += 1
        else:
            out.append(None)
    return out


def lead(pos, a, b):
    la = next((lp for t, lp, _ in pos['top'] if t == a), None)
    lb = next((lp for t, lp, _ in pos['top'] if t == b), None)
    return '' if la is None or lb is None else f'{la - lb:.4f}'


cols = ['loads', 'depth', 'same_tokens', 'first_differing_token', 'tokens_compared', 'exactly_equal',
        'median_abs_shift', 'max_abs_shift', 'first_token_shift', 'lead_of_own_token_load_a', 'lead_of_own_token_load_b']
ra, rb = load(A.a), load(A.b)
rows = []
for d in sorted(set(ra) & set(rb)):
    a, b = ra[d], rb[d]
    n = min(len(a['genp_ids']), len(b['genp_ids']))
    c = next((i for i in range(n) if a['genp_ids'][i] != b['genp_ids'][i]), n)
    pa, pb = aligned(a), aligned(b)
    shifts = [abs(pa[i]['lp'] - pb[i]['lp']) for i in range(c) if pa[i] and pb[i]]
    row = {'loads': A.label, 'depth': d, 'same_tokens': 'yes' if c == n and a['ids'] == b['ids'] else 'no',
           'first_differing_token': 'none' if c == n else c, 'tokens_compared': len(shifts),
           'exactly_equal': sum(s == 0 for s in shifts),
           'median_abs_shift': f'{statistics.median(shifts):.4f}' if shifts else '',
           'max_abs_shift': f'{max(shifts):.4f}' if shifts else '',
           'first_token_shift': f'{abs(pa[0]["lp"] - pb[0]["lp"]):.4f}' if c > 0 and pa[0] and pb[0] else ''}
    if c < n and pa[c] and pb[c]:
        row['lead_of_own_token_load_a'] = lead(pa[c], a['genp_ids'][c], b['genp_ids'][c])
        row['lead_of_own_token_load_b'] = lead(pb[c], b['genp_ids'][c], a['genp_ids'][c])
    rows.append(row)
with open(A.out, 'a' if A.append else 'w') as f:
    if not A.append:
        f.write('\t'.join(cols) + '\n')
    for r in rows:
        f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print('\t'.join(str(r.get(c, '')) for c in cols))
