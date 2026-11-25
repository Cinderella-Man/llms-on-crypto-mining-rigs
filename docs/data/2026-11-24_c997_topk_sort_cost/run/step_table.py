#!/usr/bin/env python3
"""step_table.py: the sampled pass of a step.sh run (SAMPLER_PASS=1), standard library only.

    python3 -I step_table.py OUT [--control unset] [--out table.md]

Reads OUT/requests.csv (bench/deep_lane/deep_client.py's line per request) and prints Markdown, also written to --out:
- per depth and arm, the greedy decode step (the two passes at temperature 0, top-k 1): each load's ms a token on pass 1
  and pass 2 and over both (their time over their tokens), with the spread, the arm against the control and its class;
- per depth and arm, the sampled pass's ms a token of every load by value (its generation time over its tokens), the
  spread between the arm's loads, the arm against the control and its class, by deep_table.py's rule (the spread is
  the larger of the two arms' differences between their slowest and fastest load; for a time, gain when the arm's mean
  is below the control's by more than the spread, loss when above by more than it, else negligible);
- per depth, each load's sampled reply (the md5 of its token ids) and whether the loads of each arm wrote one reply:
  the sampled pass asks with the server line's own sampler (Flint's preset: top-k 20, top-p 0.95, temperature 1.0) and
  one seed a depth, the same on every load, so with GGML_CUDA_TOP_K_SORT=1 the loads of one arm should agree.
bench/deep_lane/deep_table.py prints the same run in tok/s with the checks; at top-k 1 the card's top-k has one candidate
to sort, so only the sampled pass runs the sort at the preset's k.
"""
import argparse
import csv
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--control', default='unset')
ap.add_argument('--out', default='')
A = ap.parse_args()

with open(os.path.join(A.src, 'requests.csv'), newline='') as f:
    every = [r for r in csv.DictReader(f) if r['load'] != 'warm']
rows = [r for r in every if r['kind'] == 'sample']
G = {(r['load'], r['pass'], int(r['depth'])): r for r in every if r['kind'] == 'decode'}
if not rows:
    sys.exit('no sampled pass in requests.csv (SAMPLER_PASS=1)')
loads = list(dict.fromkeys(r['load'] for r in rows))
arm_of = {r['load']: r['arm'] for r in rows}
arms = list(dict.fromkeys(arm_of[ld] for ld in loads))
if A.control not in arms:
    sys.exit(f'no arm {A.control} among {", ".join(arms)}')
by_arm = {a: [ld for ld in loads if arm_of[ld] == a] for a in arms}
depths = sorted({int(r['depth']) for r in rows})
R = {(r['load'], int(r['depth'])): r for r in rows}
valid = all(r['valid'] == '1' for r in rows)


def ms_tok(r):
    n, ms = float(r['gen_n'] or 0), float(r['gen_ms'] or 0)
    return ms / n if n and ms > 0 else None


def reply(r):
    return r['ids_md5'] or r['text_md5']


def classed(vals, a, sp):
    own, base = [x for x in vals[a] if x is not None], [x for x in vals[A.control] if x is not None]
    if a == A.control or not own or not base or not valid:
        return '', ''
    dl = sum(own) / len(own) - sum(base) / len(base)
    pct = dl * 100 / (sum(base) / len(base))
    return (f'{dl:+.3f} ({pct:+.2f} %)',
            'no spread' if sp is None else 'gain' if dl < -sp else 'loss' if dl > sp else 'negligible')


def both(ld, d):
    rs = [G.get((ld, p, d)) for p in ('1', '2')]
    if not all(rs):
        return None
    n, ms = sum(float(r['gen_n'] or 0) for r in rs), sum(float(r['gen_ms'] or 0) for r in rs)
    return ms / n if n and ms > 0 else None


md = ['## The greedy decode step (step.sh)', '',
      'Per load: ms a token on pass 1, pass 2 → both passes (their time over their tokens). Class on both passes.', '',
      '| Depth (tokens) | Arm | Per load (ms a token) | Spread | Against ' + A.control + ' | Class |',
      '|---:|---|---|---:|---:|---|']
for d in depths:
    vals = {a: [both(ld, d) for ld in by_arm[a]] for a in arms}
    spreads = [max(v) - min(v) for v in ([x for x in vals[a] if x is not None] for a in arms) if len(v) > 1]
    sp = max(spreads) if spreads else None
    for i, a in enumerate(arms):
        cells = []
        for ld, v in zip(by_arm[a], vals[a]):
            p = [G.get((ld, q, d)) for q in ('1', '2')]
            pp = ', '.join(f'{ms_tok(r):.3f}' if r and ms_tok(r) else 'none' for r in p)
            cells.append(f'{ld}: {pp} → **{v:.3f}**' if v is not None else f'{ld}: none')
        own = [x for x in vals[a] if x is not None]
        own_sp = f'{max(own) - min(own):.3f}' if len(own) > 1 else ''
        delta, cls = classed(vals, a, sp)
        md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {"; ".join(cells)} | {own_sp} | {delta} | {cls} |')

md += ['', '## The sampled pass (step.sh, SAMPLER_PASS=1)', '',
      'One request a depth and load with the server line\'s own sampler and a fixed seed a depth; '
      + ('ms a token (generation time over tokens).' if valid else 'times not measurements (valid=0).'), '',
      '| Depth (tokens) | Arm | Per load (ms a token) | Spread | Against ' + A.control + ' | Class |',
      '|---:|---|---|---:|---:|---|']
for d in depths:
    vals = {a: [ms_tok(R[(ld, d)]) if (ld, d) in R else None for ld in by_arm[a]] for a in arms}
    spreads = [max(v) - min(v) for v in ([x for x in vals[a] if x is not None] for a in arms) if len(v) > 1]
    sp = max(spreads) if spreads else None
    for i, a in enumerate(arms):
        cells = '; '.join(f'{ld}: {v:.3f}' if v is not None else f'{ld}: none' for ld, v in zip(by_arm[a], vals[a]))
        own = [x for x in vals[a] if x is not None]
        own_sp = f'{max(own) - min(own):.3f}' if len(own) > 1 else ''
        delta, cls = classed(vals, a, sp)
        md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {cells} | {own_sp} | {delta} | {cls} |')

md += ['', '### The sampled replies', '', '| Depth (tokens) | Arm | Per load (md5 of the token ids) | One reply |',
       '|---:|---|---|---|']
for d in depths:
    for i, a in enumerate(arms):
        reps = [(ld, reply(R[(ld, d)])) for ld in by_arm[a] if (ld, d) in R]
        one = 'yes' if len({m for _, m in reps}) == 1 else f'no: {len({m for _, m in reps})} replies'
        cells = '; '.join(f'{ld}: `{m[:12]}`' for ld, m in reps)
        md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {cells} | {one} |')
gen = sorted({r['gen_n'] for r in rows})
md += ['', f'- Tokens out: {", ".join(gen)} on every sampled request.']
text = '\n'.join(md) + '\n'
print(text, end='')
if A.out:
    with open(A.out, 'w') as f:
        f.write(text)
