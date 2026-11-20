#!/usr/bin/env python3
"""draft_cost.py: the MTP draft's cost in prompt reading, from a deep-lane run (bench/deep_lane) with a draft arm and
an arm without it. Standard library only.

    python3 -I draft_cost.py OUT [--draft draft] [--nodraft nodraft] [--tsv primes.tsv] > cost.md

Reads OUT/requests.csv and takes the `prime` requests of the timed loads (every load but `warm`): each reads a chunk
on top of the previous depth's cache and generates one token. Per load and depth it gives the chunk (tokens read,
tokens cached), the server's prompt time as tok/s and ms a prompt token, the request's wall time as the client saw it
(prompt, the one token and anything the server does before it answers), and wall minus prompt time. Per depth it then
gives the draft's cost: the mean ms a prompt token with the draft minus without it, and the draft arm's prompt tok/s
against the arm without it in %, on the server's prompt time and on wall time. The same over the whole load (every
chunk's tokens over their summed time).
"""
import argparse
import csv
import sys
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--draft', default='draft')
ap.add_argument('--nodraft', default='nodraft')
ap.add_argument('--tsv', default='')
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(f'{a.src}/requests.csv')) if r['kind'] == 'prime' and r['load'] != 'warm']
if not rows:
    sys.exit('no timed prime requests')
loads = defaultdict(list)  # arm -> loads in run order
for r in rows:
    if r['load'] not in loads[r['arm']]:
        loads[r['arm']].append(r['load'])
depths = sorted({int(r['depth']) for r in rows})
by = {(r['load'], int(r['depth'])): r for r in rows}


def f(x, n=2):
    return f'{x:,.{n}f}'


def stats(load, depth):
    r = by.get((load, depth))
    if not r:
        return None
    n, ms, wall = int(r['prompt_n']), float(r['prompt_ms']), float(r['wall_s']) * 1000
    return dict(n=n, cached=int(r['cache_n']), ms=ms, wall=wall, tps=n / ms * 1000, mspt=ms / n,
                wtps=n / wall * 1000, wmspt=wall / n, after=wall - ms)


out = []
tsv = [['load', 'arm', 'depth', 'read', 'cached', 'prompt_ms', 'wall_ms', 'prompt_tok_s', 'ms_per_prompt_token',
        'wall_tok_s', 'wall_ms_per_prompt_token', 'wall_minus_prompt_ms']]
out.append('## Each prime (server prompt time; wall time as the client saw it)\n')
out.append('| Depth | Load | Arm | Read on cached | Prompt tok/s | ms a prompt token | Wall tok/s | Wall − prompt, ms |')
out.append('|---:|---|---|---|---:|---:|---:|---:|')
for d in depths:
    for arm in (a.draft, a.nodraft):
        for ld in loads[arm]:
            s = stats(ld, d)
            if not s:
                continue
            out.append(f"| {d:,} | {ld} | {arm} | {s['n']:,} on {s['cached']:,} | {f(s['tps'], 1)} | {f(s['mspt'], 3)} | "
                       f"{f(s['wtps'], 1)} | {f(s['after'], 0)} |")
            tsv.append([ld, arm, d, s['n'], s['cached'], f"{s['ms']:.2f}", f"{s['wall']:.0f}", f"{s['tps']:.2f}",
                        f"{s['mspt']:.4f}", f"{s['wtps']:.2f}", f"{s['wmspt']:.4f}", f"{s['after']:.0f}"])


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def arm_vals(arm, d, key):
    return [s[key] for s in (stats(ld, d) for ld in loads[arm]) if s]


out.append('\n## The draft\'s cost per depth (mean of each arm\'s loads)\n')
out.append('| Depth | Read on cached | ms a prompt token, draft | without | the draft adds, ms | Prompt tok/s, draft vs without | '
           'Wall ms a prompt token, draft | without | adds, ms | Wall tok/s vs without |')
out.append('|---:|---|---|---|---:|---:|---|---|---:|---:|')
whole = {}
for d in depths:
    s0 = stats(loads[a.draft][0], d)
    md, mn = arm_vals(a.draft, d, 'mspt'), arm_vals(a.nodraft, d, 'mspt')
    wd, wn = arm_vals(a.draft, d, 'wmspt'), arm_vals(a.nodraft, d, 'wmspt')
    if not md or not mn:
        continue
    td, tn = mean(arm_vals(a.draft, d, 'tps')), mean(arm_vals(a.nodraft, d, 'tps'))
    wtd, wtn = mean(arm_vals(a.draft, d, 'wtps')), mean(arm_vals(a.nodraft, d, 'wtps'))
    out.append(f"| {d:,} | {s0['n']:,} on {s0['cached']:,} | {', '.join(f(x, 3) for x in md)} | "
               f"{', '.join(f(x, 3) for x in mn)} | {f(mean(md) - mean(mn), 3)} | {f((td / tn - 1) * 100, 1)} % | "
               f"{', '.join(f(x, 3) for x in wd)} | {', '.join(f(x, 3) for x in wn)} | {f(mean(wd) - mean(wn), 3)} | "
               f"{f((wtd / wtn - 1) * 100, 1)} % |")
out.append('\n## The whole load (every chunk\'s tokens over their summed time)\n')
out.append('| Load | Arm | Tokens | Prompt s | Prompt tok/s | Wall s | Wall tok/s |')
out.append('|---|---|---:|---:|---:|---:|---:|')
for arm in (a.draft, a.nodraft):
    for ld in loads[arm]:
        ss = [s for s in (stats(ld, d) for d in depths) if s]
        if len(ss) < len(depths):
            continue
        n, ms, wall = sum(s['n'] for s in ss), sum(s['ms'] for s in ss), sum(s['wall'] for s in ss)
        whole.setdefault(arm, []).append((n / ms * 1000, n / wall * 1000))
        out.append(f'| {ld} | {arm} | {n:,} | {f(ms / 1000, 2)} | {f(n / ms * 1000, 1)} | {f(wall / 1000, 2)} | '
                   f'{f(n / wall * 1000, 1)} |')
if whole.get(a.draft) and whole.get(a.nodraft):
    pd, pn = mean([x[0] for x in whole[a.draft]]), mean([x[0] for x in whole[a.nodraft]])
    wd, wn = mean([x[1] for x in whole[a.draft]]), mean([x[1] for x in whole[a.nodraft]])
    out.append(f'\nThe draft arm against the arm without it over the whole load: {f((pd / pn - 1) * 100, 1)} % on the '
               f'server\'s prompt time, {f((wd / wn - 1) * 100, 1)} % on wall time.')
print('\n'.join(out))
if a.tsv:
    with open(a.tsv, 'w', newline='') as fh:
        csv.writer(fh, delimiter='\t', lineterminator='\n').writerows(tsv)
