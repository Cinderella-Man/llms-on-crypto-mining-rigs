#!/usr/bin/env python3
"""v4c_table.py: one deep-lane run read as an arm against the control in decode and in prompts, per depth and load.

    python3 -I v4c_table.py OUT [--control rel] [--arm v4c] [--md summary.md] [--tsv loads.tsv] [--classes classes.tsv]

OUT is a deep-lane results folder (bench/deep_lane: requests.csv, loads.tsv, ram_watchdog-<load>.tsv). Per depth and
timed load:
- decode: the two greedy passes' tokens over their time (tok/s) and its ms a token; the step on pass 1 and on pass 2
  (generation time over steps, a step being generated tokens minus accepted draft tokens); the draft's share accepted
  and tokens a step over both passes;
- the sampler pass (deep_lane SAMPLER_PASS=1: the server line's own sampler and a fixed seed, the only request where a
  temperature above 0 lets probabilistic verification act): tok/s, share accepted, tokens a step, ms a step;
- prompts: the prime's tokens read over their time (tok/s), with the tokens read and cached; at depth 0, which has
  no prime, the first decode request's read of the 64-token tail on an empty cache; over all depths the primes alone.
Then each metric's class of the arm against the control, per depth and over all depths, by deep_table.py's rule: the
spread is the larger of the two arms' differences between their fastest and slowest load; the arm is a gain when its
mean is better than the control's by more than the spread, a loss when worse by more than it, else negligible. Last,
per load, the host's lowest MemAvailable and the swap in use, from the RAM watchdog's samples (one a second).
Standard library only.
"""
import argparse
import csv
import glob
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--control', default='rel')
ap.add_argument('--arm', default='v4c')
ap.add_argument('--md', default='')
ap.add_argument('--tsv', default='')
ap.add_argument('--classes', default='')
A = ap.parse_args()
MIN_READ = 32

with open(os.path.join(A.src, 'requests.csv'), newline='') as f:
    rows = list(csv.DictReader(f))
loads = {}
with open(os.path.join(A.src, 'loads.tsv'), newline='') as f:
    for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
        loads[r['load']] = r  # a resumed run lists a load again: its last line counts
R = {(r['load'], r['kind'], r['pass'], int(r['depth'])): r for r in rows}
timed = [n for n, ld in loads.items() if ld['timed'] == '1' and any(r['load'] == n and r['kind'] == 'decode' for r in rows)]
by_arm = {a: [n for n in timed if loads[n]['arm'] == a] for a in (A.control, A.arm)}
if not by_arm[A.control] or not by_arm[A.arm]:
    sys.exit(f'need timed loads of {A.control} and {A.arm}; found {", ".join(timed)}')
depths = sorted({k[3] for k in R if k[0] in timed and k[1] == 'decode'})
if not all(r['valid'] == '1' for r in rows if r['load'] in timed):
    sys.exit('a timed request has valid=0: its times are not measurements')


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def get(ld, kind, p, d):
    return R.get((ld, kind, p, d))


def tps(rs):
    rs = [r for r in rs if r]
    n, ms = sum(num(r['gen_n']) for r in rs), sum(num(r['gen_ms']) for r in rs)
    return n * 1000 / ms if rs and ms > 0 else None


def steps(r):
    return num(r['gen_n']) - num(r['draft_accepted'])


def step(rs):
    rs = [r for r in rs if r]
    s, ms = sum(steps(r) for r in rs), sum(num(r['gen_ms']) for r in rs)
    return ms / s if rs and s > 0 else None


def share(rs):
    rs = [r for r in rs if r]
    n, a = sum(num(r['draft_n']) for r in rs), sum(num(r['draft_accepted']) for r in rs)
    return a / n if n else None


def per_step(rs):
    rs = [r for r in rs if r]
    s = sum(steps(r) for r in rs)
    return sum(num(r['gen_n']) for r in rs) / s if s > 0 else None


def prompt(rs):
    rs = [r for r in rs if r and num(r['prompt_n']) >= MIN_READ]
    n, ms = sum(num(r['prompt_n']) for r in rs), sum(num(r['prompt_ms']) for r in rs)
    return n * 1000 / ms if rs and ms > 0 else None


def greedy(ld, ds):
    return [get(ld, 'decode', p, d) for d in ds for p in ('1', '2')]


def sample(ld, ds):
    return [get(ld, 'sample', 's', d) for d in ds]


def primes(ld, ds):
    """The prompt reading at these depths: each depth's prime; a depth without one (depth 0, the tail alone) counts
    its first decode request's read when it is the only depth asked for. Over several depths: their primes alone."""
    rs = [get(ld, 'prime', '0', d) for d in ds]
    if len(ds) == 1 and not (rs[0] and num(rs[0]['prompt_n']) >= MIN_READ):
        rs = [get(ld, 'decode', '1', ds[0])]
    return rs


# metric: (label, unit, decimals, lower is better, value(load, depths))
METRICS = [
    ('decode', 'decode tok/s (greedy, both passes)', 'tok/s', 2, False, lambda ld, ds: tps(greedy(ld, ds))),
    ('decode_ms_token', 'decode ms a token', 'ms', 2, True,
     lambda ld, ds: (lambda v: 1000 / v if v else None)(tps(greedy(ld, ds)))),
    ('step_pass1', 'the step on pass 1, ms', 'ms', 2, True, lambda ld, ds: step([get(ld, 'decode', '1', d) for d in ds])),
    ('step_pass2', 'the step on pass 2 (the repeated request), ms', 'ms', 2, True,
     lambda ld, ds: step([get(ld, 'decode', '2', d) for d in ds])),
    ('share', 'share of drafted tokens accepted (greedy)', '', 4, False, lambda ld, ds: share(greedy(ld, ds))),
    ('tokens_step', 'tokens a step (greedy)', '', 3, False, lambda ld, ds: per_step(greedy(ld, ds))),
    ('sample', 'sampler pass tok/s', 'tok/s', 2, False, lambda ld, ds: tps(sample(ld, ds))),
    ('sample_share', 'sampler pass share accepted', '', 4, False, lambda ld, ds: share(sample(ld, ds))),
    ('sample_tokens_step', 'sampler pass tokens a step', '', 3, False, lambda ld, ds: per_step(sample(ld, ds))),
    ('sample_step', 'sampler pass ms a step', 'ms', 2, True, lambda ld, ds: step(sample(ld, ds))),
    ('prompt', 'prompt tok/s (the prime)', 'tok/s', 1, False, lambda ld, ds: prompt(primes(ld, ds))),
]


def fmt(v, nd):
    return '—' if v is None else f'{v:,.{nd}f}'


def verdict(ctl, arm, lower):
    ctl, arm = [v for v in ctl if v is not None], [v for v in arm if v is not None]
    if not ctl or not arm:
        return None, None, None, '—'
    mc, ma = sum(ctl) / len(ctl), sum(arm) / len(arm)
    delta, pct = ma - mc, 100 * (ma - mc) / mc if mc else None
    if len(ctl) < 2 or len(arm) < 2:
        return None, delta, pct, 'no class (one load)'
    s = max(max(ctl) - min(ctl), max(arm) - min(arm))
    better = -delta if lower else delta
    return s, delta, pct, 'gain' if better > s else 'loss' if better < -s else 'negligible'


groups = [(f'{d:,}', [d]) for d in depths] + [('all depths', depths)]
long_rows, class_rows = [], []
md = [f'## `{A.arm}` against `{A.control}`, decode and prompts', '',
      f'Timed loads, in their order: {", ".join(timed)}. Every value is one load.', '']
for a in (A.control, A.arm):
    ld = loads[by_arm[a][0]]
    own = ', '.join(f'`{ld[k]}`' for k in ('arm_env', 'arm_args') if ld.get(k))
    md.append(f'- `{a}`: {own or "the preset as it is"}')

md += ['', '### Per depth and load', '',
       '| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted '
       '| Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |',
       '|---:|---|---|---:|---|---:|---:|---|---|']
for d in depths:
    first = True
    for a in (A.control, A.arm):
        for ld in by_arm[a]:
            g1, g2 = get(ld, 'decode', '1', d), get(ld, 'decode', '2', d)
            v = tps([g1, g2])
            s = get(ld, 'sample', 's', d)
            pr = primes(ld, [d])[0]
            pr_cell = '—'
            if pr and num(pr['prompt_n']) >= MIN_READ:
                pr_cell = f'{fmt(prompt([pr]), 1)} ({int(num(pr["prompt_n"])):,} on {int(num(pr["cache_n"])):,})'
            smp = '—' if not s else (f'{fmt(tps([s]), 2)}, {fmt(share([s]), 4)}, {fmt(per_step([s]), 3)}, '
                                     f'{fmt(step([s]), 2)}')
            md.append(f'| {f"{d:,}" if first else ""} | {ld} | {fmt(tps([g1]), 2)}, {fmt(tps([g2]), 2)} → **{fmt(v, 2)}** '
                      f'| {fmt(1000 / v if v else None, 2)} | {fmt(step([g1]), 2)}, {fmt(step([g2]), 2)} '
                      f'| {fmt(share([g1, g2]), 4)} | {fmt(per_step([g1, g2]), 3)} | {smp} | {pr_cell} |')
            first = False

md += ['', f'### `{A.arm}` against `{A.control}`: classes', '',
       'Per metric: each load by value (the control\'s, then the arm\'s), the spread (the larger of the two arms\' '
       'differences between their two loads), the arm against the control and the class. For a time less is better.',
       '', f'| Depth (tokens) | Metric | `{A.control}` loads | `{A.arm}` loads | Spread | Against `{A.control}` | % | Class |',
       '|---:|---|---|---|---:|---:|---:|---|']
for label, ds in groups:
    first = True
    for key, name, unit, nd, lower, fn in METRICS:
        cv = [fn(ld, ds) for ld in by_arm[A.control]]
        av = [fn(ld, ds) for ld in by_arm[A.arm]]
        for a, vals in ((A.control, cv), (A.arm, av)):
            for ld, v in zip(by_arm[a], vals):
                long_rows.append([label.replace(',', ''), a, ld, key, fmt(v, nd).replace(',', '')])
        if all(v is None for v in cv + av):
            continue
        s, delta, pct, cls = verdict(cv, av, lower)
        class_rows.append([key, label.replace(',', ''), A.arm, A.control, ' '.join(fmt(v, nd).replace(',', '') for v in cv),
                           ' '.join(fmt(v, nd).replace(',', '') for v in av), fmt(s, nd).replace(',', ''),
                           fmt(delta, nd).replace(',', ''), fmt(pct, 2), cls])
        md.append(f'| {label if first else ""} | {name} | {", ".join(fmt(v, nd) for v in cv)} '
                  f'| {", ".join(fmt(v, nd) for v in av)} | {fmt(s, nd)} '
                  f'| {"—" if delta is None else f"{delta:+,.{nd}f}"}{(" " + unit) if unit and delta is not None else ""} '
                  f'| {"—" if pct is None else f"{pct:+.2f} %"} | {cls} |')
        first = False

wd = {}
for path in sorted(glob.glob(os.path.join(A.src, 'ram_watchdog-*.tsv'))):
    name = os.path.basename(path)[len('ram_watchdog-'):-len('.tsv')]
    with open(path) as f:
        wd[name] = [[float(x) for x in line.split('\t')[:8]] for line in f if line[:1].isdigit()]
md += ['', '### The host\'s memory per load', '',
       'From the RAM watchdog\'s samples, one a second: the lowest MemAvailable, the most swap in use and its change '
       'over the load, the highest memory pressure (psi full, 10 s average, %).', '',
       '| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |',
       '|---|---:|---:|---:|---:|---:|']
for ld in by_arm[A.control] + by_arm[A.arm]:
    s = wd.get(ld)
    if not s:
        md.append(f'| {ld} | 0 | — | — | — | — |')
        continue
    md.append(f'| {ld} | {len(s)} | {min(r[1] for r in s):,.0f} | {max(r[4] for r in s):,.0f} '
              f'| {s[-1][4] - s[0][4]:+,.0f} | {max(r[7] for r in s):.2f} |')
    for key, v in (('min_avail_mib', min(r[1] for r in s)), ('max_swap_used_mib', max(r[4] for r in s)),
                   ('swap_change_mib', s[-1][4] - s[0][4]), ('max_psi_full', max(r[7] for r in s))):
        long_rows.append(['load', loads[ld]['arm'], ld, key, f'{v:.2f}'])

text = '\n'.join(md) + '\n'
if A.md:
    with open(A.md, 'w') as f:
        f.write(text)
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write('depth\tarm\tload\tmetric\tvalue\n' + ''.join('\t'.join(r) + '\n' for r in long_rows))
if A.classes:
    with open(A.classes, 'w') as f:
        f.write('metric\tdepth\tarm\tcontrol\tcontrol_loads\tarm_loads\tspread\tdelta\tpercent\tclass\n'
                + ''.join('\t'.join(r) + '\n' for r in class_rows))
sys.stdout.write(text)
