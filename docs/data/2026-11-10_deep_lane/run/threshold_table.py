#!/usr/bin/env python3
"""threshold_table.py RUN [--control ARM] [--out FILE.md] [--tsv FILE.tsv]: a deep-lane run of several arms of one
kernel switch read on the time of a step, with tokens a second and the draft's accepted share beside it.

RUN is a deep-lane results folder (bench/deep_lane: requests.csv and loads.tsv). Per depth and arm, every load by
value:
- the step on pass 2 (the same request again) and on pass 1 (the first request at the depth), ms: the request's
  generation time over its steps, a step being one pass of the main model (generated tokens minus accepted draft
  tokens);
- the load's tokens a second (both passes' tokens over their time) and the share of drafted tokens it accepted over
  both passes, since loads that write different replies accept different shares;
- the arm against the control on the step of pass 2, by bench/deep_lane/deep_table.py's rule: the spread is the
  larger of the two arms' differences between their loads, gain when the arm's mean is below the control's by more
  than the spread, loss when above by more than it, else negligible.
The TSV has one line per depth, arm and measure (step_pass2, step_pass1, tok_s): each load's value, spread, delta,
percent and class (the class only for step_pass2; tok/s is classed by deep_table.py's classes.tsv).
Python 3.9 or newer, standard library only."""
import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--control')
ap.add_argument('--out')
ap.add_argument('--tsv')
A = ap.parse_args()

with open(os.path.join(A.run, 'requests.csv'), newline='') as f:
    rows = list(csv.DictReader(f))
with open(os.path.join(A.run, 'loads.tsv'), newline='') as f:
    # a load that was run again (a resumed run) is listed twice: its last line is the one its requests come from
    loads = [r for r in {r['load']: r for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)}.values()
             if r['timed'] == '1']
arms = list(dict.fromkeys(r['arm'] for r in loads))
by_arm = {a: [r['load'] for r in loads if r['arm'] == a] for a in arms}
control = A.control or arms[0]
R = {(r['load'], r['pass'], int(r['depth'])): r for r in rows if r['kind'] == 'decode'}
depths = sorted({k[2] for k in R})


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def step(ld, p, d):
    r = R.get((ld, p, d))
    if not r:
        return None
    n = num(r['gen_n']) - num(r['draft_accepted'])
    return num(r['gen_ms']) / n if n > 0 and num(r['gen_ms']) > 0 else None


def tok_s(ld, d):
    rs = [R.get((ld, p, d)) for p in ('1', '2')]
    rs = [r for r in rs if r]
    n, ms = sum(num(r['gen_n']) for r in rs), sum(num(r['gen_ms']) for r in rs)
    return n * 1000 / ms if n and ms > 0 else None


def accepted(ld, d):
    rs = [r for r in (R.get((ld, p, d)) for p in ('1', '2')) if r]
    dn, da = sum(num(r['draft_n']) for r in rs), sum(num(r['draft_accepted']) for r in rs)
    return da / dn if dn else None


def mean(v):
    return sum(v) / len(v)


def verdict(ctl, arm, lower=True):
    ctl, arm = [v for v in ctl if v is not None], [v for v in arm if v is not None]
    if not ctl or not arm:
        return None, None, None, ''
    delta = mean(arm) - mean(ctl)
    pct = 100 * delta / mean(ctl)
    if len(ctl) < 2 or len(arm) < 2:
        return None, delta, pct, 'no class (one load)'
    s = max(max(ctl) - min(ctl), max(arm) - min(arm))
    better = -delta if lower else delta
    return s, delta, pct, 'gain' if better > s else 'loss' if better < -s else 'negligible'


def f2(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def sg(v, nd=2):
    return '' if v is None else f'{v:+.{nd}f}'


md = ['## The step against the threshold', '',
      f'Control `{control}`. The step on pass 2 (the same request again), ms, every load by value; the spread is the '
      'larger of the two arms\' differences between their loads; class: gain when the mean is below the control\'s by '
      'more than the spread, loss when above by more than it, else negligible. Beside it the step on pass 1, and each '
      'load\'s tok/s over both passes with the share of drafted tokens it accepted.', '',
      '| Depth (tokens) | Arm | Step, pass 2, ms | Spread | Against the control | | Class | Step, pass 1, ms | tok/s (accepted) |',
      '|---:|---|---|---:|---:|---:|---|---|---|']
tsv = [['measure', 'depth', 'arm', 'control', 'loads', 'control_loads', 'spread', 'delta', 'percent', 'class']]
for d in depths:
    for i, a in enumerate(arms):
        lds = by_arm[a]
        s2 = [step(ld, '2', d) for ld in lds]
        s1 = [step(ld, '1', d) for ld in lds]
        ts = [tok_s(ld, d) for ld in lds]
        c2 = [step(ld, '2', d) for ld in by_arm[control]]
        c1 = [step(ld, '1', d) for ld in by_arm[control]]
        ct = [tok_s(ld, d) for ld in by_arm[control]]
        sp2 = max(s2) - min(s2) if None not in s2 and len(s2) > 1 else None
        if not any(v is not None for v in s2 + s1 + ts):
            md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | — | | — | | | — | — |')
            continue
        if a == control:
            tail = ' control | | |'
        else:
            s, delta, pct, cls = verdict(c2, s2)
            tail = ' — | — | — |' if delta is None else f' {sg(delta)} ms | {sg(pct, 1)} % | {cls} |'
        md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {", ".join(f2(v) for v in s2)} | {f2(sp2)} |{tail} '
                  f'{", ".join(f2(v) for v in s1)} | '
                  f'{", ".join(f"{f2(t)} ({f2(accepted(ld, d), 3)})" for t, ld in zip(ts, lds))} |')
        if a != control:
            for name, vals, cvals, lower, classed in (('step_pass2', s2, c2, True, True), ('step_pass1', s1, c1, True, False),
                                                       ('tok_s', ts, ct, False, False)):
                s, delta, pct, cls = verdict(cvals, vals, lower)
                tsv.append([name, str(d), a, control, ' '.join(f2(v) for v in vals), ' '.join(f2(v) for v in cvals),
                            f2(s), f2(delta), f2(pct, 1), cls if classed else ''])
text = '\n'.join(md) + '\n'
if A.out:
    with open(A.out, 'w') as f:
        f.write(text)
else:
    print(text, end='')
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write(''.join('\t'.join(r) + '\n' for r in tsv))
