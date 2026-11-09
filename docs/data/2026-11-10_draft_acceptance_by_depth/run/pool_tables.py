#!/usr/bin/env python3
"""pool_tables.py: the draft's counters per context depth over several deep-lane runs, one run per tail.

    python3 -I pool_tables.py TAIL_DIR [TAIL_DIR ...] --out OUT [--arm draft] [--ref 8192]

Each TAIL_DIR is the results folder of one bench/deep_lane run (its requests.csv); the folder's name is the tail's
name. Standard library only. No time is read: only the token counts, the server's draft counters and the reply hashes.

Per request the server reports gen_n (tokens generated), draft_n (tokens the draft proposed) and draft_accepted (the
proposed tokens the main model kept). acceptance = draft_accepted / draft_n. A step is one pass of the main model: it
yields one token of its own and the draft tokens it accepts, so steps = gen_n - draft_accepted and tokens a step =
gen_n / steps.

Writes to OUT:
- by_tail.tsv: one row per tail, depth and pass (1 and 2: greedy; s: the server line's sampler) of --arm;
- pooled.tsv: per depth and pass, the counters summed over the tails, the acceptance with a 95 % Wilson interval,
  tokens a step, and the difference in acceptance to the depth --ref with a 95 % interval (Newcombe's method 10);
- reply_check.tsv: per tail and depth, whether every arm's greedy passes wrote the same reply (the hash of its token
  ids), for the tails that ran more than one arm;
- summary.md: the same as Markdown tables.
The intervals treat every drafted token as an independent trial. Drafted tokens are not independent (the tokens of
one draft are accepted in order, and all come from one reply per tail), so the intervals are a lower bound on the
uncertainty; the per-tail values show the spread between replies.
"""
import argparse
import csv
import math
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('tails', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--arm', default='draft')
ap.add_argument('--ref', type=int, default=8192)
A = ap.parse_args()
Z = 1.959964
PASSES = [('1', 'greedy, pass 1'), ('2', 'greedy, pass 2'), ('s', 'sampler')]


def wilson(k, n):
    if not n:
        return None, None
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return mid - half, mid + half


def newcombe(k1, n1, k2, n2):
    """95 % interval of p1 - p2 from the two Wilson intervals (Newcombe 1998, method 10)."""
    if not n1 or not n2:
        return None, None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def f(v, nd=3, sign=False):
    return '' if v is None else f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'


names, rows = [], {}  # rows[(tail, arm, kind, pass, depth)] = the request's CSV line
for path in A.tails:
    name = os.path.basename(os.path.normpath(path))
    names.append(name)
    with open(os.path.join(path, 'requests.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            if r['kind'] in ('decode', 'sample'):
                rows[(name, r['arm'], r['kind'], r['pass'], int(r['depth']))] = r
depths = sorted({k[4] for k in rows if k[1] == A.arm})
if not depths:
    sys.exit(f'no decode request of arm {A.arm} in {", ".join(A.tails)}')
os.makedirs(A.out, exist_ok=True)


def cell(tail, pas, depth):
    """(gen_n, drafted, accepted) of one request, or None."""
    r = rows.get((tail, A.arm, 'sample' if pas == 's' else 'decode', pas, depth))
    return (int(r['gen_n']), int(r['draft_n']), int(r['draft_accepted'])) if r else None


def measures(gen, drafted, accepted):
    steps = gen - accepted
    return (accepted / drafted if drafted else None), steps, (gen / steps if steps > 0 else None)


with open(os.path.join(A.out, 'by_tail.tsv'), 'w') as fh:
    fh.write('tail\tarm\tdepth\tpass\tprompt_tokens\tgen_n\tdrafted\taccepted\tacceptance\tsteps\ttokens_per_step\t'
             'reply_ids_md5\n')
    for t in names:
        for d in depths:
            for pas, _ in PASSES:
                c = cell(t, pas, d)
                if not c:
                    continue
                r = rows[(t, A.arm, 'sample' if pas == 's' else 'decode', pas, d)]
                acc, steps, tps = measures(*c)
                fh.write(f'{t}\t{A.arm}\t{d}\t{pas}\t{r["prompt_tokens"]}\t{c[0]}\t{c[1]}\t{c[2]}\t{f(acc, 4)}\t{steps}\t'
                         f'{f(tps)}\t{r["ids_md5"] or r["text_md5"]}\n')

pooled = {}  # pooled[(pass, depth)] = (tails, gen, drafted, accepted)
for pas, _ in PASSES:
    for d in depths:
        cs = [c for c in (cell(t, pas, d) for t in names) if c]
        if cs:
            pooled[(pas, d)] = (len(cs), sum(c[0] for c in cs), sum(c[1] for c in cs), sum(c[2] for c in cs))
with open(os.path.join(A.out, 'pooled.tsv'), 'w') as fh:
    fh.write('pass\tdepth\ttails\tgen_n\tdrafted\taccepted\tacceptance\tci95_low\tci95_high\tsteps\ttokens_per_step\t'
             f'acceptance_minus_depth_{A.ref}\tdiff_ci95_low\tdiff_ci95_high\n')
    for pas, _ in PASSES:
        for d in depths:
            if (pas, d) not in pooled:
                continue
            n, gen, drafted, accepted = pooled[(pas, d)]
            acc, steps, tps = measures(gen, drafted, accepted)
            lo, hi = wilson(accepted, drafted)
            diff = dlo = dhi = None
            ref = pooled.get((pas, A.ref))
            if ref and d != A.ref and drafted and ref[2]:
                diff = acc - ref[3] / ref[2]
                dlo, dhi = newcombe(accepted, drafted, ref[3], ref[2])
            fh.write(f'{pas}\t{d}\t{n}\t{gen}\t{drafted}\t{accepted}\t{f(acc, 4)}\t{f(lo, 4)}\t{f(hi, 4)}\t{steps}\t'
                     f'{f(tps)}\t{f(diff, 4, True)}\t{f(dlo, 4, True)}\t{f(dhi, 4, True)}\n')

checks = []  # (tail, depth, arms, same)
for t in names:
    arms = sorted({k[1] for k in rows if k[0] == t})
    if len(arms) < 2:
        continue
    for d in depths:
        per_arm = {a: sorted({r['ids_md5'] or r['text_md5'] for k, r in rows.items()
                              if k[0] == t and k[1] == a and k[2] == 'decode' and k[4] == d}) for a in arms}
        hashes = {h for hs in per_arm.values() for h in hs}
        checks.append((t, d, arms, len(hashes) == 1, per_arm))
with open(os.path.join(A.out, 'reply_check.tsv'), 'w') as fh:
    fh.write('tail\tdepth\tarms\tsame_greedy_reply\treply_ids_md5_per_arm\n')
    for t, d, arms, same, per_arm in checks:
        fh.write(f'{t}\t{d}\t{" ".join(arms)}\t{"yes" if same else "no"}\t'
                 + '; '.join(f'{a}: {" ".join(per_arm[a])}' for a in arms) + '\n')


def tail_cell(c):
    if not c:
        return '—'
    acc, _, tps = measures(*c)
    return f'{c[2]} of {c[1]} ({f(acc)}), {f(tps, 2)}' if c[1] else f'no draft, {f(tps, 2)}'


md = [f'## The draft by depth: arm `{A.arm}`, tails {", ".join(names)}', '',
      'Per tail: draft tokens accepted of drafted (the share), then tokens a step. Pooled: the counters summed over '
      'the tails, the acceptance with a 95 % Wilson interval, tokens a step, and the acceptance minus the acceptance '
      f'at depth {A.ref:,} with a 95 % interval. The intervals count every drafted token as an independent trial.']
same12 = all(cell(t, '1', d) == cell(t, '2', d) for t in names for d in depths)
for pas, label in PASSES:
    if pas == '2' and same12:
        md += ['', '### Greedy, pass 2', '', 'The same counters as pass 1 at every depth and tail.']
        continue
    if not any((pas, d) in pooled for d in depths):
        continue
    md += ['', f'### {label[0].upper()}{label[1:]}', '',
           '| Depth (tokens) | ' + ' | '.join(names) + ' | Pooled | Acceptance | 95 % interval | Tokens a step | '
           f'Against {A.ref:,} | 95 % interval |', '|---:|' + '---|' * len(names) + '---|---:|---|---:|---:|---|']
    for d in depths:
        if (pas, d) not in pooled:
            continue
        n, gen, drafted, accepted = pooled[(pas, d)]
        acc, steps, tps = measures(gen, drafted, accepted)
        lo, hi = wilson(accepted, drafted)
        diff, span = '—', '—'
        ref = pooled.get((pas, A.ref))
        if ref and d != A.ref and drafted and ref[2]:
            dlo, dhi = newcombe(accepted, drafted, ref[3], ref[2])
            diff, span = f(acc - ref[3] / ref[2], 3, True), f'{f(dlo, 3, True)} to {f(dhi, 3, True)}'
        md.append(f'| {d:,} | ' + ' | '.join(tail_cell(cell(t, pas, d)) for t in names)
                  + f' | {accepted} of {drafted} | {f(acc)} | {f(lo)} to {f(hi)} | {f(tps, 2)} | {diff} | {span} |')
if checks:
    md += ['', '### The greedy reply, arm against arm', '']
    for t in sorted({c[0] for c in checks}):
        mine = [c for c in checks if c[0] == t]
        off = [f'{c[1]:,}' for c in mine if not c[3]]
        md.append(f'- {t} ({", ".join(mine[0][2])}): '
                  + (f'**the replies differ at depth {", ".join(off)}.**' if off
                     else f'the same reply on every greedy pass of every arm at all {len(mine)} depths.'))
text = '\n'.join(md) + '\n'
with open(os.path.join(A.out, 'summary.md'), 'w') as fh:
    fh.write(text)
sys.stdout.write(text)
