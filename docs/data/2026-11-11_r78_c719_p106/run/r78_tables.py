#!/usr/bin/env python3
"""r78_tables.py: the tables of R78 (Q1284), C719's QSA pool mask built on the cards (LLAMA_QSA_POOL_MASK_DEV=1) on
top of C100's k-pool tables on the cards (LLAMA_QSA_KPOOL_DEV=1) on a P106, from one deep-lane run (bench/deep_lane)
with the loads warm (the control), kpool (C100's switch alone), mask (both switches), check (both switches and both
checks) and hostmask (both switches under the host profile). Standard library only; r74_tables.py's method.

    python3 -I r78_tables.py OUT [--out RES] [--ref LABEL=FILE:LOAD ...] > r78.md

OUT is the deep lane's results folder (requests.csv, loads.tsv, fit_<load>.txt, gpu.csv, server-<load>.log).
- The step: per load, depth and pass, ms a step = gen_ms / (gen_n - draft_accepted) (deep_table.py's step), with the
  steps, the draft's accepted of drafted tokens and tok/s.
- The verdict: at each depth and pass, the switch's arm (mask, hostmask: both switches, unprofiled and under the
  host profile) against the base arm (kpool: C100's switch alone), each arm's mean over its loads; the spread is the
  largest difference between two loads of one arm (the switch's pair holds the host profile's own cost, so it is
  conservative); gain when the switch's mean is below the base's by more than the spread, loss when above by more than
  it, else negligible. Twice: with this run's loads only (the base arm one load), and with --ref loads added to the
  base arm (the same build, rig and settings in other runs: each a steps.tsv of r74_tables.py's form, or a deep-lane
  requests.csv, and the load's name in it). The check load synchronizes and reads the cards back every micro-batch,
  so its step is not a speed of the switch.
- The replies: per depth and pass the token ids' md5 of every load, and which loads share it.
- The k-pool and pool-mask lines of each server log: the tables' allocation, the mask's switch, both checks' counts,
  the counts each memory logs when freed, any line that reports a difference, an abort or a switch turned off.
- The memory: per load the cards' used and free MiB after loading and the host's MemAvailable then (fit_<load>.txt),
  and per load and card the most MiB used in gpu.csv (sampled every 10 s) and so the least free.
Writes to --out: steps.tsv, verdict.tsv, replies.tsv, kpool_lines.tsv, cards.tsv."""
import argparse
import csv
import datetime
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--out', default='')
ap.add_argument('--base', default='kpool', help='the base arm\'s loads in this run')
ap.add_argument('--on', default='mask,hostmask', help='the switch arm\'s loads, unprofiled first')
ap.add_argument('--ref', action='append', default=[], help='LABEL=FILE:LOAD, a base-arm load of another run')
A = ap.parse_args()
BASE = A.base.split(',')
ON = A.on.split(',')


def rd(path):
    with open(path, newline='') as f:
        tsv = path.endswith('.tsv')
        return list(csv.DictReader(f, delimiter='\t' if tsv else ',',
                                   quoting=csv.QUOTE_NONE if tsv else csv.QUOTE_MINIMAL))


def wtsv(name, header, rows):
    if not A.out:
        return
    os.makedirs(A.out, exist_ok=True)
    with open(os.path.join(A.out, name), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def steps_of(rows):
    """{(load, depth, pass): ms a step} from a deep-lane requests.csv or a steps.tsv"""
    out = {}
    for r in rows:
        if 'ms_step' in r:
            out[(r['load'], int(r['depth']), r['pass'])] = float(r['ms_step'])
        elif r.get('kind') == 'decode':
            n = int(r['gen_n']) - int(r['draft_accepted'] or 0)
            out[(r['load'], int(r['depth']), r['pass'])] = float(r['gen_ms']) / n if n > 0 else float('nan')
    return out


reqs = rd(os.path.join(A.src, 'requests.csv'))
loads = rd(os.path.join(A.src, 'loads.tsv'))
order = [ld['load'] for ld in loads]
P = print

# --- the step per load, depth and pass
step = steps_of(reqs)
rows = []
for r in reqs:
    if r['kind'] != 'decode':
        continue
    n = int(r['gen_n']) - int(r['draft_accepted'] or 0)
    tps = 1000.0 * int(r['gen_n']) / float(r['gen_ms'])
    rows.append([r['load'], r['arm'], r['depth'], r['pass'], r['gen_n'], r['draft_accepted'], r['draft_n'], n,
                 r['gen_ms'], '%.2f' % step[(r['load'], int(r['depth']), r['pass'])], '%.2f' % tps, r['ids_md5'],
                 r['text_md5']])
wtsv('steps.tsv', ['load', 'arm', 'depth', 'pass', 'gen_n', 'draft_accepted', 'draft_n', 'steps', 'gen_ms', 'ms_step',
                   'tok_s', 'ids_md5', 'text_md5'], rows)
depths = sorted({int(r['depth']) for r in reqs if r['kind'] == 'decode'})

P('## The step, ms (gen_ms over the steps: generated tokens less accepted draft tokens)\n')
P('| Depth | Pass | ' + ' | '.join(order) + ' |')
P('|---:|---:|' + '---:|' * len(order))
for d in depths:
    for p in ('1', '2'):
        cells = []
        for ld in order:
            r = next((x for x in reqs if x['load'] == ld and x['kind'] == 'decode' and int(x['depth']) == d
                      and x['pass'] == p), None)
            cells.append('%.2f (%d steps, %s of %s)' % (step[(ld, d, p)], int(r['gen_n']) - int(r['draft_accepted']),
                                                         r['draft_accepted'], r['draft_n']) if r else '')
        P('| %s | %s | %s |' % (format(d, ','), p, ' | '.join(cells)))
P()

# --- the base arm's loads of other runs
refs = []
for spec in A.ref:
    label, rest = spec.split('=', 1)
    path, load = rest.rsplit(':', 1)
    st = steps_of(rd(path))
    refs.append((label, {(d, p): v for (ld, d, p), v in st.items() if ld == load}))


# --- the verdict
def verdict(name, with_refs):
    P('## The switch against the base, ms a step (%s)\n' % name)
    base_cols = BASE + ([lb for lb, _ in refs] if with_refs else [])
    P('Base arm: %s. Switch arm: %s. Spread: the largest difference between two loads of one arm. Class: gain when '
      'the switch\'s mean is below the base\'s by more than the spread, loss when above by more than it, else '
      'negligible.\n' % (', '.join(base_cols), ', '.join(ON)))
    P('| Depth | Pass | warm | ' + ' | '.join(base_cols) + ' | ' + ' | '.join(ON) +
      ' | %s − %s | base mean | switch mean | Δ | spread | class |' % (ON[0], BASE[0]))
    P('|---:|---:|' + '---:|' * (1 + len(base_cols) + len(ON) + 5) + '---|')
    out = []
    for d in depths:
        for p in ('1', '2'):
            try:
                b = [step[(x, d, p)] for x in BASE]
                s = [step[(x, d, p)] for x in ON]
            except KeyError:
                continue
            if with_refs:
                b += [rf[(d, p)] for _, rf in refs]
            w = step.get(('warm', d, p), float('nan'))
            mb, ms = sum(b) / len(b), sum(s) / len(s)
            sp = max(max(b) - min(b), max(s) - min(s))
            delta = ms - mb
            cls = 'gain' if delta < -sp else 'loss' if delta > sp else 'negligible'
            P('| %s | %s | %.2f | %s | %s | %+.2f | %.2f | %.2f | %+.2f | %.2f | %s |' % (
                format(d, ','), p, w, ' | '.join('%.2f' % x for x in b), ' | '.join('%.2f' % x for x in s),
                s[0] - b[0], mb, ms, delta, sp, cls))
            out.append([name, d, p, '%.2f' % w, ';'.join('%s=%.2f' % (c, x) for c, x in zip(base_cols, b)),
                        ';'.join('%s=%.2f' % (c, x) for c, x in zip(ON, s)), '%.2f' % (s[0] - b[0]), '%.2f' % mb,
                        '%.2f' % ms, '%.2f' % delta, '%.2f' % sp, cls])
    P()
    return out


vrows = verdict('this run', False)
if refs:
    vrows += verdict('with the base arm\'s loads of other runs', True)
wtsv('verdict.tsv', ['set', 'depth', 'pass', 'warm', 'base_loads', 'switch_loads', ON[0] + '_minus_' + BASE[0],
                     'base_mean', 'switch_mean', 'delta', 'spread', 'class'], vrows)

# --- the replies
P('## The replies (token ids\' md5; loads that share it)\n')
P('| Depth | Pass | groups of loads with one reply |')
P('|---:|---:|---|')
rrows = []
for d in depths:
    for p in ('1', '2'):
        g = {}
        for ld in order:
            r = next((x for x in reqs if x['load'] == ld and x['kind'] == 'decode' and int(x['depth']) == d
                      and x['pass'] == p), None)
            if r:
                g.setdefault(r['ids_md5'], []).append(ld)
                rrows.append([d, p, ld, r['ids_md5'], r['text_md5']])
        P('| %s | %s | %s |' % (format(d, ','), p, '; '.join('%s `%s`' % (', '.join(v), k[:8]) for k, v in g.items())))
wtsv('replies.tsv', ['depth', 'pass', 'load', 'ids_md5', 'text_md5'], rrows)
P()

# --- the k-pool and pool-mask lines
PAT = re.compile(r'k-pool tables kept on|QSA pool mask built on the devices: |QSA pool mask built on the devices \(|'
                 r'masks built on the devices identical|copies identical to the tables|k-pool tables on \d+ devices|'
                 r'differs|does not hold|LLAMA_QSA_KPOOL_DEV needs|LLAMA_QSA_POOL_MASK_DEV needs|'
                 r'pipelined decoding: the k-pool|GGML_ABORT|GGML_ASSERT|failed to allocate the k-pool|'
                 r'QSA k-pool inputs, last')
P('## The k-pool and pool-mask lines per server log\n')
krows = []
for ld in order:
    f = os.path.join(A.src, 'server-%s.log' % ld)
    if not os.path.exists(f):
        continue
    counts = {}
    first = {}
    last = {}
    with open(f, errors='replace') as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            k = m.group(0)
            if k.startswith('QSA k-pool inputs'):
                k = 'QSA k-pool inputs (LLAMA_QSA_INPUT_STATS)'
            counts[k] = counts.get(k, 0) + 1
            first.setdefault(k, line.strip())
            last[k] = line.strip()
    P('- `%s`: %s' % (ld, '; '.join('%s × %d' % (k, v) for k, v in counts.items()) or 'none'))
    for k, v in counts.items():
        krows.append([ld, k, v, first[k][:400], last[k][:400]])
wtsv('kpool_lines.tsv', ['load', 'line', 'count', 'first', 'last'], krows)
P()

# --- the memory
P('## The cards and the host\n')
crows = []
fit = {}
for ld in order:
    f = os.path.join(A.src, 'fit_%s.txt' % ld)
    if not os.path.exists(f):
        continue
    used, free, avail = {}, {}, ''
    for line in open(f):
        m = re.match(r'(\d+), (\d+), (\d+)$', line.strip())
        if m:
            used[int(m.group(1))] = int(m.group(2))
            free[int(m.group(1))] = int(m.group(3)) - int(m.group(2))
        if line.startswith('MemAvailable'):
            avail = str(int(line.split()[1]) // 1024)
    fit[ld] = (used, free, avail)
starts = []
for ld in loads:
    starts.append((datetime.datetime.fromisoformat(ld['started']).replace(tzinfo=None), ld['load']))
peak = {}
gf = os.path.join(A.src, 'gpu.csv')
if os.path.exists(gf):
    for line in open(gf, errors='replace'):
        p = [x.strip() for x in line.split(',')]
        if len(p) < 6 or not p[1].isdigit():
            continue
        try:
            t = datetime.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f')
            mib = int(p[5].split()[0])
        except ValueError:
            continue
        cur = None
        for s, ld in starts:
            if t >= s:
                cur = ld
        if cur is None:
            continue
        k = (cur, int(p[1]))
        peak[k] = max(peak.get(k, 0), mib)
cards = sorted({c for (_, c) in peak} | {c for v in fit.values() for c in v[0]})
P('Used MiB after loading (fit), the most used over the load (gpu.csv, every 10 s); free = 6,144 − used.\n')
P('| Load | MemAvailable after loading, MiB | ' + ' | '.join('card %d' % c for c in cards) + ' |')
P('|---|---:|' + '---|' * len(cards))
for ld in order:
    u, fr, av = fit.get(ld, ({}, {}, ''))
    cells = []
    for c in cards:
        pk = peak.get((ld, c))
        cells.append('%s / %s' % (u.get(c, ''), pk if pk is not None else ''))
        crows.append([ld, c, u.get(c, ''), fr.get(c, ''), pk if pk is not None else '',
                      6144 - pk if pk is not None else ''])
    P('| %s | %s | %s |' % (ld, av, ' | '.join(cells)))
wtsv('cards.tsv', ['load', 'card', 'used_after_load_mib', 'free_after_load_mib', 'most_used_mib', 'least_free_mib'],
     crows)
