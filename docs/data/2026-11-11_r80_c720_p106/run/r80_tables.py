#!/usr/bin/env python3
"""r80_tables.py: the tables of R80 (Q1293), C720's small k-pool inputs in C100's update (LLAMA_QSA_KPOOL_STAGE=1 with
LLAMA_QSA_KPOOL_DEV=1) on a P106, from one deep-lane run (bench/deep_lane) with the loads warm (the preset's control),
dev (C100's switch), stage (C100's and C720's), check (stage and C100's check), hostdev and hoststage (dev and stage
under the host profile). R74's r74_tables.py with these loads as its defaults and the stage switch's line among the
k-pool lines. Standard library only.

    python3 -I r80_tables.py OUT [--out RES] > r80.md

OUT is the deep lane's results folder (requests.csv, loads.tsv, fit_<load>.txt, gpu.csv, server-<load>.log).
- The step: per load, depth and pass, ms a step = gen_ms / (gen_n - draft_accepted) (deep_table.py's step), with the
  steps, the draft's accepted of drafted tokens and tok/s.
- The verdict: at each depth and pass, stage against dev (the row's comparison), hoststage against hostdev (the same
  pair under the host profile), and both pairs together: the switch's two loads (stage, hoststage) against the
  control's two (dev, hostdev), each arm's mean over its two loads, the spread the larger of the two arms' differences
  between their loads (it holds the host profile's own cost, so it is conservative); gain when the switch's mean is
  below the control's by more than the spread, loss when above by more than it, else negligible. The check load
  synchronizes and reads the cards back every micro-batch, so its step is not a speed of the switch.
- The replies: per depth and pass the token ids' md5 of every load, and which loads share it.
- The k-pool lines of each server log: the tables' allocation, the stage switch's line, the check's counts, the counts
  each memory logs when freed, any line that reports a difference, an abort or the switch turned off.
- The memory: per load the cards' used and free MiB after loading and the host's MemAvailable then (fit_<load>.txt),
  and per load and card the most MiB used in gpu.csv (sampled every 10 s) and so the least free.
Writes to --out: steps.tsv, verdict.tsv, replies.tsv, kpool_lines.tsv, cards.tsv."""
import argparse
import csv
import datetime
import os
import re
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--out', default='')
ap.add_argument('--off', default='dev,hostdev', help='the control loads, unprofiled first')
ap.add_argument('--on', default='stage,hoststage', help='the switch loads, in the same order')
A = ap.parse_args()
OFF = A.off.split(',')
ON = A.on.split(',')


def rd(name):
    with open(os.path.join(A.src, name), newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if name.endswith('.tsv') else ',', quoting=csv.QUOTE_NONE
                                   if name.endswith('.tsv') else csv.QUOTE_MINIMAL))


def wtsv(name, header, rows):
    if not A.out:
        return
    os.makedirs(A.out, exist_ok=True)
    with open(os.path.join(A.out, name), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


reqs = rd('requests.csv')
loads = rd('loads.tsv')
order = [ld['load'] for ld in loads]
P = print

# --- the step per load, depth and pass
step = {}
rows = []
for r in reqs:
    if r['kind'] != 'decode':
        continue
    n = int(r['gen_n']) - int(r['draft_accepted'] or 0)
    ms = float(r['gen_ms']) / n if n > 0 else float('nan')
    tps = 1000.0 * int(r['gen_n']) / float(r['gen_ms'])
    key = (r['load'], int(r['depth']), r['pass'])
    step[key] = ms
    rows.append([r['load'], r['arm'], r['depth'], r['pass'], r['gen_n'], r['draft_accepted'], r['draft_n'], n,
                 r['gen_ms'], '%.2f' % ms, '%.2f' % tps, r['ids_md5'], r['text_md5']])
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

# --- the verdict
P('## The switch against the control, ms a step\n')
P('Spread: the larger of |%s − %s| and |%s − %s|. Class: gain when the switch\'s mean is below the control\'s by more '
  'than the spread, loss when above by more than it, else negligible.\n' % (OFF[0], OFF[1], ON[0], ON[1]))
P('| Depth | Pass | %s | %s | %s − %s | %s | %s | %s − %s | control mean | switch mean | Δ | spread | class |' % (
    OFF[0], ON[0], ON[0], OFF[0], OFF[1], ON[1], ON[1], OFF[1]))
P('|---:|---:|' + '---:|' * 10 + '---|')
vrows = []
for d in depths:
    for p in ('1', '2'):
        try:
            a0, b0, a1, b1 = (step[(x, d, p)] for x in (OFF[0], ON[0], OFF[1], ON[1]))
        except KeyError:
            continue
        mo, mn = (a0 + a1) / 2, (b0 + b1) / 2
        sp = max(abs(a0 - a1), abs(b0 - b1))
        delta = mn - mo
        cls = 'gain' if delta < -sp else 'loss' if delta > sp else 'negligible'
        P('| %s | %s | %.2f | %.2f | %+.2f | %.2f | %.2f | %+.2f | %.2f | %.2f | %+.2f | %.2f | %s |' % (
            format(d, ','), p, a0, b0, b0 - a0, a1, b1, b1 - a1, mo, mn, delta, sp, cls))
        vrows.append([d, p, '%.2f' % a0, '%.2f' % b0, '%.2f' % (b0 - a0), '%.2f' % a1, '%.2f' % b1, '%.2f' % (b1 - a1),
                      '%.2f' % mo, '%.2f' % mn, '%.2f' % delta, '%.2f' % sp, cls])
wtsv('verdict.tsv', ['depth', 'pass', OFF[0], ON[0], ON[0] + '_minus_' + OFF[0], OFF[1], ON[1],
                     ON[1] + '_minus_' + OFF[1], 'control_mean', 'switch_mean', 'delta', 'spread', 'class'], vrows)
P()

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

# --- the k-pool lines
PAT = re.compile(r'k-pool tables kept on|go up in the k-pool tables|ubatches checked|k-pool tables on \d+ devices|differs|does not hold|'
                 r'LLAMA_QSA_KPOOL_DEV needs|pipelined decoding: the k-pool|GGML_ABORT|GGML_ASSERT|failed to allocate the k-pool|'
                 r'QSA k-pool inputs, last')
P('## The k-pool lines per server log\n')
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
total = {}
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
