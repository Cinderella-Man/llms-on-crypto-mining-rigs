#!/usr/bin/env python3
"""r91_tables.py: the tables of R91 (Q1323), C748's KV-only draft pass (LLAMA_MTP_KV_ONLY=1) with C100's k-pool tables
on the cards (LLAMA_QSA_KPOOL_DEV=1, with C720's LLAMA_QSA_KPOOL_STAGE=1) on a P106, from one deep-lane run
(bench/deep_lane) with the loads warm (the preset's control, the warm-up), two loads each of kv (C620's switch), dev
(C100's and C720's) and kvdev (all three), and check (kvdev with C100's check). R80's r80_tables.py with arms in
place of fixed loads, the prompt reading classed too, and the k-pool tables' counts at exit. Standard library only.

    python3 -I r91_tables.py OUT [--out RES] [--pairs kvdev:kv,kvdev:dev] > r91.md

OUT is the deep lane's results folder (requests.csv, loads.tsv, fit_<load>.txt, gpu.csv, server-<load>.log).
- The step: per load, depth and pass, ms a step = gen_ms / (gen_n - draft_accepted) (deep_table.py's step), with the
  steps, the draft's accepted of drafted tokens and tok/s.
- The verdict, per pair switch:control (arms), at each depth and pass: every load of each arm by value, each arm's
  mean, the spread (the larger of the two arms' differences between their fastest and slowest load), the difference
  of the means and the class: for the step a gain when the switch's mean is below the control's by more than the
  spread, a loss when above by more than it, else negligible. The same for prompt reading (each depth's prime: the
  tokens it read over its time, tok/s; more is the gain).
- The replies: per depth and pass the token ids' md5 of every load, and which loads share it.
- The k-pool lines of each server log: the tables' allocation, the stage switch's line, the switches logged at load,
  the check's counts, the counts each memory logs when freed (micro-batches, full rewrites, rows sent, checked), any
  line that reports a difference, an abort or the switch turned off; and per load the counts as numbers, with the
  decode steps and requests of the load beside them.
- The memory: per load the cards' used and free MiB after loading and the host's MemAvailable then (fit_<load>.txt),
  and per load and card the most MiB used in gpu.csv (sampled every 10 s) and so the least free.
Writes to --out: steps.tsv, verdict.tsv, prompt.tsv, replies.tsv, kpool_lines.tsv, kpool_counts.tsv, cards.tsv."""
import argparse
import csv
import datetime
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--out', default='')
ap.add_argument('--pairs', default='kvdev:kv,kvdev:dev', help='switch_arm:control_arm, comma separated')
A = ap.parse_args()
PAIRS = [tuple(p.split(':')) for p in A.pairs.split(',')]


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


def fmt(d):
    return format(d, ',')


reqs = rd('requests.csv')
loads = rd('loads.tsv')
order = [ld['load'] for ld in loads]
arm_of = {ld['load']: ld['arm'] for ld in loads}
timed = {ld['load']: ld['timed'] == '1' for ld in loads}
loads_of = {}
for ld in order:
    if timed[ld]:
        loads_of.setdefault(arm_of[ld], []).append(ld)
P = print


def req(ld, kind, d, p=None):
    return next((x for x in reqs if x['load'] == ld and x['kind'] == kind and int(x['depth']) == d
                 and (p is None or x['pass'] == p)), None)


# --- the step per load, depth and pass
step = {}
rows = []
for r in reqs:
    if r['kind'] != 'decode':
        continue
    n = int(r['gen_n']) - int(r['draft_accepted'] or 0)
    ms = float(r['gen_ms']) / n if n > 0 else float('nan')
    tps = 1000.0 * int(r['gen_n']) / float(r['gen_ms'])
    step[(r['load'], int(r['depth']), r['pass'])] = ms
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
            r = req(ld, 'decode', d, p)
            cells.append('%.2f (%d steps, %s of %s, %.2f tok/s)' % (
                step[(ld, d, p)], int(r['gen_n']) - int(r['draft_accepted']), r['draft_accepted'], r['draft_n'],
                1000.0 * int(r['gen_n']) / float(r['gen_ms'])) if r else '')
        P('| %s | %s | %s |' % (fmt(d), p, ' | '.join(cells)))
P()

# --- prompt reading per load and depth (the prime)
prompt = {}
prows = []
for r in reqs:
    if r['kind'] != 'prime' or not float(r['prompt_ms'] or 0):
        continue
    tps = 1000.0 * int(r['prompt_n']) / float(r['prompt_ms'])
    prompt[(r['load'], int(r['depth']))] = tps
    prows.append([r['load'], r['arm'], r['depth'], r['prompt_tokens'], r['cache_n'], r['prompt_n'], r['prompt_ms'],
                  '%.2f' % tps])
pdepths = sorted({d for (_, d) in prompt})
P('## Prompt reading, tok/s (each depth\'s prime: tokens read over their time; read on cached)\n')
P('| Depth | ' + ' | '.join(order) + ' |')
P('|---:|' + '---:|' * len(order))
for d in pdepths:
    cells = []
    for ld in order:
        r = req(ld, 'prime', d)
        cells.append('%.2f (%s on %s)' % (prompt[(ld, d)], r['prompt_n'], r['cache_n']) if (ld, d) in prompt else '')
    P('| %s | %s |' % (fmt(d), ' | '.join(cells)))
P()


def classify(on_vals, off_vals, less_is_gain):
    mo, mn = sum(off_vals) / len(off_vals), sum(on_vals) / len(on_vals)
    sp = max(max(off_vals) - min(off_vals), max(on_vals) - min(on_vals))
    delta = mn - mo
    better = -delta if less_is_gain else delta
    cls = 'gain' if better > sp else 'loss' if better < -sp else 'negligible'
    if len(on_vals) < 2 or len(off_vals) < 2:
        cls = 'no class (one load)'
    return mo, mn, delta, sp, cls


# --- the verdicts
vrows = []
for on, off in PAIRS:
    lon, loff = loads_of.get(on, []), loads_of.get(off, [])
    if not lon or not loff:
        continue
    P('## %s against %s: the step, ms\n' % (on, off))
    P('Spread: the larger of the two arms\' differences between their loads. Class: gain when %s\'s mean is below '
      '%s\'s by more than the spread, loss when above by more than it, else negligible.\n' % (on, off))
    P('| Depth | Pass | %s | %s | %s mean | %s mean | Δ | Δ %% | spread | class |' % (
        ', '.join(loff), ', '.join(lon), off, on))
    P('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for d in depths:
        for p in ('1', '2'):
            try:
                a = [step[(x, d, p)] for x in loff]
                b = [step[(x, d, p)] for x in lon]
            except KeyError:
                continue
            mo, mn, delta, sp, cls = classify(b, a, True)
            P('| %s | %s | %s | %s | %.2f | %.2f | %+.2f | %+.2f %% | %.2f | %s |' % (
                fmt(d), p, ', '.join('%.2f' % x for x in a), ', '.join('%.2f' % x for x in b), mo, mn, delta,
                100.0 * delta / mo, sp, cls))
            vrows.append(['step_ms', on, off, d, p, ' '.join('%.2f' % x for x in a), ' '.join('%.2f' % x for x in b),
                          '%.2f' % mo, '%.2f' % mn, '%.2f' % delta, '%.2f' % (100.0 * delta / mo), '%.2f' % sp, cls])
    P()
    P('## %s against %s: prompt reading, tok/s\n' % (on, off))
    P('| Depth | %s | %s | %s mean | %s mean | Δ | Δ %% | spread | class |' % (', '.join(loff), ', '.join(lon), off, on))
    P('|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for d in pdepths:
        try:
            a = [prompt[(x, d)] for x in loff]
            b = [prompt[(x, d)] for x in lon]
        except KeyError:
            continue
        mo, mn, delta, sp, cls = classify(b, a, False)
        P('| %s | %s | %s | %.2f | %.2f | %+.2f | %+.2f %% | %.2f | %s |' % (
            fmt(d), ', '.join('%.2f' % x for x in a), ', '.join('%.2f' % x for x in b), mo, mn, delta,
            100.0 * delta / mo, sp, cls))
        vrows.append(['prompt_tok_s', on, off, d, '', ' '.join('%.2f' % x for x in a), ' '.join('%.2f' % x for x in b),
                      '%.2f' % mo, '%.2f' % mn, '%.2f' % delta, '%.2f' % (100.0 * delta / mo), '%.2f' % sp, cls])
    P()
wtsv('verdict.tsv', ['measure', 'switch', 'control', 'depth', 'pass', 'control_loads', 'switch_loads', 'control_mean',
                     'switch_mean', 'delta', 'delta_pct', 'spread', 'class'], vrows)
wtsv('prompt.tsv', ['load', 'arm', 'depth', 'prompt_tokens', 'cache_n', 'prompt_n', 'prompt_ms', 'tok_s'], prows)

# --- the replies
P('## The replies (token ids\' md5; loads that share it)\n')
P('| Depth | Pass | groups of loads with one reply |')
P('|---:|---:|---|')
rrows = []
for d in depths:
    for p in ('1', '2'):
        g = {}
        for ld in order:
            r = req(ld, 'decode', d, p)
            if r:
                g.setdefault(r['ids_md5'], []).append(ld)
                rrows.append([d, p, ld, r['ids_md5'], r['text_md5']])
        P('| %s | %s | %s |' % (fmt(d), p, '; '.join('%s `%s`' % (', '.join(v), k[:8]) for k, v in g.items())))
wtsv('replies.tsv', ['depth', 'pass', 'load', 'ids_md5', 'text_md5'], rrows)
P()

# --- the k-pool lines
PAT = re.compile(r'k-pool tables kept on|go up in the k-pool tables|ubatches checked|k-pool tables on \d+ devices|'
                 r'differs|does not hold|LLAMA_QSA_KPOOL_DEV needs|pipelined decoding: the k-pool|GGML_ABORT|'
                 r'GGML_ASSERT|failed to allocate the k-pool|QSA k-pool inputs, last|LLAMA_MTP_KV_ONLY: |switches: |'
                 r'not read by this build')
CNT = re.compile(r'k-pool tables on (\d+) devices: (\d+) ubatches, (\d+) full, (\d+) of (\d+) rows uploaded'
                 r'(?: \(([\d.]+) a ubatch\))?(?:, (\d+) checked)?')
P('## The k-pool lines per server log\n')
krows = []
crows_k = []
for ld in order:
    f = os.path.join(A.src, 'server-%s.log' % ld)
    if not os.path.exists(f):
        continue
    counts, first, last = {}, {}, {}
    with open(f, errors='replace') as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                k = m.group(0)
                if k.startswith('QSA k-pool inputs'):
                    k = 'QSA k-pool inputs (LLAMA_QSA_INPUT_STATS)'
                if k.startswith('k-pool tables on'):
                    k = 'k-pool tables on N devices (at free)'
                counts[k] = counts.get(k, 0) + 1
                first.setdefault(k, line.strip())
                last[k] = line.strip()
            c = CNT.search(line)
            if c:
                n_steps = sum(int(r['gen_n']) - int(r['draft_accepted'] or 0) for r in reqs
                              if r['load'] == ld and r['kind'] in ('decode', 'warmup', 'sample'))
                n_req = sum(1 for r in reqs if r['load'] == ld)
                crows_k.append([ld, arm_of[ld], c.group(1), c.group(2), c.group(3), c.group(4), c.group(5),
                                c.group(6) or '', c.group(7) or '', n_steps, n_req])
    P('- `%s`: %s' % (ld, '; '.join('%s × %d' % (k, v) for k, v in counts.items()) or 'none'))
    for k, v in counts.items():
        krows.append([ld, k, v, first[k][:400], last[k][:400]])
wtsv('kpool_lines.tsv', ['load', 'line', 'count', 'first', 'last'], krows)
wtsv('kpool_counts.tsv', ['load', 'arm', 'devices', 'ubatches', 'full', 'rows_uploaded', 'rows_total', 'rows_a_ubatch',
                          'checked', 'gen_steps_of_load', 'requests_of_load'], crows_k)
P()
P('## The k-pool tables at exit (each memory\'s line when it is freed)\n')
P('| Load | Devices | Micro-batches | Full rewrites | Rows sent of rows | Rows a micro-batch | Checked | The load\'s '
  'generation steps | Its requests |')
P('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in crows_k:
    P('| %s | %s | %s | %s | %s of %s | %s | %s | %s | %s |' % (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                                                               r[10]))
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
starts = [(datetime.datetime.fromisoformat(ld['started']).replace(tzinfo=None), ld['load']) for ld in loads]
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
