#!/usr/bin/env python3
"""first.py: R174's table: each load's first request at a new depth against its repeat, prompt and generation time.

    python3 -I first.py OUT [--control c] [--arm off] [--tsv first.tsv] > first.md

OUT is a deep lane's results folder (bench/deep_lane). From OUT/requests.csv and OUT/loads.tsv, per depth and timed
load: the greedy `decode` request's pass 1 (the first request at that depth) and pass 2 (the same request again), each
with its prompt time (prompt_ms, the tokens it read: 64 on pass 1, 4 on pass 2), its generation time (gen_ms) and its
steps (generated tokens minus accepted draft tokens); the first request's excess, pass 1 minus pass 2, for generation,
for prompt time and for both; the step (generation time over steps) of each pass and its excess (pass 1's step minus pass 2's, since the two passes
can take different numbers of steps); and the `prime` request's prompt time
(the chunk read on top of the previous depth's cache, cut into calls of -b tokens). Each figure is classed for the arm
against the control: the spread is the larger of the two arms' differences between their two loads; gain when the
arm's mean is below the control's by more than the spread, loss when above by more than it, else negligible. "all"
adds every depth. Last, the greedy replies: the token ids' md5 of every decode pass, per depth, and whether every
timed load of both arms wrote the same. Standard library only.
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out')
ap.add_argument('--control', default='c')
ap.add_argument('--arm', default='off')
ap.add_argument('--tsv', default='')
A = ap.parse_args()

timed = None
lt = os.path.join(A.out, 'loads.tsv')
if os.path.exists(lt):
    with open(lt, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if rows and 'timed' in rows[0]:
        timed = {r['load'] for r in rows if r.get('timed') in ('1', 'yes', 'true')}
with open(os.path.join(A.out, 'requests.csv'), newline='') as f:
    reqs = [r for r in csv.DictReader(f) if r['load'] != 'warm' and (timed is None or r['load'] in timed)]
arms = [A.control, A.arm]
loads = {a: [] for a in arms}
for r in reqs:
    if r['arm'] in loads and r['load'] not in loads[r['arm']]:
        loads[r['arm']].append(r['load'])
depths = sorted({int(r['depth']) for r in reqs})


def get(load, kind, pas, depth):
    for r in reqs:
        if r['load'] == load and r['kind'] == kind and r['pass'] == pas and int(r['depth']) == depth:
            return r
    return None


def fig(load, depth):
    p1, p2, pr = get(load, 'decode', '1', depth), get(load, 'decode', '2', depth), get(load, 'prime', '0', depth)
    if not p1 or not p2:
        return None
    s1 = int(p1['gen_n']) - int(p1['draft_accepted'] or 0)
    s2 = int(p2['gen_n']) - int(p2['draft_accepted'] or 0)
    d = {
        'p1_prompt': float(p1['prompt_ms']), 'p2_prompt': float(p2['prompt_ms']),
        'p1_gen': float(p1['gen_ms']), 'p2_gen': float(p2['gen_ms']), 'p1_steps': s1, 'p2_steps': s2,
        'p1_n': int(p1['prompt_n']), 'p2_n': int(p2['prompt_n']),
        'prime': float(pr['prompt_ms']) if pr else None, 'prime_n': int(pr['prompt_n']) if pr else None,
        'ids1': p1['ids_md5'] or p1['text_md5'], 'ids2': p2['ids_md5'] or p2['text_md5'],
    }
    d['x_gen'] = d['p1_gen'] - d['p2_gen']
    d['x_prompt'] = d['p1_prompt'] - d['p2_prompt']
    d['x_both'] = d['x_gen'] + d['x_prompt']
    d['p1_step'] = d['p1_gen'] / s1
    d['p2_step'] = d['p2_gen'] / s2
    d['x_step'] = d['p1_step'] - d['p2_step']
    return d


F = {(l, dp): fig(l, dp) for a in arms for l in loads[a] for dp in depths}
for a in arms:
    for l in loads[a]:
        fs = [F[(l, dp)] for dp in depths]
        if all(fs):
            al = {k: sum(f[k] for f in fs) for k in ('p1_prompt', 'p2_prompt', 'p1_gen', 'p2_gen', 'x_gen', 'x_prompt', 'x_both', 'p1_steps', 'p2_steps')}
            al['p1_step'] = al['p1_gen'] / al['p1_steps']
            al['p2_step'] = al['p2_gen'] / al['p2_steps']
            al['x_step'] = al['p1_step'] - al['p2_step']
            pr = [f['prime'] for f in fs if f['prime'] is not None]
            al['prime'] = sum(pr) if pr else None
            F[(l, 'all')] = al
        else:
            F[(l, 'all')] = None

METRICS = [
    ('x_gen', 'first request\'s excess in generation (pass 1 minus pass 2), ms'),
    ('x_prompt', 'first request\'s excess in prompt time (pass 1 minus pass 2), ms'),
    ('x_both', 'first request\'s excess, generation and prompt together, ms'),
    ('x_step', 'first request\'s excess a step (pass 1 step minus pass 2 step), ms'),
    ('p1_gen', 'pass 1 generation, ms'),
    ('p2_gen', 'pass 2 generation, ms'),
    ('p1_prompt', 'pass 1 prompt time, ms'),
    ('p2_prompt', 'pass 2 prompt time, ms'),
    ('p1_step', 'pass 1 step, ms'),
    ('p2_step', 'pass 2 step, ms'),
    ('prime', 'prime\'s prompt time (the chunk on top of the previous depth\'s cache), ms'),
]
tsv = [['metric', 'depth', 'arm', 'control', 'arm_loads', 'control_loads', 'spread', 'delta', 'class']]


def klass(m, dep):
    cv = [F[(l, dep)][m] for l in loads[A.control] if F[(l, dep)] and F[(l, dep)][m] is not None]
    av = [F[(l, dep)][m] for l in loads[A.arm] if F[(l, dep)] and F[(l, dep)][m] is not None]
    if len(cv) < 2 or len(av) < 2:
        return cv, av, None, None, ''
    spread = max(max(cv) - min(cv), max(av) - min(av))
    delta = sum(av) / len(av) - sum(cv) / len(cv)
    c = 'gain' if delta < -spread else 'loss' if delta > spread else 'negligible'
    return cv, av, spread, delta, c


out = []
out.append('## R174: the first request against its repeat, arms `%s` (control) and `%s`\n' % (A.control, A.arm))
out.append('Timed loads: ' + ', '.join(l for a in arms for l in loads[a]) + '.\n')
out.append('### Per load and depth, ms\n')
out.append('| Depth | Load | Pass 1 prompt (tokens) | Pass 2 prompt (tokens) | Pass 1 generation (steps) | Pass 2 generation (steps) | Excess: generation | Excess: prompt | Excess: both | Step pass 1 | Step pass 2 | Prime prompt (tokens) |')
out.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for dp in depths:
    for a in arms:
        for l in loads[a]:
            f = F[(l, dp)]
            if not f:
                out.append('| %d | %s | missing |||||||||| |' % (dp, l))
                continue
            out.append('| %s | %s | %.3f (%d) | %.3f (%d) | %.3f (%d) | %.3f (%d) | %.3f | %.3f | %.3f | %.2f | %.2f | %s |' % (
                format(dp, ','), l, f['p1_prompt'], f['p1_n'], f['p2_prompt'], f['p2_n'], f['p1_gen'], f['p1_steps'],
                f['p2_gen'], f['p2_steps'], f['x_gen'], f['x_prompt'], f['x_both'], f['p1_step'], f['p2_step'],
                '%.3f (%d)' % (f['prime'], f['prime_n']) if f['prime'] is not None else '—'))
out.append('')
out.append('### Classes of `%s` against `%s` (time: lower is a gain)\n' % (A.arm, A.control))
out.append('| Measure | Depth | `%s` loads | `%s` loads | Spread | `%s` minus `%s` | Class |' % (A.control, A.arm, A.arm, A.control))
out.append('|---|---:|---|---|---:|---:|---|')
for m, title in METRICS:
    for dp in depths + ['all']:
        cv, av, spread, delta, c = klass(m, dp)
        if not cv and not av:
            continue
        ds = 'all' if dp == 'all' else format(dp, ',')
        out.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            title, ds, ', '.join('%.3f' % v for v in cv), ', '.join('%.3f' % v for v in av),
            '%.3f' % spread if spread is not None else '—', '%+.3f' % delta if delta is not None else '—', c or '—'))
        tsv.append([m, str(dp), A.arm, A.control, ','.join('%.3f' % v for v in av), ','.join('%.3f' % v for v in cv),
                    '%.3f' % spread if spread is not None else '', '%.3f' % delta if delta is not None else '', c])
out.append('')
out.append('### The greedy replies (token ids\' md5 of each decode pass)\n')
out.append('| Depth | ' + ' | '.join('%s p1 | %s p2' % (l, l) for a in arms for l in loads[a]) + ' | All the same |')
out.append('|---:|' + '---|---|' * sum(len(loads[a]) for a in arms) + '---|')
same_all = True
for dp in depths:
    ids = []
    for a in arms:
        for l in loads[a]:
            f = F[(l, dp)]
            ids += [f['ids1'][:8], f['ids2'][:8]] if f else ['—', '—']
    s = len(set(ids)) == 1
    same_all &= s
    out.append('| %s | %s | %s |' % (format(dp, ','), ' | '.join(ids), 'yes' if s else 'no'))
out.append('')
out.append('Every greedy reply the same on both passes of every timed load of both arms: %s.' % ('yes' if same_all else 'no'))
print('\n'.join(out))
if A.tsv:
    with open(A.tsv, 'w', newline='') as f:
        csv.writer(f, delimiter='\t', lineterminator='\n').writerows(tsv)
