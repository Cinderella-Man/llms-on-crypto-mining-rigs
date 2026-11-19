#!/usr/bin/env python3
"""trace_table.py OUT [--arm-map a=b,...] [--out trace.md] [--tsv trace.tsv]: the draft's acceptance by position and
its time a step, per request, from a deep-lane run whose loads wrote LLAMA_STEP_TRACE files. Standard library only.

Reads OUT/requests-<load>.csv (deep_client.py's line per request, in the order sent) and OUT/st-<load>.jsonl (the
server's step trace: one line per server step, `n` its counts, `t` its host marks) for every load that has both.
A request's steps are cut from the trace in order: its prompt steps (steps without a draft whose batch tokens add up
to the request's prompt_n), then its generation steps (every step with a draft, and the reply's last one-token pass)
up to the next prompt step. Each request is checked: its steps' accepted tokens must add up to the server's
draft_accepted and its drafted tokens to draft_n.

Per generation request (decode passes and the sampled pass) and per load, then per arm, depth and pass:
- the trace's steps (the request's first token comes from its prompt, so the lane counts one step more), drafted and
  accepted tokens, tokens a step and ms a step as the lane counts them (a step per generated token less the accepted
  drafts: gen_n / (gen_n - accepted), gen_ms / (gen_n - accepted));
- the steps that drafted 3 tokens by how many they accepted (0, 1, 2, 3), and the steps that drafted fewer;
- acceptance by draft position j: P_j, the share of steps whose j-th drafted token was accepted, over the steps that
  drafted at least j tokens (the server's n_accepted_per_pos over its steps); and a_j, the share accepted at j of the
  steps that accepted every token before j (P_j / P_(j-1) where every step drafts j);
- the draft's span a step: the server's drafting stage, from `s.drf` to `s.drf.end` (the drafter's passes, the
  verify pass prepared inside the last one, the yield's end), median and mean over the request's steps that drafted;
  the drafter's own part `mtp.drf` to `mtp.drf.end`, median; the whole step from one step's first mark to the next
  step's first mark, median (the trace's own clock; the lane's ms a step is gen_ms over steps).
--arm-map renames arms (load own1 and own2 both as own, for example). FN115 (Q1545) is its first use.
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--arm-map', default='')
ap.add_argument('--out', default='')
ap.add_argument('--tsv', default='')
A = ap.parse_args()
AMAP = dict(x.split('=', 1) for x in A.arm_map.split(',') if '=' in x)
NMAX = 3


def mark(step, name):
    for m, t in step['t']:
        if m == name:
            return t
    return None


def last_mark(step, name):
    v = None
    for m, t in step['t']:
        if m == name:
            v = t
    return v


def read_trace(path):
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue  # a last line cut by the server's stop
            if 'step' in d:
                steps.append(d)
    return steps


def is_prompt(s):
    return 'drf' not in s['n'] and s['n'].get('tok', 0) > 1


def split(rows, steps, load):
    """[(row, gen_steps, problems)] in the rows' order."""
    out, i, n = [], 0, len(steps)
    for r in rows:
        need = int(r['prompt_n'] or 0)
        got = 0
        while i < n and got < need and is_prompt(steps[i]):
            got += steps[i]['n'].get('tok', 0)
            i += 1
        gen = []
        while i < n and not is_prompt(steps[i]):
            gen.append(steps[i])
            i += 1
        acc = sum(s['n'].get('acc', 0) for s in gen)
        drf = sum(s['n'].get('drf', 0) for s in gen)
        bad = []
        if got != need:
            bad.append(f'prompt steps {got} of {need} tokens')
        if acc != int(r['draft_accepted'] or 0):
            bad.append(f"accepted {acc} against the server's {r['draft_accepted']}")
        if drf != int(r['draft_n'] or 0):
            bad.append(f"drafted {drf} against the server's {r['draft_n']}")
        out.append((r, gen, bad))
    if i != n:
        out.append(({'load': load, 'kind': 'left over', 'pass': '', 'depth': ''}, steps[i:], [f'{n - i} steps not matched']))
    return out


def stats(gen, gen_n):
    d = [s for s in gen if 'drf' in s['n']]
    hist = [0] * (NMAX + 1)
    short = 0
    pos_n = [0] * NMAX   # steps that drafted at least j+1 tokens
    pos_a = [0] * NMAX   # of those, steps that accepted at least j+1
    for s in d:
        k, a = s['n'].get('drf', 0), s['n'].get('acc', 0)
        if k == NMAX:
            hist[min(a, NMAX)] += 1
        else:
            short += 1
        for j in range(min(k, NMAX)):
            pos_n[j] += 1
            if a > j:
                pos_a[j] += 1
    span = [mark(s, 's.drf.end') - mark(s, 's.drf') for s in d if mark(s, 's.drf') is not None and mark(s, 's.drf.end') is not None]
    own = [last_mark(s, 'mtp.drf.end') - mark(s, 'mtp.drf') for s in d
           if mark(s, 'mtp.drf') is not None and last_mark(s, 'mtp.drf.end') is not None]
    starts = [s['t'][0][1] for s in gen if s['t']]
    walls = [b - a for a, b in zip(starts, starts[1:])]
    P = [pos_a[j] / pos_n[j] if pos_n[j] else None for j in range(NMAX)]
    a = []
    for j in range(NMAX):
        if P[j] is None:
            a.append(None)
        elif j == 0:
            a.append(P[0])
        else:
            a.append(P[j] / P[j - 1] if P[j - 1] else None)
    steps = len(gen)
    return {
        'steps': steps, 'drafting_steps': len(d), 'drafted': sum(s['n'].get('drf', 0) for s in d),
        'accepted': sum(s['n'].get('acc', 0) for s in d),
        'tokens_per_step': gen_n / (gen_n - sum(s['n'].get('acc', 0) for s in d)) if gen_n else None,
        'hist': hist, 'short': short, 'P': P, 'a': a,
        'span_med': st.median(span) if span else None, 'span_mean': st.mean(span) if span else None,
        'own_med': st.median(own) if own else None, 'step_med': st.median(walls) if walls else None,
    }


def f(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


loads = []
if os.path.exists(os.path.join(A.src, 'loads.tsv')):
    with open(os.path.join(A.src, 'loads.tsv'), newline='') as fh:
        for r in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['timed'] != 'stopped' and r['load'] not in loads:
                loads.append(r['load'])
recs, problems = [], []
for load in loads:
    tp, cp = os.path.join(A.src, f'st-{load}.jsonl'), os.path.join(A.src, f'requests-{load}.csv')
    if not (os.path.exists(tp) and os.path.exists(cp)):
        continue
    with open(cp, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r, gen, bad in split(rows, read_trace(tp), load):
        if bad:
            problems.append(f"{load} {r.get('kind')} pass {r.get('pass')} depth {r.get('depth')}: {'; '.join(bad)}")
        if r.get('kind') not in ('decode', 'sample'):
            continue
        s = stats(gen, int(r['gen_n']))
        s.update({'load': load, 'arm': AMAP.get(r['arm'], r['arm']), 'kind': r['kind'], 'pass': r['pass'],
                  'depth': int(r['depth']), 'gen_n': int(r['gen_n']), 'gen_ms': float(r['gen_ms'])})
        s['ms_step'] = s['gen_ms'] / (s['gen_n'] - s['accepted']) if s['gen_n'] > s['accepted'] else None
        recs.append(s)
if not recs:
    sys.exit('no load with both a step trace and its requests')

out = []
out.append('# The draft by position and its time a step (step traces)\n')
out.append(f"Loads: {', '.join(dict.fromkeys(r['load'] for r in recs))}. A step drafts up to {NMAX} tokens. P_j: the share of "
           'steps whose j-th drafted token was accepted (over the steps that drafted at least j); a_j: accepted at j '
           'of the steps that accepted every token before it. Span: `s.drf` to `s.drf.end`; drafter: `mtp.drf` to '
           '`mtp.drf.end`; ms a step: the lane\'s, gen_ms over steps.\n')
out.append('| arm | load | depth | pass | trace steps | drafted | accepted | tokens a step | 3 drafted: accepted 0/1/2/3 | fewer drafted | P_1 | P_2 | P_3 | a_2 | a_3 | span median ms | span mean ms | drafter median ms | step median ms (trace) | ms a step |')
out.append('|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
order = sorted(recs, key=lambda r: (r['depth'], r['pass'], r['arm'], r['load']))
for r in order:
    out.append(f"| {r['arm']} | {r['load']} | {r['depth']:,} | {r['pass']} | {r['steps']} | {r['drafted']} | {r['accepted']} | "
               f"{f(r['tokens_per_step'])} | {'/'.join(map(str, r['hist']))} | {r['short']} | {f(r['P'][0])} | {f(r['P'][1])} | "
               f"{f(r['P'][2])} | {f(r['a'][1])} | {f(r['a'][2])} | {f(r['span_med'], 2)} | {f(r['span_mean'], 2)} | "
               f"{f(r['own_med'], 2)} | {f(r['step_med'], 2)} | {f(r['ms_step'], 2)} |")

# per arm, depth and pass: the loads pooled (counts added), and each load's P and span by value
out.append('\n## Per arm, the loads pooled\n')
out.append('| arm | depth | pass | loads | trace steps | accepted of drafted | tokens a step, each load | P_1 | P_2 | P_3 | a_2 | a_3 | span median ms, each load | ms a step, each load |')
out.append('|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|')
keys = list(dict.fromkeys((r['arm'], r['depth'], r['pass']) for r in sorted(recs, key=lambda r: (r['depth'], r['pass'], r['arm']))))
pooled = []
for arm, depth, ps in keys:
    g = [r for r in recs if (r['arm'], r['depth'], r['pass']) == (arm, depth, ps)]
    steps = sum(r['steps'] for r in g)
    acc = sum(r['accepted'] for r in g)
    drf = sum(r['drafted'] for r in g)
    hist = [sum(r['hist'][k] for r in g) for k in range(NMAX + 1)]
    n3 = sum(hist)
    P = [sum(hist[k] for k in range(j + 1, NMAX + 1)) / n3 if n3 else None for j in range(NMAX)]
    a = [P[0]] + [P[j] / P[j - 1] if P[j - 1] else None for j in range(1, NMAX)]
    pooled.append({'arm': arm, 'depth': depth, 'pass': ps, 'loads': len(g), 'steps': steps, 'accepted': acc,
                   'drafted': drf, 'P': P, 'a': a, 'tps': [r['tokens_per_step'] for r in g],
                   'span': [r['span_med'] for r in g], 'ms': [r['ms_step'] for r in g]})
    out.append(f"| {arm} | {depth:,} | {ps} | {len(g)} | {steps} | {acc} of {drf} ({acc / drf:.4f}) | "
               f"{', '.join(f(x) for x in pooled[-1]['tps'])} | {f(P[0])} | {f(P[1])} | {f(P[2])} | {f(a[1])} | {f(a[2])} | "
               f"{', '.join(f(x, 2) for x in pooled[-1]['span'])} | {', '.join(f(x, 2) for x in pooled[-1]['ms'])} |")
out.append('\nP_j and a_j pooled here are over the steps that drafted all 3 tokens (the histogram); the per-request '
           'table counts every step that drafted at least j.')
out.append('\n## Checks\n')
out.append('- every request\'s steps matched the server\'s drafted and accepted counts' if not problems else
           '\n'.join(f'- {p}' for p in problems))
text = '\n'.join(out) + '\n'
print(text)
if A.out:
    with open(A.out, 'w') as fh:
        fh.write(text)
if A.tsv:
    with open(A.tsv, 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['arm', 'load', 'kind', 'pass', 'depth', 'steps', 'drafted', 'accepted', 'tokens_per_step', 'acc0', 'acc1',
                    'acc2', 'acc3', 'fewer_drafted', 'P1', 'P2', 'P3', 'a2', 'a3', 'span_median_ms', 'span_mean_ms',
                    'drafter_median_ms', 'step_median_ms_trace', 'ms_a_step'])
        for r in order:
            w.writerow([r['arm'], r['load'], r['kind'], r['pass'], r['depth'], r['steps'], r['drafted'], r['accepted'],
                        f(r['tokens_per_step']), *r['hist'], r['short'], f(r['P'][0], 4), f(r['P'][1], 4), f(r['P'][2], 4),
                        f(r['a'][1], 4), f(r['a'][2], 4), f(r['span_med']), f(r['span_mean']), f(r['own_med']),
                        f(r['step_med']), f(r['ms_step'])])
