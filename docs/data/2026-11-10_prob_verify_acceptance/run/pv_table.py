#!/usr/bin/env python3
"""pv_table.py: the tables of the probabilistic-verification replay (registry Q1173). Counters only, no time.

    python3 -I pv_table.py WORK [--control pv0] [--out DIR] [--boot 10000] [--deep 8192]

Reads WORK/replay-<arm>-<n>.jsonl (the rows of bench/agent_check/replay.py: one per request and load) and
WORK/server-<arm>-<n>.log (llama-server's output with LLAMA_TRACE=1), and writes into --out (default WORK/tables):

  requests.tsv    one row per request and load: prompt tokens, generated, drafted, accepted, the share accepted, steps,
                  tokens a step, tokens sampled on the device, finish reason, the reply's hash
  loads.tsv       one row per load: the counters summed over its requests, plus the recorded session's own
  compare.tsv     each arm against the control and each arm's second load against its first, on tokens a step and on
                  the share accepted: every load by value, the difference, the spreads, the class, bootstrap intervals
  same_reply.tsv  per pair of loads: the requests with the same reply and with the same counters, by prompt depth
  same_reply_sessions.tsv  per pair of loads and session: the requests with the same reply, the first one that differs,
                  and how many after it are the same again
  log_check.tsv   per load, from the server log: sampling on the device, steps, restores inside a step, error lines
  steps.tsv       per load: verify steps by drafted and accepted tokens (the LLAMA_TRACE lines)
  summary.md      the tables above as Markdown

Definitions. generated: the request's predicted_n. drafted and accepted: draft_n and draft_n_accepted of the server's
timings (0 when a request drafted nothing). Share accepted: accepted / drafted. Steps: generated - accepted, the
passes of the main model that yielded tokens. Tokens a step: generated / steps. "Pooled" sums the counters over
requests before dividing; "per request" divides first and averages over requests.

An arm's value is the mean of its loads' pooled values. Spread: the difference between an arm's two loads. Class of an
arm against the control: gain when its value is above the control's by more than the spread, loss when below by more
than it, else negligible. Two spreads are printed: the control's own, and the larger of the control's and the arm's
(the rule of bench/deep_lane).

Intervals: 95 % percentile bootstrap of the difference, resampling the requests with replacement (each request keeps
its rows in every load, so the comparison stays paired); a second interval resamples whole traces (the sessions), since
the requests of one session share their task.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('work', type=Path)
ap.add_argument('--control', default='pv0')
ap.add_argument('--out', type=Path)
ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--deep', type=int, default=8192, help='prompt tokens from which a request counts as deep')
A = ap.parse_args()
OUT = A.out or A.work / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the rows
loads = {}  # label -> {(trace, idx): row}
for p in sorted(A.work.glob('replay-*.jsonl')):
    label = p.name[len('replay-'):-len('.jsonl')]
    rows = {}
    for line in p.open():
        if line.strip():
            r = json.loads(line)
            if not r.get('error'):
                rows[(r['trace'], r['idx'])] = r
    loads[label] = rows
if not loads:
    raise SystemExit(f'no replay-*.jsonl in {A.work}')
labels = sorted(loads, key=lambda l: (l.rsplit('-', 1)[0] != A.control, l))
arms = []
for l in labels:
    if l.rsplit('-', 1)[0] not in arms:
        arms.append(l.rsplit('-', 1)[0])
arm_loads = {a: [l for l in labels if l.rsplit('-', 1)[0] == a] for a in arms}
keys = sorted(set.intersection(*(set(v) for v in loads.values())))
missing = {l: len(set.union(*(set(v) for v in loads.values())) - set(loads[l])) for l in labels}
traces = sorted({k[0] for k in keys})


def cnt(r):
    """(generated, drafted, accepted) of a row."""
    return r.get('predicted_n') or 0, r.get('draft_n') or 0, r.get('draft_n_accepted') or 0


def ratio(a, b):
    return a / b if b else float('nan')


C = {l: {k: cnt(r) for k, r in rows.items()} for l, rows in loads.items()}


def pooled(label, ks, counts=None):
    """(generated, drafted, accepted) summed over the requests ks of a load."""
    rows = counts or C[label]
    g = d = a = 0
    for k in ks:
        x = rows[k]
        g, d, a = g + x[0], d + x[1], a + x[2]
    return g, d, a


def tps(c):
    return ratio(c[0], c[0] - c[2])


def acc(c):
    return ratio(c[2], c[1])


METRICS = (('tokens a step', tps, 3), ('share accepted', acc, 4))


def task(trace):
    """The task id of a trace "<arm>/<task>/a<k>/t<n>"."""
    return trace.split('/')[1] if trace.count('/') >= 3 else trace


def depth(k):
    r = loads[labels[0]][k]
    return (r.get('prompt_n') or 0) + (r.get('cache_n') or 0)


# ---------------------------------------------------------------- requests.tsv
with (OUT / 'requests.tsv').open('w') as f:
    f.write('task\trequest\tprompt_tokens\treply_limit\tload\tgenerated\tdrafted\taccepted\tshare_accepted\tsteps\t'
            'tokens_per_step\tsampled_on_device\tgrammar_tokens\tgrammar_rejected\tfinish_reason\treply_md5\n')
    for k in keys:
        for l in labels:
            r = loads[l][k]
            g, d, a = cnt(r)
            f.write('\t'.join(str(x) for x in (
                task(k[0]), k[1] + 1, (r.get('prompt_n') or 0) + (r.get('cache_n') or 0), r.get('max_tokens'), l, g, d, a,
                f'{acc((g, d, a)):.4f}' if d else '', g - a, f'{tps((g, d, a)):.3f}',
                r.get('sampled_on_device', ''), r.get('grammar_n', ''), r.get('grammar_rejected_n', ''),
                r.get('finish_reason'), (r.get('reply_md5') or '')[:12])) + '\n')

# ---------------------------------------------------------------- loads.tsv
load_rows = []
for l in labels:
    c = pooled(l, keys)
    per = [tps(cnt(loads[l][k])) for k in keys]
    load_rows.append(dict(arm=l.rsplit('-', 1)[0], load=l, requests=len(keys), generated=c[0], drafted=c[1],
                          accepted=c[2], acc=acc(c), steps=c[0] - c[2], tps=tps(c), tps_req=sum(per) / len(per)))
rec = {k: dict(loads[labels[0]][k].get('rec') or {}) for k in keys}
rec_c = pooled(None, keys, {k: cnt(rec[k]) for k in keys})
if rec_c[1]:
    per = [tps(cnt(rec[k])) for k in keys if cnt(rec[k])[0]]
    load_rows.append(dict(arm='recorded', load='the recorded session', requests=len(keys), generated=rec_c[0],
                          drafted=rec_c[1], accepted=rec_c[2], acc=acc(rec_c), steps=rec_c[0] - rec_c[2],
                          tps=tps(rec_c), tps_req=sum(per) / len(per)))
with (OUT / 'loads.tsv').open('w') as f:
    f.write('arm\tload\trequests\tgenerated\tdrafted\taccepted\tshare_accepted\tsteps\ttokens_per_step\t'
            'tokens_per_step_mean_of_requests\n')
    for r in load_rows:
        f.write(f"{r['arm']}\t{r['load']}\t{r['requests']}\t{r['generated']}\t{r['drafted']}\t{r['accepted']}\t"
                f"{r['acc']:.4f}\t{r['steps']}\t{r['tps']:.3f}\t{r['tps_req']:.3f}\n")

# ---------------------------------------------------------------- compare.tsv
rng = random.Random(1)
by_trace = defaultdict(list)
for k in keys:
    by_trace[k[0]].append(k)
samples_req = [[rng.choice(keys) for _ in keys] for _ in range(A.boot)]
samples_trace = [[k for t in (rng.choice(traces) for _ in traces) for k in by_trace[t]] for _ in range(A.boot)]


def value(ls, fn, ks):
    """An arm's value on the requests ks: the mean of its loads' pooled values."""
    return sum(fn(pooled(l, ks)) for l in ls) / len(ls)


def interval(stat, samples):
    v = sorted(x for x in (stat(s) for s in samples) if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def klass(delta, spread):
    if spread is None:
        return 'no class (one load)'
    return 'gain' if delta > spread else 'loss' if delta < -spread else 'negligible'


def spread_of(ls, fn):
    v = [fn(pooled(l, keys)) for l in ls]
    return max(v) - min(v) if len(v) > 1 else None


compare = []
pairs = [(a, arm_loads[a], A.control, arm_loads[A.control], 'arm against the control') for a in arms if a != A.control]
pairs += [(f'{a}-2', arm_loads[a][1:2], f'{a}-1', arm_loads[a][0:1], 'second load against the first')
          for a in arms if len(arm_loads[a]) > 1]
for name_a, la, name_b, lb, kind in pairs:
    for metric, fn, nd in METRICS:
        va, vb = [fn(pooled(l, keys)) for l in la], [fn(pooled(l, keys)) for l in lb]
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        delta = ma - mb
        stat = lambda ks: value(la, fn, ks) - value(lb, fn, ks)  # noqa: E731
        lo, hi = interval(stat, samples_req)
        tlo, thi = interval(stat, samples_trace)
        row = dict(kind=kind, a=name_a, b=name_b, metric=metric, nd=nd, a_loads=va, b_loads=vb, a_value=ma, b_value=mb,
                   delta=delta, percent=100 * delta / mb, lo=lo, hi=hi, tlo=tlo, thi=thi)
        if kind == 'arm against the control':
            sa, sb = spread_of(la, fn), spread_of(lb, fn)
            row.update(spread_a=sa, spread_b=sb, class_control=klass(delta, sb),
                       class_larger=klass(delta, None if sa is None or sb is None else max(sa, sb)))
        # per request: each request's value averaged over the loads, then the mean difference over requests
        per = []
        for k in keys:
            xa = [fn(cnt(loads[l][k])) for l in la]
            xb = [fn(cnt(loads[l][k])) for l in lb]
            if all(x == x for x in xa + xb):
                per.append((k, sum(xa) / len(xa) - sum(xb) / len(xb)))
        pd = dict(per)
        row.update(req_n=len(per), req_mean=sum(d for _, d in per) / len(per),
                   req_up=sum(d > 0 for _, d in per), req_down=sum(d < 0 for _, d in per),
                   req_same=sum(d == 0 for _, d in per))
        row['rlo'], row['rhi'] = interval(lambda ks: ratio(sum(pd[k] for k in ks if k in pd),
                                                           sum(k in pd for k in ks)), samples_req)
        compare.append(row)


def num(x, nd, sign=False):
    return '' if x is None else f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


with (OUT / 'compare.tsv').open('w') as f:
    f.write('kind\ta\tb\tmetric\ta_loads\tb_loads\ta_value\tb_value\tdelta\tpercent\tspread_a\tspread_b\t'
            'class_control_spread\tclass_larger_spread\tboot_requests_lo\tboot_requests_hi\tboot_traces_lo\t'
            'boot_traces_hi\trequests\tper_request_mean_delta\tper_request_lo\tper_request_hi\trequests_higher\t'
            'requests_lower\trequests_equal\n')
    for r in compare:
        nd = r['nd']
        f.write('\t'.join([r['kind'], r['a'], r['b'], r['metric'], ' '.join(num(v, nd) for v in r['a_loads']),
                           ' '.join(num(v, nd) for v in r['b_loads']), num(r['a_value'], nd), num(r['b_value'], nd),
                           num(r['delta'], nd, True), num(r['percent'], 2, True), num(r.get('spread_a'), nd),
                           num(r.get('spread_b'), nd), r.get('class_control', ''), r.get('class_larger', ''),
                           num(r['lo'], nd, True), num(r['hi'], nd, True), num(r['tlo'], nd, True),
                           num(r['thi'], nd, True), str(r['req_n']), num(r['req_mean'], nd, True),
                           num(r['rlo'], nd, True), num(r['rhi'], nd, True), str(r['req_up']), str(r['req_down']),
                           str(r['req_same'])]) + '\n')

# ---------------------------------------------------------------- same_reply.tsv
shallow = [k for k in keys if depth(k) < A.deep]
deepk = [k for k in keys if depth(k) >= A.deep]
same = []
for i, la in enumerate(labels):
    for lb in labels[i + 1:]:
        aa, ab = la.rsplit('-', 1)[0], lb.rsplit('-', 1)[0]
        kind = 'one arm, two loads' if aa == ab else 'arm against the control' if A.control in (aa, ab) else 'two arms'
        eq = lambda ks: sum(loads[la][k].get('reply_md5') == loads[lb][k].get('reply_md5') for k in ks)  # noqa: E731
        ec = lambda ks: sum(cnt(loads[la][k]) == cnt(loads[lb][k]) for k in ks)  # noqa: E731
        same.append(dict(kind=kind, a=la, b=lb, n=len(keys), same=eq(keys), counters=ec(keys), n_shallow=len(shallow),
                         same_shallow=eq(shallow), n_deep=len(deepk), same_deep=eq(deepk)))
with (OUT / 'same_reply.tsv').open('w') as f:
    f.write(f'kind\tload_a\tload_b\trequests\tsame_reply\tsame_counters\trequests_below_{A.deep}\tsame_reply_below\t'
            f'requests_from_{A.deep}\tsame_reply_from\n')
    for r in same:
        f.write(f"{r['kind']}\t{r['a']}\t{r['b']}\t{r['n']}\t{r['same']}\t{r['counters']}\t{r['n_shallow']}\t"
                f"{r['same_shallow']}\t{r['n_deep']}\t{r['same_deep']}\n")

# ---------------------------------------------------------------- same_reply_sessions.tsv
order = sorted(traces, key=lambda t: min(loads[labels[0]][k].get('t_wall') or 0 for k in by_trace[t]))
sess = []
for i, la in enumerate(labels):
    for lb in labels[i + 1:]:
        for n, t in enumerate(order, 1):
            eq = [loads[la][k].get('reply_md5') == loads[lb][k].get('reply_md5') for k in sorted(by_trace[t])]
            first = eq.index(False) + 1 if False in eq else 0
            sess.append(dict(a=la, b=lb, session=n, task=task(t), n=len(eq), same=sum(eq), first=first,
                             again=sum(eq[first:]) if first else 0))
with (OUT / 'same_reply_sessions.tsv').open('w') as f:
    f.write('load_a\tload_b\tsession\ttask\trequests\tsame_reply\tfirst_request_that_differs\tsame_again_after_it\n')
    for r in sess:
        f.write(f"{r['a']}\t{r['b']}\t{r['session']}\t{r['task']}\t{r['n']}\t{r['same']}\t"
                f"{r['first'] or ''}\t{r['again']}\n")

# ---------------------------------------------------------------- the server logs
SAMPLING = re.compile(r'sampling =\s+(\d+) tokens, (\d+) drawn on the device, (\d+) while the grammar applied, '
                      r'(\d+) rejected \((\d+) after a device draw\), (\d+) forced')
STEP = re.compile(r'accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?')
ACCEPT = re.compile(r'draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)')
LEVEL = re.compile(r'^\d[\d.]* ([A-Z]) ')
logs, steps = [], []
for l in labels:
    p = A.work / f'server-{l}.log'
    if not p.exists():
        continue
    s = dict(load=l, lines=0, sampling_lines=0, sampled=0, device=0, grammar=0, rejected=0, rejected_device=0, forced=0,
             step_lines=0, step_restores=0, step_accepted=0, step_drafted=0, accept_lines=0, log_accepted=0,
             log_drafted=0, ckpt_restores=0, error_lines=0, cuda_errors=0, keep_draws='', flags='')
    hist = Counter()
    for line in p.open(errors='replace'):
        s['lines'] += 1
        if s['lines'] == 1:
            s['flags'] = ' '.join(w for w in ('--spec-draft-sampling probabilistic', '--spec-draft-p-min 0',
                                              'LLAMA_SPEC_KEEP_DRAWS=1', 'LLAMA_TRACE=1') if w in line)
        m = SAMPLING.search(line)
        if m:
            v = [int(x) for x in m.groups()]
            s['sampling_lines'] += 1
            for name, x in zip(('sampled', 'device', 'grammar', 'rejected', 'rejected_device', 'forced'), v):
                s[name] += x
        m = STEP.search(line)
        if m:
            s['step_lines'] += 1
            s['step_accepted'] += int(m.group(1))
            s['step_drafted'] += int(m.group(2))
            s['step_restores'] += bool(m.group(3))
            hist[(int(m.group(2)), int(m.group(1)), bool(m.group(3)))] += 1
        m = ACCEPT.search(line)
        if m:
            s['accept_lines'] += 1
            s['log_accepted'] += int(m.group(1))
            s['log_drafted'] += int(m.group(2))
        if 'restored context checkpoint' in line:
            s['ckpt_restores'] += 1
        if 'restoring speculative checkpoint' in line:
            s['step_restores'] += 1
        if 'draws kept across a replay' in line:
            s['keep_draws'] = line.split('draws kept across a replay')[1].strip()
        m = LEVEL.match(line)
        if m and m.group(1) == 'E':
            s['error_lines'] += 1
        if 'CUDA error' in line:
            s['cuda_errors'] += 1
    c = pooled(l, keys)
    rows = [loads[l][k] for k in keys]
    s.update(generated=c[0], drafted=c[1], accepted=c[2],
             timings_device=sum(r.get('sampled_on_device') or 0 for r in rows),
             timings_with_grammar=sum('sampled_on_device' in r for r in rows),
             timings_short=sum('sampled_on_device' in r and r['sampled_on_device'] < (r.get('predicted_n') or 0)
                               for r in rows))
    logs.append(s)
    for (d, a, restore), n in sorted(hist.items()):
        steps.append((l, d, a, int(restore), n))
LOG_COLS = ('load', 'flags', 'generated', 'sampling_lines', 'sampled', 'device', 'grammar', 'rejected',
            'rejected_device', 'forced', 'timings_with_grammar', 'timings_device', 'timings_short', 'step_lines',
            'step_drafted', 'step_accepted', 'step_restores', 'accept_lines', 'log_drafted', 'log_accepted', 'drafted',
            'accepted', 'ckpt_restores', 'keep_draws', 'error_lines', 'cuda_errors')
with (OUT / 'log_check.tsv').open('w') as f:
    f.write('\t'.join(LOG_COLS) + '\n')
    for s in logs:
        f.write('\t'.join(str(s[c]) for c in LOG_COLS) + '\n')
with (OUT / 'steps.tsv').open('w') as f:
    f.write('load\tdrafted\taccepted\trestore_inside_the_step\tsteps\n')
    for row in steps:
        f.write('\t'.join(str(x) for x in row) + '\n')

# ---------------------------------------------------------------- summary.md
md = [f'# Probabilistic verification and kept draws: the counters ({len(keys)} requests in {len(traces)} sessions, '
      f'{len(labels)} loads)', '']
if any(missing.values()):
    md += ['Requests missing from a load (left out of every table): ' +
           ', '.join(f'{l} {n}' for l, n in missing.items() if n), '']
md += ['## Per load', '',
       '| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step (pooled) | '
       'Tokens a step (mean of requests) |', '|---|---|---:|---|---:|---:|---:|---:|']
for r in load_rows:
    md.append(f"| {r['arm']} | {r['load']} | {r['generated']:,} | {r['accepted']:,} of {r['drafted']:,} | "
              f"{r['acc']:.4f} | {r['steps']:,} | {r['tps']:.3f} | {r['tps_req']:.3f} |")
for kind, title in (('arm against the control', f'## Each arm against `{A.control}`'),
                    ('second load against the first', '## One arm, its second load against its first')):
    md += ['', title, '']
    arm_kind = kind == 'arm against the control'
    md.append('| Metric | A | B | A, each load | B, each load | A − B | % | ' +
              ('Spread of A | Spread of B | Class (B\'s spread) | Class (the larger spread) | ' if arm_kind else '') +
              '95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |')
    md.append('|---|---|---|---|---|---:|---:|' + ('---:|---:|---|---|' if arm_kind else '') + '---|---|---:|---|---|')
    for r in compare:
        if r['kind'] != kind:
            continue
        nd = r['nd']
        cells = [r['metric'], r['a'], r['b'], ', '.join(num(v, nd) for v in r['a_loads']),
                 ', '.join(num(v, nd) for v in r['b_loads']), num(r['delta'], nd, True), num(r['percent'], 2, True)]
        if arm_kind:
            cells += [num(r['spread_a'], nd), num(r['spread_b'], nd), r['class_control'], r['class_larger']]
        cells += [f"{num(r['lo'], nd, True)} to {num(r['hi'], nd, True)}",
                  f"{num(r['tlo'], nd, True)} to {num(r['thi'], nd, True)}", num(r['req_mean'], nd, True),
                  f"{num(r['rlo'], nd, True)} to {num(r['rhi'], nd, True)}",
                  f"{r['req_up']} / {r['req_down']} / {r['req_same']}"]
        md.append('| ' + ' | '.join(cells) + ' |')
md += ['', '## The same reply in two loads', '',
       f'| Loads | Kind | Same reply | Same counters | Below {A.deep:,} prompt tokens | From {A.deep:,} |',
       '|---|---|---|---|---|---|']
for r in same:
    md.append(f"| {r['a']}, {r['b']} | {r['kind']} | {r['same']} of {r['n']} | {r['counters']} of {r['n']} | "
              f"{r['same_shallow']} of {r['n_shallow']} | {r['same_deep']} of {r['n_deep']} |")
md += ['', '## The same reply, by session', '',
       '| Loads | Sessions the same in every request | Sessions that differ from their first request | Sessions that differ '
       'from a later request | Requests the same again after a session\'s first differing one |', '|---|---:|---:|---:|---:|']
for i, la in enumerate(labels):
    for lb in labels[i + 1:]:
        rows = [r for r in sess if r['a'] == la and r['b'] == lb]
        md.append(f"| {la}, {lb} | {sum(not r['first'] for r in rows)} of {len(rows)} | "
                  f"{sum(r['first'] == 1 for r in rows)} | {sum(r['first'] > 1 for r in rows)} | "
                  f"{sum(r['again'] for r in rows)} of {sum(r['n'] - r['first'] for r in rows if r['first'])} |")
if logs:
    md += ['', '## The server logs', '',
           '| Load | Switches on its line | Generated | Requests with a grammar | Their tokens: sampled, drawn on the device | '
           'Rejected (after a device draw) | Verify steps | Restores inside a step | Accepted of drafted: step lines, '
           'request lines, timings | Checkpoint restores at a prompt | Draws kept | Error lines |',
           '|---|---|---:|---:|---|---|---:|---:|---|---:|---|---:|']
    for s in logs:
        md.append(f"| {s['load']} | {s['flags']} | {s['generated']:,} | {s['sampling_lines']} | {s['sampled']:,}, "
                  f"{s['device']:,} | {s['rejected']} ({s['rejected_device']}) | {s['step_lines']:,} | "
                  f"{s['step_restores']} | {s['step_accepted']:,} of {s['step_drafted']:,}; {s['log_accepted']:,} of "
                  f"{s['log_drafted']:,}; {s['accepted']:,} of {s['drafted']:,} | {s['ckpt_restores']} | "
                  f"{s['keep_draws'] or 'off'} | {s['error_lines'] + s['cuda_errors']} |")
(OUT / 'summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
