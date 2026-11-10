#!/usr/bin/env python3
"""pv_rig_table.py: the tables of the probabilistic-verification pair timed on the rig (registry Q731).

    python3 -I pv_rig_table.py WORK [--control pv0] [--out DIR] [--boot 10000]

Reads WORK/replay-<arm>-<n>.jsonl (the rows of bench/agent_check/replay.py: one per request and load, with the
server's timings and counters) and, per load, WORK/rig/<arm>-<n>/ as bench/agent_check/rig_server.sh writes it
(server-<arm>-<n>.log, gpu.csv, meminfo.log, ram_watchdog-<arm>-<n>.tsv), and writes into --out (default WORK/tables):

  requests.tsv    one row per request and load: prompt tokens, generated, drafted, accepted, steps, tokens a step, the
                  generation's time, ms a step, tok/s, the prompt's time, grammar counters, finish reason, reply hash
  loads.tsv       one row per load: the same summed over its requests, plus the recorded session's own
  compare.tsv     each arm against the control and each arm's second load against its first, per metric: every load by
                  value, the difference, the spreads, the class, bootstrap intervals, per-request counts
  by_length.tsv   the requests in two halves by their reply limit (the recorded reply's length, the same in every load):
                  per half and load the pooled counters, ms a step and tok/s
  fit.tsv         per arm, a least-squares split of a reply's generation time into a cost per reply, a cost per step
                  and the step's growth with context depth, with bootstrap intervals and the difference to the control
  same_reply.tsv  per pair of loads: the requests with the same reply and with the same counters
  log_check.tsv   per load, from the server log: the switches on its line, sampling on the device, grammar rejections,
                  accepted and drafted tokens (the log's lines hold the untimed warm-up request too; its counters are
                  read from WORK/warmup-<arm>-<n>.json and listed beside them), checkpoint restores, error lines
  conditions.tsv  per load: when its replay ran, the cards' temperature, clock, power and memory during it, the host's
                  lowest free memory, its swap use at the replay's start and end and what it swapped in and out during
                  the replay (vmstat.log), watchdog kills
  summary.md      the tables above as Markdown

Definitions, per request. generated: predicted_n. drafted and accepted: draft_n and draft_n_accepted of the server's
timings. Steps: generated - accepted, the passes of the main model that yielded tokens. Tokens a step: generated /
steps. The generation's time: predicted_ms, which the server counts from the end of the prompt, so a request's first
token (it comes from the prompt's last pass) is outside it: the time holds steps - 1 verify steps. Ms a step:
predicted_ms / (steps - 1). Generation tok/s: generated / predicted_ms, as the agent check's report counts it. Prompt
tok/s: prompt_n / prompt_ms (the prompts are the same in every load, so it is the comparison's control). "Pooled" sums
over requests before dividing; "per request" divides first.

An arm's value is the mean of its loads' pooled values. Spread: the difference between an arm's two loads. Class of an
arm against the control: gain when its value is better than the control's by more than the spread (higher, or lower
for ms a step), loss when worse by more than it, else negligible. Two classes are printed: against the control's own
spread, and against the larger of the control's and the arm's (the rule of bench/deep_lane).

Intervals: 95 % percentile bootstrap of the difference, resampling the requests with replacement (each request keeps
its rows in every load, so the comparison stays paired); a second interval resamples whole traces (the sessions), since
the requests of one session share their task.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('work', type=Path)
ap.add_argument('--control', default='pv0')
ap.add_argument('--out', type=Path)
ap.add_argument('--boot', type=int, default=10000)
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
arm_of = lambda l: l.rsplit('-', 1)[0]  # noqa: E731
labels = sorted(loads, key=lambda l: (arm_of(l) != A.control, l))
arms = []
for l in labels:
    if arm_of(l) not in arms:
        arms.append(arm_of(l))
arm_loads = {a: [l for l in labels if arm_of(l) == a] for a in arms}
keys = sorted(set.intersection(*(set(v) for v in loads.values())))
missing = {l: len(set.union(*(set(v) for v in loads.values())) - set(loads[l])) for l in labels}
traces = sorted({k[0] for k in keys})


def sums(r):
    """(generated, drafted, accepted, generation ms, timed steps, prompt tokens processed, prompt ms) of a row."""
    g, a = r.get('predicted_n') or 0, r.get('draft_n_accepted') or 0
    return (g, r.get('draft_n') or 0, a, r.get('predicted_ms') or 0.0, max(0, g - a - 1), r.get('prompt_n') or 0,
            r.get('prompt_ms') or 0.0)


def ratio(a, b):
    return a / b if b else float('nan')


S = {l: {k: sums(r) for k, r in rows.items()} for l, rows in loads.items()}


def pooled(label, ks, table=None):
    rows = table or S[label]
    return tuple(map(sum, zip(*(rows[k] for k in ks)))) if ks else (0,) * 7


# name, function of the sums, decimals, +1 when higher is better and -1 when lower is
METRICS = (('generation tok/s', lambda s: ratio(1000 * s[0], s[3]), 2, 1),
           ('ms a step', lambda s: ratio(s[3], s[4]), 2, -1),
           ('tokens a step', lambda s: ratio(s[0], s[0] - s[2]), 3, 1),
           ('share accepted', lambda s: ratio(s[2], s[1]), 4, 1),
           ('prompt tok/s', lambda s: ratio(1000 * s[5], s[6]), 2, 1))
TOKS, MS_STEP, TPS, ACC, PTOKS = (m[1] for m in METRICS)


def task(trace):
    """The task id of a trace "<arm>/<task>/a<k>/t<n>"."""
    return trace.split('/')[1] if trace.count('/') >= 3 else trace


def fmt(x, nd, sign=False):
    return '' if x is None or x != x else f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


# ---------------------------------------------------------------- requests.tsv
with (OUT / 'requests.tsv').open('w') as f:
    f.write('task\trequest\tprompt_tokens\tprompt_tokens_processed\treply_limit\tload\tgenerated\tdrafted\taccepted\t'
            'steps\ttokens_per_step\tgeneration_ms\tms_per_step\tgeneration_tok_per_s\tprompt_ms\tgrammar_tokens\t'
            'grammar_rejected\tfinish_reason\treply_md5\n')
    for k in keys:
        for l in labels:
            r, s = loads[l][k], S[l][k]
            f.write('\t'.join(str(x) for x in (
                task(k[0]), k[1] + 1, (r.get('prompt_n') or 0) + (r.get('cache_n') or 0), s[5], r.get('max_tokens'), l,
                s[0], s[1], s[2], s[0] - s[2], fmt(TPS(s), 3), fmt(s[3], 1), fmt(MS_STEP(s), 2), fmt(TOKS(s), 2),
                fmt(s[6], 1), r.get('grammar_n', ''), r.get('grammar_rejected_n', ''), r.get('finish_reason'),
                (r.get('reply_md5') or '')[:12])) + '\n')

# ---------------------------------------------------------------- loads.tsv
load_rows = []
for l in labels:
    s = pooled(l, keys)
    rows = [loads[l][k] for k in keys]
    load_rows.append(dict(arm=arm_of(l), load=l, s=s,
                          at_limit=sum((r.get('predicted_n') or 0) >= (r.get('max_tokens') or float('inf')) for r in rows),
                          grammar=sum(r.get('grammar_n') or 0 for r in rows),
                          rejected=sum(r.get('grammar_rejected_n') or 0 for r in rows)))
rec = {k: sums(dict(loads[labels[0]][k].get('rec') or {})) for k in keys}
rec_s = pooled(None, keys, rec)
if rec_s[0]:
    load_rows.append(dict(arm='recorded', load='the recorded session', s=rec_s, at_limit='', grammar='', rejected=''))
with (OUT / 'loads.tsv').open('w') as f:
    f.write('arm\tload\trequests\tgenerated\tdrafted\taccepted\tshare_accepted\tsteps\ttokens_per_step\t'
            'generation_s\tms_per_step\tgeneration_tok_per_s\tprompt_tokens_processed\tprompt_s\tprompt_tok_per_s\t'
            'replies_at_the_limit\tgrammar_tokens\tgrammar_rejected\n')
    for r in load_rows:
        s = r['s']
        f.write(f"{r['arm']}\t{r['load']}\t{len(keys)}\t{s[0]}\t{s[1]}\t{s[2]}\t{fmt(ACC(s), 4)}\t{s[0] - s[2]}\t"
                f"{fmt(TPS(s), 3)}\t{s[3] / 1000:.1f}\t{fmt(MS_STEP(s), 2)}\t{fmt(TOKS(s), 2)}\t{s[5]}\t"
                f"{s[6] / 1000:.1f}\t{fmt(PTOKS(s), 2)}\t{r['at_limit']}\t{r['grammar']}\t{r['rejected']}\n")

# ---------------------------------------------------------------- compare.tsv
rng = random.Random(1)
by_trace = defaultdict(list)
for k in keys:
    by_trace[k[0]].append(k)
samples_req = [[rng.choice(keys) for _ in keys] for _ in range(A.boot)]
samples_trace = [[k for t in (rng.choice(traces) for _ in traces) for k in by_trace[t]] for _ in range(A.boot)]


def interval(stat, samples):
    v = sorted(x for x in (stat(s) for s in samples) if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def klass(delta, spread, better):
    if spread is None:
        return 'no class (one load)'
    return 'gain' if better * delta > spread else 'loss' if better * delta < -spread else 'negligible'


compare = []
pairs = [(a, arm_loads[a], A.control, arm_loads[A.control], 'arm against the control') for a in arms if a != A.control]
pairs += [(f'{a}-2', arm_loads[a][1:2], f'{a}-1', arm_loads[a][0:1], 'second load against the first')
          for a in arms if len(arm_loads[a]) > 1]
for name_a, la, name_b, lb, kind in pairs:
    # the pooled sums of every load on every resample, once for all metrics
    boot = {}
    for which, samples in (('req', samples_req), ('trace', samples_trace)):
        boot[which] = [{l: pooled(l, ks) for l in la + lb} for ks in samples]
    for metric, fn, nd, better in METRICS:
        va, vb = [fn(pooled(l, keys)) for l in la], [fn(pooled(l, keys)) for l in lb]
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        delta = ma - mb
        stat = lambda p: sum(fn(p[l]) for l in la) / len(la) - sum(fn(p[l]) for l in lb) / len(lb)  # noqa: E731
        lo, hi = interval(stat, boot['req'])
        tlo, thi = interval(stat, boot['trace'])
        row = dict(kind=kind, a=name_a, b=name_b, metric=metric, nd=nd, better='higher' if better > 0 else 'lower',
                   a_loads=va, b_loads=vb, a_value=ma, b_value=mb, delta=delta, percent=100 * delta / mb, lo=lo, hi=hi,
                   tlo=tlo, thi=thi)
        if kind == 'arm against the control':
            sa = max(va) - min(va) if len(va) > 1 else None
            sb = max(vb) - min(vb) if len(vb) > 1 else None
            row.update(spread_a=sa, spread_b=sb, class_control=klass(delta, sb, better),
                       class_larger=klass(delta, None if sa is None or sb is None else max(sa, sb), better))
        # per request: each request's value averaged over the loads, then the mean difference over requests
        pd = {}
        for k in keys:
            xa, xb = [fn(S[l][k]) for l in la], [fn(S[l][k]) for l in lb]
            if all(x == x for x in xa + xb):
                pd[k] = sum(xa) / len(xa) - sum(xb) / len(xb)
        row.update(req_n=len(pd), req_mean=sum(pd.values()) / len(pd), req_up=sum(d > 0 for d in pd.values()),
                   req_down=sum(d < 0 for d in pd.values()), req_same=sum(d == 0 for d in pd.values()))
        row['rlo'], row['rhi'] = interval(lambda ks: ratio(sum(pd[k] for k in ks if k in pd),
                                                           sum(k in pd for k in ks)), samples_req)
        compare.append(row)

with (OUT / 'compare.tsv').open('w') as f:
    f.write('kind\ta\tb\tmetric\tbetter\ta_loads\tb_loads\ta_value\tb_value\tdelta\tpercent\tspread_a\tspread_b\t'
            'class_control_spread\tclass_larger_spread\tboot_requests_lo\tboot_requests_hi\tboot_traces_lo\t'
            'boot_traces_hi\trequests\tper_request_mean_delta\tper_request_lo\tper_request_hi\trequests_higher\t'
            'requests_lower\trequests_equal\n')
    for r in compare:
        nd = r['nd']
        f.write('\t'.join([r['kind'], r['a'], r['b'], r['metric'], r['better'],
                           ' '.join(fmt(v, nd) for v in r['a_loads']), ' '.join(fmt(v, nd) for v in r['b_loads']),
                           fmt(r['a_value'], nd), fmt(r['b_value'], nd), fmt(r['delta'], nd, True),
                           fmt(r['percent'], 2, True), fmt(r.get('spread_a'), nd), fmt(r.get('spread_b'), nd),
                           r.get('class_control', ''), r.get('class_larger', ''), fmt(r['lo'], nd, True),
                           fmt(r['hi'], nd, True), fmt(r['tlo'], nd, True), fmt(r['thi'], nd, True), str(r['req_n']),
                           fmt(r['req_mean'], nd, True), fmt(r['rlo'], nd, True), fmt(r['rhi'], nd, True),
                           str(r['req_up']), str(r['req_down']), str(r['req_same'])]) + '\n')

# ---------------------------------------------------------------- by_length.tsv
limit = lambda k: loads[labels[0]][k].get('max_tokens') or 0  # noqa: E731
by_limit = sorted(keys, key=lambda k: (limit(k), k))
halves = [('the shorter half', by_limit[:len(by_limit) // 2]), ('the longer half', by_limit[len(by_limit) // 2:])]
bylen = [dict(group=name, lo=limit(ks[0]), hi=limit(ks[-1]), n=len(ks), load=l, s=pooled(l, ks))
         for name, ks in halves if ks for l in labels]
with (OUT / 'by_length.tsv').open('w') as f:
    f.write('group\treply_limit_from\treply_limit_to\trequests\tload\tgenerated\tsteps\ttokens_per_step\tgeneration_s\t'
            'ms_per_step\tgeneration_tok_per_s\n')
    for r in bylen:
        s = r['s']
        f.write(f"{r['group']}\t{r['lo']}\t{r['hi']}\t{r['n']}\t{r['load']}\t{s[0]}\t{s[0] - s[2]}\t{fmt(TPS(s), 3)}\t"
                f"{s[3] / 1000:.1f}\t{fmt(MS_STEP(s), 2)}\t{fmt(TOKS(s), 2)}\n")

# ---------------------------------------------------------------- fit.tsv
# Per arm, least squares over its rows (every request in each of its loads), two models:
#   generation ms = per_reply + timed steps * (per_step + per_step_per_1000 * depth / 1000)
#   the same + per_rejection * the row's grammar rejections (only for an arm whose rows have any)
# depth: the context while the reply is written, prompt tokens (processed and cached) plus half the generated ones.
# per_reply is time a reply costs whatever its number of steps; per_step is a step at depth 0.
def solve(rows):
    """Least squares of y on the columns of rows (y, x0, x1, …). None when the columns are not independent."""
    n = len(rows[0]) - 1
    m = [[sum(r[i + 1] * r[j + 1] for r in rows) for j in range(n)] + [sum(r[i + 1] * r[0] for r in rows)]
         for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-9:
            return None
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_rows(l, k):
    s, r = S[l][k], loads[l][k]
    d = ((r.get('prompt_n') or 0) + (r.get('cache_n') or 0) + s[0] / 2) / 1000
    return (s[3], 1.0, float(s[4]), s[4] * d, float(r.get('grammar_rejected_n') or 0)) if s[4] > 0 else None


FR = {l: {k: fit_rows(l, k) for k in keys} for l in labels}
depth_mean = ratio(sum(FR[l][k][3] for l in labels for k in keys if FR[l][k]),
                   sum(FR[l][k][2] for l in labels for k in keys if FR[l][k]))  # per step, in 1,000 tokens


def fit(ls, ks, rej=False):
    """(per reply, per step at depth 0, per step per 1,000 tokens, per step at the mean depth[, per grammar
    rejection]) of the loads ls."""
    rows = [FR[l][k] if rej else FR[l][k][:4] for l in ls for k in ks if FR[l][k]]
    c = solve(rows) if rows else None
    return None if c is None else (c[0], c[1], c[2], c[1] + c[2] * depth_mean) + ((c[3],) if rej else ())


FIT_NAMES = ('ms a reply', 'ms a step at depth 0', 'ms a step per 1,000 tokens of context',
             'ms a step at the mean depth')
FIT_MODELS = (('steps and depth', False), ('steps, depth and grammar rejections', True))
has_rej = {a: any(FR[l][k] and FR[l][k][4] for l in arm_loads[a] for k in keys) for a in arms}
fits = []
nboot = min(A.boot, 2000)
for model, rej in FIT_MODELS:
    names = FIT_NAMES + (('ms a grammar rejection',) if rej else ())
    for a in arms:
        if rej and not has_rej[a]:
            continue
        point = fit(arm_loads[a], keys, rej)
        if point is None:
            continue
        boots = [b for b in (fit(arm_loads[a], ks, rej) for ks in samples_req[:nboot]) if b]
        ctl = None
        if a != A.control:
            # a control without rejections is fitted without the term; the difference covers the shared terms
            crej = rej and has_rej[A.control]
            cp = fit(arm_loads[A.control], keys, crej)
            cb = [(fit(arm_loads[a], ks, rej), fit(arm_loads[A.control], ks, crej)) for ks in samples_req[:nboot]]
            ctl = (cp, [tuple(x - y for x, y in zip(p, q)) for p, q in cb if p and q])
        for i, name in enumerate(names):
            v = sorted(b[i] for b in boots)
            row = dict(model=model, arm=a, name=name, value=point[i], lo=v[int(0.025 * len(v))],
                       hi=v[int(0.975 * len(v)) - 1], rows=sum(FR[l][k] is not None for l in arm_loads[a] for k in keys))
            if ctl and i < len(ctl[0]):
                d = sorted(x[i] for x in ctl[1])
                row.update(delta=point[i] - ctl[0][i], dlo=d[int(0.025 * len(d))], dhi=d[int(0.975 * len(d)) - 1])
            fits.append(row)
with (OUT / 'fit.tsv').open('w') as f:
    f.write('model\tarm\trows\tterm\tvalue\tboot_lo\tboot_hi\tminus_control\tminus_control_lo\tminus_control_hi\n')
    for r in fits:
        f.write(f"{r['model']}\t{r['arm']}\t{r['rows']}\t{r['name']}\t{fmt(r['value'], 2)}\t{fmt(r['lo'], 2)}\t"
                f"{fmt(r['hi'], 2)}\t{fmt(r.get('delta'), 2, True)}\t{fmt(r.get('dlo'), 2, True)}\t"
                f"{fmt(r.get('dhi'), 2, True)}\n")

# ---------------------------------------------------------------- same_reply.tsv
same = []
for i, la in enumerate(labels):
    for lb in labels[i + 1:]:
        aa, ab = arm_of(la), arm_of(lb)
        kind = 'one arm, two loads' if aa == ab else 'arm against the control' if A.control in (aa, ab) else 'two arms'
        same.append(dict(kind=kind, a=la, b=lb, n=len(keys),
                         same=sum(loads[la][k].get('reply_md5') == loads[lb][k].get('reply_md5') for k in keys),
                         counters=sum(S[la][k][:3] == S[lb][k][:3] for k in keys)))
with (OUT / 'same_reply.tsv').open('w') as f:
    f.write('kind\tload_a\tload_b\trequests\tsame_reply\tsame_counters\n')
    for r in same:
        f.write(f"{r['kind']}\t{r['a']}\t{r['b']}\t{r['n']}\t{r['same']}\t{r['counters']}\n")

# ---------------------------------------------------------------- the server logs
SAMPLING = re.compile(r'sampling =\s+(\d+) tokens, (\d+) drawn on the device, (\d+) while the grammar applied, '
                      r'(\d+) rejected \((\d+) after a device draw\), (\d+) forced')
ACCEPT = re.compile(r'draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)')
LEVEL = re.compile(r'^\d[\d.]* ([A-Z]) ')
SWITCHES = ('--spec-draft-sampling probabilistic', '--spec-draft-p-min 0', 'LLAMA_SPEC_KEEP_DRAWS=1', 'LLAMA_TRACE=1',
            'GGML_CUDA_PROFILE')
logs = []
for l in labels:
    p = A.work / 'rig' / l / f'server-{l}.log'
    if not p.exists():
        continue
    s = dict(load=l, lines=0, sampling_lines=0, sampled=0, device=0, grammar=0, rejected=0, rejected_device=0, forced=0,
             accept_lines=0, log_accepted=0, log_drafted=0, ckpt_restores=0, error_lines=0, cuda_errors=0,
             watchdog_lines=0, flags='', warmup_generated='', warmup_drafted='', warmup_accepted='')
    for line in p.open(errors='replace'):
        s['lines'] += 1
        if s['lines'] == 1:
            s['flags'] = ' '.join(w for w in SWITCHES if w in line)
        m = SAMPLING.search(line)
        if m:
            s['sampling_lines'] += 1
            for name, x in zip(('sampled', 'device', 'grammar', 'rejected', 'rejected_device', 'forced'), m.groups()):
                s[name] += int(x)
        m = ACCEPT.search(line)
        if m:
            s['accept_lines'] += 1
            s['log_accepted'] += int(m.group(1))
            s['log_drafted'] += int(m.group(2))
        s['ckpt_restores'] += 'restored context checkpoint' in line
        m = LEVEL.match(line)
        s['error_lines'] += bool(m and m.group(1) == 'E')
        s['cuda_errors'] += 'CUDA error' in line
        s['watchdog_lines'] += line.startswith('watchdog:')
    c = pooled(l, keys)
    rows = [loads[l][k] for k in keys]
    s.update(generated=c[0], drafted=c[1], accepted=c[2],
             timings_device=sum(r.get('sampled_on_device') or 0 for r in rows),
             timings_grammar=sum(r.get('grammar_n') or 0 for r in rows),
             timings_rejected=sum(r.get('grammar_rejected_n') or 0 for r in rows))
    p = A.work / f'warmup-{l}.json'  # the untimed request before the replay: in the log's sums, in no replay row
    if p.exists():
        t = (json.loads(p.read_text()).get('timings') or {})
        s.update(warmup_generated=t.get('predicted_n', 0), warmup_drafted=t.get('draft_n', 0),
                 warmup_accepted=t.get('draft_n_accepted', 0))
    logs.append(s)
LOG_COLS = ('load', 'flags', 'generated', 'sampling_lines', 'sampled', 'device', 'grammar', 'rejected',
            'rejected_device', 'forced', 'timings_device', 'timings_grammar', 'timings_rejected', 'accept_lines',
            'log_drafted', 'log_accepted', 'drafted', 'accepted', 'warmup_generated', 'warmup_drafted',
            'warmup_accepted', 'ckpt_restores', 'error_lines', 'cuda_errors', 'watchdog_lines')
with (OUT / 'log_check.tsv').open('w') as f:
    f.write('\t'.join(LOG_COLS) + '\n')
    for s in logs:
        f.write('\t'.join(str(s[c]) for c in LOG_COLS) + '\n')


# ---------------------------------------------------------------- conditions.tsv
def first_number(text):
    m = re.match(r'\s*(-?\d+(?:\.\d+)?)', text)
    return float(m.group(1)) if m else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float('nan')


cond = []
for l in labels:
    rows = [loads[l][k] for k in keys]
    t0 = min((r.get('t_wall') or 0) - (r.get('t_end') or 0) for r in rows)
    t1 = max(r.get('t_wall') or 0 for r in rows)
    c = dict(load=l, start=datetime.fromtimestamp(t0).strftime('%H:%M:%S'),
             end=datetime.fromtimestamp(t1).strftime('%H:%M:%S'), minutes=(t1 - t0) / 60)
    d = A.work / 'rig' / l
    # gpu.csv: timestamp, index, temperature.gpu, power.draw, clocks.sm, memory.used, utilization.gpu, every 10 s; the
    # rig's local time, read here as this machine's local time
    per_card = defaultdict(lambda: defaultdict(list))
    p = d / 'gpu.csv'
    if p.exists():
        for line in p.open(errors='replace'):
            x = [y.strip() for y in line.split(',')]
            if len(x) < 7 or not x[1].isdigit():
                continue
            try:
                t = datetime.strptime(x[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            except ValueError:
                continue
            if t0 <= t <= t1:
                for name, y in zip(('temp', 'power', 'clock', 'mem', 'util'), x[2:7]):
                    per_card[int(x[1])][name].append(first_number(y))
    if per_card:
        temps = {i: mean(v['temp']) for i, v in per_card.items()}
        c.update(cards=len(per_card), samples=len(next(iter(per_card.values()))['temp']), temp=mean(temps.values()),
                 temp_max=max(temps.values()), clock=mean(mean(v['clock']) for v in per_card.values()),
                 clock_min=min(min(y for y in v['clock'] if y is not None) for v in per_card.values()),
                 power=mean(mean(v['power']) for v in per_card.values()),
                 util=mean(mean(v['util']) for v in per_card.values()),
                 mem=' '.join(f"{max(y for y in per_card[i]['mem'] if y is not None):.0f}" for i in sorted(per_card)))
    # meminfo.log: "<epoch> <MemAvailable kB> <swap used kB>" every 2 s
    p = d / 'meminfo.log'
    if p.exists():
        m = [(int(x[1]), int(x[2])) for x in (line.split() for line in p.open(errors='replace'))
             if len(x) == 3 and x[0].isdigit() and t0 <= int(x[0]) <= t1]
        if m:
            c.update(avail_min=min(a for a, _ in m) / 1024, swap_start=m[0][1] / 1024, swap_end=m[-1][1] / 1024,
                     swap_max=max(s for _, s in m) / 1024)
    # vmstat.log (vmstat -t 10): si and so are kB a second over the 10 s before the line's time
    p = d / 'vmstat.log'
    if p.exists():
        si = so = 0
        for line in p.open(errors='replace'):
            x = line.split()
            if len(x) < 20 or not x[0].isdigit():
                continue
            try:
                t = datetime.strptime(' '.join(x[-2:]), '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                continue
            if t0 + 10 <= t <= t1:
                si, so = si + 10 * int(x[6]), so + 10 * int(x[7])
        c.update(swapped_in=si / 1024, swapped_out=so / 1024)
    p = d / f'ram_watchdog-{l}.tsv'
    if p.exists():
        c['kills'] = sum(line.startswith('#') and 'KILL' in line for line in p.open(errors='replace'))
    cond.append(c)
COND_COLS = (('load', None), ('start', None), ('end', None), ('minutes', 1), ('cards', None), ('samples', None),
             ('temp', 1), ('temp_max', 1), ('clock', 0), ('clock_min', 0), ('power', 1), ('util', 1), ('mem', None),
             ('avail_min', 0), ('swap_start', 0), ('swap_end', 0), ('swap_max', 0), ('swapped_out', 1),
             ('swapped_in', 1), ('kills', None))
with (OUT / 'conditions.tsv').open('w') as f:
    f.write('load\treplay_start\treplay_end\treplay_minutes\tcards\tsamples_per_card\tmean_temperature_c\t'
            'hottest_card_mean_c\tmean_sm_clock_mhz\tlowest_sm_clock_mhz\tmean_power_w\tmean_utilization_pct\t'
            'memory_used_mib_per_card\tlowest_mem_available_mib\tswap_used_at_start_mib\tswap_used_at_end_mib\t'
            'highest_swap_used_mib\tswapped_out_during_mib\tswapped_in_during_mib\twatchdog_kills\n')
    for c in cond:
        f.write('\t'.join('' if c.get(k) is None else fmt(c[k], nd) if nd is not None else str(c[k])
                          for k, nd in COND_COLS) + '\n')

# ---------------------------------------------------------------- summary.md
md = [f'# Probabilistic verification on the rig: time ({len(keys)} requests in {len(traces)} sessions, '
      f'{len(labels)} loads)', '']
if any(missing.values()):
    md += ['Requests missing from a load (left out of every table): ' +
           ', '.join(f'{l} {n}' for l, n in missing.items() if n), '']
md += ['## Per load', '',
       '| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step | Generation, s | '
       'Ms a step | Generation tok/s | Prompt tokens processed | Prompt tok/s | Replies at the limit | '
       'Rejected by the grammar |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in load_rows:
    s = r['s']
    md.append(f"| {r['arm']} | {r['load']} | {s[0]:,} | {s[2]:,} of {s[1]:,} | {fmt(ACC(s), 4)} | {s[0] - s[2]:,} | "
              f"{fmt(TPS(s), 3)} | {s[3] / 1000:,.1f} | {fmt(MS_STEP(s), 2)} | {fmt(TOKS(s), 2)} | {s[5]:,} | "
              f"{fmt(PTOKS(s), 2)} | {r['at_limit']} | {r['rejected']} |")
for kind, title in (('arm against the control', f'## Each arm against `{A.control}`'),
                    ('second load against the first', '## One arm, its second load against its first')):
    md += ['', title, '']
    arm_kind = kind == 'arm against the control'
    md.append('| Metric | Better | A | B | A, each load | B, each load | A − B | % | ' +
              ('Spread of A | Spread of B | Class (B\'s spread) | Class (the larger spread) | ' if arm_kind else '') +
              '95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |')
    md.append('|---|---|---|---|---|---|---:|---:|' + ('---:|---:|---|---|' if arm_kind else '') + '---|---|---:|---|---|')
    for r in compare:
        if r['kind'] != kind:
            continue
        nd = r['nd']
        cells = [r['metric'], r['better'], r['a'], r['b'], ', '.join(fmt(v, nd) for v in r['a_loads']),
                 ', '.join(fmt(v, nd) for v in r['b_loads']), fmt(r['delta'], nd, True), fmt(r['percent'], 2, True)]
        if arm_kind:
            cells += [fmt(r['spread_a'], nd), fmt(r['spread_b'], nd), r['class_control'], r['class_larger']]
        cells += [f"{fmt(r['lo'], nd, True)} to {fmt(r['hi'], nd, True)}",
                  f"{fmt(r['tlo'], nd, True)} to {fmt(r['thi'], nd, True)}", fmt(r['req_mean'], nd, True),
                  f"{fmt(r['rlo'], nd, True)} to {fmt(r['rhi'], nd, True)}",
                  f"{r['req_up']} / {r['req_down']} / {r['req_same']}"]
        md.append('| ' + ' | '.join(cells) + ' |')
md += ['', '## By reply length', '',
       'The requests in two halves by their reply limit (the recorded reply\'s length, the same in every load); pooled '
       'within a half.', '',
       f'| Half | Reply limit, tokens | Requests | Metric | Arm | Its loads | `{A.control}`, each load | Difference | % |',
       '|---|---|---:|---|---|---|---|---:|---:|']
for name, ks in halves:
    if not ks:
        continue
    for metric, fn, nd, _ in METRICS[:3]:
        vb = [fn(pooled(l, ks)) for l in arm_loads[A.control]]
        for a in arms:
            if a == A.control:
                continue
            va = [fn(pooled(l, ks)) for l in arm_loads[a]]
            delta = sum(va) / len(va) - sum(vb) / len(vb)
            md.append(f"| {name} | {limit(ks[0]):,} to {limit(ks[-1]):,} | {len(ks)} | {metric} | {a} | "
                      f"{', '.join(fmt(v, nd) for v in va)} | {', '.join(fmt(v, nd) for v in vb)} | "
                      f"{fmt(delta, nd, True)} | {fmt(100 * delta / (sum(vb) / len(vb)), 2, True)} |")
md += ['', '## A reply\'s time as a cost per reply and a cost per step', '',
       'Per arm, least squares over every request in each of its loads: generation ms = a reply\'s cost + timed steps × '
       '(a step at depth 0 + its growth per 1,000 tokens of context × depth), and a second model with a cost per '
       'grammar rejection added (for an arm whose rows have any; a control without rejections keeps the first model, '
       'and the difference covers the shared terms). Depth is the prompt\'s tokens plus half the generated ones; the '
       f'mean depth over all steps is {1000 * depth_mean:,.0f} tokens. Intervals: 95 % bootstrap over requests, '
       f'{nboot:,} resamples.', '',
       f'| Model | Arm | Rows | Term | Value | 95 % | Against `{A.control}` | 95 % |',
       '|---|---|---:|---|---:|---|---:|---|']
for r in fits:
    md.append(f"| {r['model']} | {r['arm']} | {r['rows']} | {r['name']} | {fmt(r['value'], 2)} | {fmt(r['lo'], 2)} to "
              f"{fmt(r['hi'], 2)} | {fmt(r.get('delta'), 2, True)} | " +
              (f"{fmt(r['dlo'], 2, True)} to {fmt(r['dhi'], 2, True)} |" if 'delta' in r else '|'))
md += ['', '## The same reply in two loads', '', '| Loads | Kind | Same reply | Same counters |', '|---|---|---|---|']
for r in same:
    md.append(f"| {r['a']}, {r['b']} | {r['kind']} | {r['same']} of {r['n']} | {r['counters']} of {r['n']} |")
if logs:
    md += ['', '## The server logs', '',
           '| Load | Switches on its line | Generated | Requests with a grammar | Their tokens: sampled, drawn on the '
           'device | Rejected (after a device draw) | Accepted of drafted: the log\'s request lines; the replay\'s '
           'timings; the warm-up request | Checkpoint restores at a prompt | Error lines | Watchdog lines |',
           '|---|---|---:|---:|---|---|---|---:|---:|---:|']
    for s in logs:
        md.append(f"| {s['load']} | {s['flags']} | {s['generated']:,} | {s['sampling_lines']} | {s['sampled']:,}, "
                  f"{s['device']:,} | {s['rejected']} ({s['rejected_device']}) | {s['log_accepted']:,} of "
                  f"{s['log_drafted']:,}; {s['accepted']:,} of {s['drafted']:,}; {s['warmup_accepted']} of "
                  f"{s['warmup_drafted']} | {s['ckpt_restores']} | {s['error_lines'] + s['cuda_errors']} | "
                  f"{s['watchdog_lines']} |")
md += ['', '## The loads\' conditions', '',
       '| Load | Replay | Minutes | Cards: mean °C | Hottest card, mean °C | Mean SM clock, MHz | Lowest SM clock, MHz | '
       'Mean power, W | Mean utilization, % | Memory used per card, MiB | Lowest MemAvailable, MiB | Swap used at the '
       'replay\'s start, end, highest, MiB | Swapped out, in during the replay, MiB | Watchdog kills |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|']
for c in cond:
    g = lambda k, nd=None: '' if c.get(k) is None else fmt(c[k], nd) if nd is not None else str(c[k])  # noqa: E731
    md.append(f"| {c['load']} | {c['start']} to {c['end']} | {g('minutes', 1)} | {g('temp', 1)} | {g('temp_max', 1)} | "
              f"{g('clock', 0)} | {g('clock_min', 0)} | {g('power', 1)} | {g('util', 1)} | {g('mem')} | "
              f"{g('avail_min', 0)} | {g('swap_start', 0)}, {g('swap_end', 0)}, {g('swap_max', 0)} | "
              f"{g('swapped_out', 1)}, {g('swapped_in', 1)} | {g('kills')} |")
(OUT / 'summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
