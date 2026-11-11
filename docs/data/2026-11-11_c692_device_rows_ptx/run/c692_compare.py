#!/usr/bin/env python3
"""c692_compare.py: C692's tables (registry Q1225) from run_c692.sh's work folder. No time is read.

  python3 -I c692_compare.py <work folder>

Reads loads.tsv, requests_<label>.jsonl (the replies), dump_<label>.jsonl (LLAMA_SPEC_DUMP: one line per verify step)
and server-<label>.log (the load lines). A load's dump lines are grouped by task in the order they appear and matched
to its requests in order (the warm-up first); each group's n_prompt must equal the request's prompt length.
Writes <work>/tables/:
  replies.tsv        one row per load and request: the prompt's ids (md5), the reply's ids (md5), the draft counters,
                     the verify steps in the dump and how many rejected a drafted token
  pairs.tsv          every two loads, every request: the same reply ids or the first differing token; the same steps
                     (the position, the last token, the drafted tokens, the target's tokens, the drafter's candidate
                     ids per position) or the first differing step and its first differing field; over the steps
                     before it, the largest difference of the drafter's candidate probabilities (rounded to 1e-5 in
                     the dump) and of the target's top probabilities
  pairs_summary.tsv  every two loads: requests with the same ids, with the same steps, and those two largest differences
  load_lines.tsv     each load's lines on the devices, the h_nextn buffer, the draft's switches and its row cache, and
                     every warning or error line
"""
import itertools
import json
import os
import re
import sys

W = sys.argv[1]
T = os.path.join(W, 'tables')
os.makedirs(T, exist_ok=True)
STEP_KEYS = ('pos', 'last', 'draft', 'sampled', 'cand')


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


loads = []
with open(os.path.join(W, 'loads.tsv')) as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if os.path.exists(os.path.join(W, f'load-{p[0]}.done')):
            loads.append({'label': p[0], 'build': p[1], 'arm': p[2]})

data = {}
notes = []
for ld in loads:
    lab = ld['label']
    reqs = read_jsonl(os.path.join(W, f'requests_{lab}.jsonl'))
    dump = read_jsonl(os.path.join(W, f'dump_{lab}.jsonl'))
    groups, order = {}, []
    for s in dump:
        t = s.get('task')
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(s)
    if len(order) != len(reqs):
        notes.append(f'{lab}: {len(order)} tasks in the dump, {len(reqs)} requests')
    per = {}
    for i, r in enumerate(reqs):
        steps = groups[order[i]] if i < len(order) else []
        if steps and steps[0].get('n_prompt') not in (None, r['prompt_tokens']):
            notes.append(f"{lab} {r['prompt']} {r['rep']}: the dump's n_prompt {steps[0].get('n_prompt')} is not the "
                         f"prompt's {r['prompt_tokens']} tokens")
        per[(r['prompt'], r['rep'])] = {'req': r, 'steps': steps}
    data[lab] = per

with open(os.path.join(T, 'replies.tsv'), 'w') as f:
    f.write('label\tbuild\tarm\tprompt\trep\tprompt_tokens\tprompt_md5\tn_gen\tids_md5\tdraft_n\tdraft_accepted\t'
            'steps\trejecting_steps\n')
    for ld in loads:
        for (name, rep), x in data[ld['label']].items():
            r, st = x['req'], x['steps']
            f.write(f"{ld['label']}\t{ld['build']}\t{ld['arm']}\t{name}\t{rep}\t{r['prompt_tokens']}\t{r['prompt_md5']}\t"
                    f"{r['n_gen']}\t{r['ids_md5']}\t{r['draft_n']}\t{r['draft_accepted']}\t{len(st)}\t"
                    f"{sum(1 for s in st if s.get('verify') == 'reject')}\n")


def maxdiff(a, b):
    """the largest |a - b| over two nested lists of numbers of the same shape, None if the shapes differ"""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        m = 0.0
        for x, y in zip(a, b):
            d = maxdiff(x, y)
            if d is None:
                return None
            m = max(m, d)
        return m
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    return 0.0 if a == b else None


def compare(xa, xb):
    ra, rb = xa['req'], xb['req']
    ia, ib = ra['ids'], rb['ids']
    n = min(len(ia), len(ib))
    tok = next((i for i in range(n) if ia[i] != ib[i]), n if len(ia) == len(ib) else n)
    same_ids = ia == ib
    sa, sb = xa['steps'], xb['steps']
    m = min(len(sa), len(sb))
    step, field = None, ''
    for k in range(m):
        diff = [key for key in STEP_KEYS if sa[k].get(key) != sb[k].get(key)]
        if diff:
            step, field = k, ','.join(diff)
            break
    if step is None and len(sa) != len(sb):
        step, field = m, 'count'
    upto = m if step is None else step
    dp = dt = 0.0
    for k in range(upto):
        d = maxdiff(sa[k].get('p', []), sb[k].get('p', []))
        dp = max(dp, d) if d is not None else dp
        if sa[k].get('tgt_ids') == sb[k].get('tgt_ids'):
            d = maxdiff(sa[k].get('tgt_p', []), sb[k].get('tgt_p', []))
            dt = max(dt, d) if d is not None else dt
    return {'same_ids': same_ids, 'first_diff_token': '' if same_ids else tok, 'steps_a': len(sa), 'steps_b': len(sb),
            'same_steps': step is None and len(sa) > 0, 'first_diff_step': '' if step is None else step,
            'field': field, 'max_dp_draft': dp, 'max_dp_target': dt,
            'pos_at_diff': '' if step is None or step >= m else sa[step].get('pos')}


rows, summ = [], []
for la, lb in itertools.combinations([ld['label'] for ld in loads], 2):
    keys = [k for k in data[la] if k in data[lb]]
    n_ids = n_steps = 0
    mdp = mdt = 0.0
    for k in keys:
        c = compare(data[la][k], data[lb][k])
        rows.append((la, lb, k[0], k[1], c))
        n_ids += c['same_ids']
        n_steps += c['same_steps']
        mdp, mdt = max(mdp, c['max_dp_draft']), max(mdt, c['max_dp_target'])
    summ.append((la, lb, len(keys), n_ids, n_steps, mdp, mdt))
# within each load: each prompt's repeats against its first
for ld in loads:
    lab = ld['label']
    for (name, rep) in data[lab]:
        if rep > 1 and (name, 1) in data[lab]:
            c = compare(data[lab][(name, 1)], data[lab][(name, rep)])
            rows.append((f'{lab}#1', f'{lab}#{rep}', name, rep, c))

with open(os.path.join(T, 'pairs.tsv'), 'w') as f:
    f.write('a\tb\tprompt\trep\tsame_ids\tfirst_diff_token\tsteps_a\tsteps_b\tsame_steps\tfirst_diff_step\tfield\t'
            'pos_at_diff\tmax_dp_draft\tmax_dp_target\n')
    for la, lb, name, rep, c in rows:
        f.write(f"{la}\t{lb}\t{name}\t{rep}\t{int(c['same_ids'])}\t{c['first_diff_token']}\t{c['steps_a']}\t"
                f"{c['steps_b']}\t{int(c['same_steps'])}\t{c['first_diff_step']}\t{c['field']}\t{c['pos_at_diff']}\t"
                f"{c['max_dp_draft']:.5f}\t{c['max_dp_target']:.5f}\n")
with open(os.path.join(T, 'pairs_summary.tsv'), 'w') as f:
    f.write('a\tb\trequests\tsame_ids\tsame_steps\tmax_dp_draft\tmax_dp_target\n')
    for la, lb, n, ni, ns, dp, dt in summ:
        f.write(f'{la}\t{lb}\t{n}\t{ni}\t{ns}\t{dp:.5f}\t{dt:.5f}\n')

PAT = re.compile(r'h_nextn|target h rows|draft hidden rows|hidden-row carry|LLAMA_DRAFT_H_CARRY|embedding row cache|'
                 r'LLAMA_DRAFT_EMBD_CACHE|shortlist|using device|found \d+ CUDA|Device \d+:|model buffer size|'
                 r'KV buffer size|RS buffer size|compute buffer size|output buffer size|offloaded|LLAMA_SPEC_DUMP|'
                 r'(?i:\bwarn(ing)?\b|\berror\b|\bfailed\b|abort)')
with open(os.path.join(T, 'load_lines.tsv'), 'w') as f:
    f.write('label\tn\tline\n')
    for ld in loads:
        path = os.path.join(W, f"server-{ld['label']}.log")
        n = 0
        with open(path, errors='replace') as g:
            for line in g:
                if line.startswith('=== '):
                    continue
                if PAT.search(line):
                    s = re.sub(r'\x1b\[[0-9;]*m', '', line.rstrip('\n'))
                    s = re.sub(r'^[0-9.]+ [A-Z] ', '', s)
                    n += 1
                    if n <= 400:
                        f.write(f"{ld['label']}\t{n}\t{s}\n")

for x in notes:
    print('note:', x)
for la, lb, n, ni, ns, dp, dt in summ:
    print(f'{la} vs {lb}: {ni}/{n} requests the same ids, {ns}/{n} the same steps; largest difference of the drafter\'s '
          f'probabilities {dp:.5f}, of the target\'s {dt:.5f}')
for la, lb, name, rep, c in rows:
    if '#' in la and not (c['same_ids'] and c['same_steps']):
        print(f"repeat differs: {la} {lb} {name}: first token {c['first_diff_token']}, first step {c['first_diff_step']}")
