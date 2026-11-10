#!/usr/bin/env python3
"""exact_steps.py: the draft line's verify steps around each token that is not the plain line's (registry Q1175).

  python3 exact_steps.py --dump spec_dump_<arm>.jsonl --replies replies_<arm>.json --plain replies_nodraft.json \
      --out <folder> [--same-as replies_draft.json --requests requests_nodraft.jsonl --tf-reply draft]

--dump is the file the server writes with LLAMA_SPEC_DUMP: one JSON line per verify step (the position of the last
token before the draft, the drafted tokens, the tokens the step yielded: the accepted draft tokens and one token of the
main model's own). A step is partly accepted when it keeps fewer tokens than were drafted; the main model's state then
goes back to the last kept token before the next step.

Writes steps_<arm>.tsv, per depth for the first request at that depth: steps, drafted and accepted tokens, steps by
outcome, and for the first token that differs from the plain line's reply: the step that wrote it, whether that token is
an accepted draft token or the step's own last token, and whether the step before it was partly accepted.
With --same-as (another load of the same arm whose replies were teacher-forced on the plain line in --requests, as
reply --tf-reply): when this load wrote the same tokens, also off_tokens_<arm>.tsv: the same two facts for every token
of the reply that is not the plain line's first choice, beside their rates over all the reply's tokens.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True)
ap.add_argument('--replies', required=True)
ap.add_argument('--plain', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--same-as', default='')
ap.add_argument('--requests', default='')
ap.add_argument('--tf-reply', default='draft')
A = ap.parse_args()


def load(path):
    with open(path) as f:
        return json.load(f)


mine = load(A.replies)
arm = mine['arm']
replies = {int(d): r for d, r in mine['replies'].items()}
plain = {int(d): r for d, r in load(A.plain)['replies'].items()}
same = {int(d): r for d, r in load(A.same_as)['replies'].items()} if A.same_as else {}

# the steps of each task; a depth's first task with that many prompt tokens is its first request
tasks = {}
with open(A.dump) as f:
    for raw in f:
        s = json.loads(raw)
        tasks.setdefault((s['n_prompt'], s['task']), []).append(s)

# end-of-text tokens (listed above the token a greedy line wrote) are left out of a first choice, as in exact_table.py
banned = set()
for rep in list(plain.values()) + list(replies.values()) + list(same.values()):
    for pos in rep['probs']:
        for tid, _, _ in pos['top']:
            if tid == pos['id']:
                break
            banned.add(tid)
off = {}  # depth -> tokens of the teacher-forced reply that are not the plain line's first choice
if A.requests:
    with open(A.requests) as f:
        for raw in f:
            r = json.loads(raw)
            if r['kind'] != 'tf' or r['reply'] != A.tf_reply:
                continue
            want = same[r['depth']]['ids'][r['i']]
            first = (next(t[0] for t in r['probs'][0]['top'] if t[0] not in banned) if r['probs']
                     else (r['ids'][0] if r['ids'] else None))
            off.setdefault(r['depth'], {})[r['i']] = first != want

cols = ['depth', 'arm', 'same_tokens_as_the_scored_load', 'steps', 'drafted', 'accepted', 'steps_all_accepted',
        'steps_partly_accepted', 'steps_none_accepted', 'tokens_from_steps', 'first_differing_token', 'written_as',
        'step_drafted', 'step_accepted', 'step_before_drafted', 'step_before_accepted',
        'follows_a_partly_accepted_draft']
off_cols = ['depth', 'arm', 'token', 'written_as', 'step_drafted', 'step_accepted',
            'follows_a_partly_accepted_draft']
rows, off_rows, rates = [], [], {'all': [0, 0, 0], 'off': [0, 0, 0]}  # tokens, own tokens, after a partly accepted step
for d in sorted(replies):
    ids = replies[d]['ids']
    # the request: of the tasks with this prompt length (depth 0 is the tail alone, as long as the warm-up's prompt),
    # the first whose steps wrote this reply's tokens
    steps, where = [], {}  # where: reply token index -> (step number in this request, index within the step's tokens)
    for n_prompt, task in sorted(tasks, key=lambda k: k[1]):
        if n_prompt != (d or len(plain[d].get('tail', [])) or 64):
            continue
        cand = sorted(tasks[(n_prompt, task)], key=lambda s: s['step'])
        found = {}
        for k, s in enumerate(cand):
            for j, tok in enumerate(s['sampled']):
                i = s['pos'] - n_prompt + 1 + j
                if 0 <= i < len(ids) and ids[i] == tok:
                    found[i] = (k, j)
        if len(found) > len(where):
            steps, where = cand, found

    def facts(i):
        if i not in where:
            return {'written_as': 'outside a verify step'}
        k, j = where[i]
        s = steps[k]
        before = steps[k - 1] if k > 0 else None
        partly = before is not None and before['n_acc'] < len(before['draft'])
        return {'written_as': 'an accepted draft token' if j < s['n_acc'] else 'the step\'s own token',
                'step_drafted': len(s['draft']), 'step_accepted': s['n_acc'],
                'step_before_drafted': len(before['draft']) if before else '',
                'step_before_accepted': before['n_acc'] if before else '',
                'follows_a_partly_accepted_draft': 'yes' if partly else 'no'}

    o = plain[d]['ids']
    c = next((i for i, (x, y) in enumerate(zip(ids, o)) if x != y), None)
    is_same = bool(same) and same.get(d, {}).get('ids') == ids
    rows.append({'depth': d, 'arm': arm, 'same_tokens_as_the_scored_load': ('yes' if is_same else 'no') if same else '',
                 'steps': len(steps), 'drafted': sum(len(s['draft']) for s in steps),
                 'accepted': sum(s['n_acc'] for s in steps),
                 'steps_all_accepted': sum(s['n_acc'] == len(s['draft']) for s in steps),
                 'steps_partly_accepted': sum(0 < s['n_acc'] < len(s['draft']) for s in steps),
                 'steps_none_accepted': sum(s['n_acc'] == 0 for s in steps), 'tokens_from_steps': len(where),
                 'first_differing_token': 'none' if c is None else c, **(facts(c) if c is not None else {})})
    if is_same and d in off:
        for i in sorted(off[d]):
            f = facts(i)
            for name in ('all', 'off') if off[d][i] else ('all',):
                rates[name][0] += 1
                rates[name][1] += f['written_as'] == 'the step\'s own token'
                rates[name][2] += f.get('follows_a_partly_accepted_draft') == 'yes'
            if off[d][i]:
                off_rows.append({'depth': d, 'arm': arm, 'token': i, **f})


def write_tsv(name, cols, rows):
    with open(os.path.join(A.out, name), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')


os.makedirs(A.out, exist_ok=True)
write_tsv(f'steps_{arm}.tsv', cols, rows)
print('\t'.join(cols))
for r in rows:
    print('\t'.join(str(r.get(c, '')) for c in cols))
if off_rows or rates['all'][0]:
    write_tsv(f'off_tokens_{arm}.tsv', off_cols, off_rows)
    with open(os.path.join(A.out, f'off_tokens_{arm}_rates.tsv'), 'w') as f:
        f.write('tokens\tcount\tthe_step_own_token\tafter_a_partly_accepted_step\n')
        for name, label in (('all', 'every scored token of the reply'), ('off', 'not the plain line\'s first choice')):
            f.write(f'{label}\t{rates[name][0]}\t{rates[name][1]}\t{rates[name][2]}\n')
            print(f'{label}: {rates[name][0]} tokens, {rates[name][1]} a step\'s own token, '
                  f'{rates[name][2]} after a partly accepted step')
