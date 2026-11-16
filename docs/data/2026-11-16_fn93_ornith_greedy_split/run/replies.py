#!/usr/bin/env python3
"""replies.py: FN93's replies table. Every greedy and sampled reply of the deep-lane runs given, against the control
arms' replies at the same depth and pass, and against the rig's replies of R162 for the same arm.

    python3 -I replies.py --run STEP1_OUT [STEP2_OUT …] --rig R162_OUT --out replies   (writes replies.tsv, replies.md)

--run folders were run with KEEP_REPLY=1, so their <load>.jsonl hold each reply whole (ids, pieces). --rig is R162's
collected run folder, whose <load>.jsonl hold each reply's ids_md5 and its first 160 characters. A comparison reads
"same" or the position of the first token that differs, counted from 1 (token 1 is the first written token):
- between two replies of the --run folders: by token ids;
- against a rig reply: "same" when the ids' md5 is the rig's; else the first character that differs from the rig's
  first 160, turned into the token that holds it through this reply's pieces ("token N"); "after char 160" when the
  first 160 characters agree (the rig kept no more).
The controls: CTRL (default v4, its first load) and STACK (default v4s), from the first --run folder that holds them.
With --private PREFIX also PREFIX.tsv: the tokens' text at each first difference (reply text: keep it private).
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--run', nargs='+', required=True)
ap.add_argument('--rig', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ctrl', default='v4')
ap.add_argument('--stack', default='v4s')
ap.add_argument('--private', default='')
A = ap.parse_args()
KINDS = ('decode', 'sample')


def load(folder):
    recs = []
    for f in sorted(glob.glob(os.path.join(folder, '*.jsonl'))):
        if os.path.basename(f).startswith('warm'):
            continue
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get('kind') in KINDS:
                    r['run'] = os.path.basename(os.path.normpath(folder))
                    recs.append(r)
    return recs


def key(r):
    return (r['kind'], str(r['pass']), int(r['depth']))


def first_token_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def char_to_token(pieces, c):
    n = 0
    for i, p in enumerate(pieces):
        n += len(p)
        if n > c:
            return i + 1
    return len(pieces) + 1


def vs_run(r, ref):
    if ref is None:
        return '-'
    if r['ids_md5'] == ref['ids_md5']:
        return 'same'
    t = first_token_diff(r['ids'], ref['ids'])
    return f'token {t}'


def vs_rig(r, rig):
    if rig is None:
        return '-'
    if r['ids_md5'] == rig['ids_md5']:
        return 'same'
    head, mine = rig['content_head'], r['content']
    for c, (x, y) in enumerate(zip(mine, head)):
        if x != y:
            return f'token {char_to_token(r["pieces"], c)}' if r.get('pieces') else f'char {c + 1}'
    if len(mine) < len(head):
        return f'token {len(r["ids"]) + 1}'
    return f'after char {len(head)}'


runs = [load(f) for f in A.run]
rig = load(A.rig)
first_load = {}
for recs in runs:
    for r in recs:
        first_load.setdefault(r['arm'], r['load'])
ctrl = {key(r): r for recs in runs for r in recs if r['arm'] == A.ctrl and r['load'] == first_load.get(A.ctrl)}
stack = {key(r): r for recs in runs for r in recs if r['arm'] == A.stack and r['load'] == first_load.get(A.stack)}
rig_first = {}
for r in rig:
    rig_first.setdefault(r['arm'], r['load'])
rig_by = {(r['arm'], key(r)): r for r in rig if r['load'] == rig_first[r['arm']]}
rig_loads = {}
for r in rig:
    rig_loads.setdefault((r['arm'], key(r)), set()).add(r['ids_md5'])

cols = ['run', 'arm', 'load', 'kind', 'pass', 'depth', 'gen_n', 'ids_md5', f'vs_{A.ctrl}', f'vs_{A.stack}',
        f'vs_rig_{A.ctrl}', f'vs_rig_{A.stack}', 'vs_rig_own_arm']
rows = []
priv = []
for recs in runs:
    for r in sorted(recs, key=lambda r: (r['load'], int(r['depth']), r['kind'], str(r['pass']))):
        k = key(r)
        own = rig_by.get((r['arm'], k))
        row = {'run': r['run'], 'arm': r['arm'], 'load': r['load'], 'kind': r['kind'], 'pass': r['pass'],
               'depth': r['depth'], 'gen_n': r['gen_n'], 'ids_md5': r['ids_md5'][:8],
               f'vs_{A.ctrl}': vs_run(r, ctrl.get(k)), f'vs_{A.stack}': vs_run(r, stack.get(k)),
               f'vs_rig_{A.ctrl}': vs_rig(r, rig_by.get((A.ctrl, k))),
               f'vs_rig_{A.stack}': vs_rig(r, rig_by.get((A.stack, k))),
               'vs_rig_own_arm': vs_rig(r, own) if own else '-'}
        rows.append(row)
        for name, ref in ((A.ctrl, ctrl.get(k)), (A.stack, stack.get(k))):
            if ref is not None and r['ids_md5'] != ref['ids_md5'] and r.get('pieces') and ref.get('pieces'):
                t = first_token_diff(r['ids'], ref['ids'])
                priv.append([r['run'], r['arm'], r['load'], r['kind'], r['pass'], r['depth'], name, t,
                             json.dumps(''.join(r['pieces'][max(t - 6, 0):t - 1])),
                             json.dumps(r['pieces'][t - 1] if t - 1 < len(r['pieces']) else ''),
                             json.dumps(ref['pieces'][t - 1] if t - 1 < len(ref['pieces']) else '')])

with open(A.out + '.tsv', 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for row in rows:
        f.write('\t'.join(str(row[c]) for c in cols) + '\n')

# the Markdown table: greedy pass 1 per arm and depth (pass 2, the other loads and the sampler pass are in the TSV)
depths = sorted({int(r['depth']) for recs in runs for r in recs})
arms = []
for recs in runs:
    for r in recs:
        if r['arm'] not in arms:
            arms.append(r['arm'])
with open(A.out + '.md', 'w') as f:
    f.write(f'Greedy pass 1 of each arm\'s first load: "same", or the first token that differs (counted from 1), '
            f'against this machine\'s `{A.ctrl}` and `{A.stack}` and the rig\'s (R162\'s first loads).\n\n')
    f.write(f'| Arm | Depth | vs `{A.ctrl}` | vs `{A.stack}` | vs rig `{A.ctrl}` | vs rig `{A.stack}` | ids md5 |\n')
    f.write('|---|---:|---|---|---|---|---|\n')
    for arm in arms:
        for d in depths:
            for row in rows:
                if (row['arm'] == arm and row['load'] == first_load[arm] and row['kind'] == 'decode'
                        and str(row['pass']) == '1' and int(row['depth']) == d):
                    f.write(f"| `{arm}` | {d:,} | {row[f'vs_{A.ctrl}']} | {row[f'vs_{A.stack}']} | "
                            f"{row[f'vs_rig_{A.ctrl}']} | {row[f'vs_rig_{A.stack}']} | `{row['ids_md5']}` |\n")
    # the checks: pass 2 against pass 1, a second load against the first, the rig's loads among themselves
    f.write('\nChecks:\n\n')
    for recs in runs:
        by = {}
        for r in recs:
            by.setdefault((r['arm'], key(r)), set()).add(r['ids_md5'])
        p12 = sum(1 for r in recs if r['kind'] == 'decode' and str(r['pass']) == '2'
                  and next((q for q in recs if q['load'] == r['load'] and q['kind'] == 'decode'
                            and str(q['pass']) == '1' and q['depth'] == r['depth']), {}).get('ids_md5') != r['ids_md5'])
        multi = sum(1 for v in by.values() if len(v) > 1)
        name = os.path.basename(os.path.normpath(recs[0]['run'])) if recs else '?'
        f.write(f'- `{name}`: {len(recs)} replies; greedy pass 2 unlike its pass 1 in {p12}; '
                f'arm, kind, pass and depth with more than one reply over its loads: {multi}\n')
    rmulti = sum(1 for v in rig_loads.values() if len(v) > 1)
    f.write(f'- the rig (R162): arm, kind, pass and depth with more than one reply over its loads: {rmulti}\n')

if A.private:
    with open(A.private + '.tsv', 'w') as f:
        f.write('run\tarm\tload\tkind\tpass\tdepth\tagainst\ttoken\tbefore\tthis\tthat\n')
        for p in priv:
            f.write('\t'.join(str(x) for x in p) + '\n')
print(f'{len(rows)} replies; {A.out}.tsv, {A.out}.md' + (f', {A.private}.tsv' if A.private else ''))
