#!/usr/bin/env python3
"""replies.py: every arm's greedy reply per depth in a bench/deep_lane run, against the replies of a reference run.

  python3 -I replies.py --out <this run's folder> --ref <reference run's requests.csv> [--ref-arms v4,v4s]
      [--ref-jsonl <reference run's <load>.jsonl files>...] [--md replies.md] [--tsv replies.tsv]

For each arm and depth the reply is the decode pass 1's token ids (its md5, `ids_md5` in requests.csv). Against each
reference arm it is "same" (the same md5) or "differs at token N": the first N - 1 tokens are the same and token N is
another. The reference reply's ids come from this run: from any reply here with its md5 (deep_client.py keeps each
reply's ids in <load>.jsonl). When no reply here has it, the first 160 characters of the two replies are compared
instead (--ref-jsonl), and the line gives the first differing character. Checks: every arm wrote the same reply on pass
2 as on pass 1; every prompt is the reference run's (prompt_md5); every prime read the reference run's chunk on the same
cached tokens (cache_n, prompt_n). Standard library only.
"""
import argparse
import csv
import glob
import hashlib
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--ref-arms', default='v4,v4s')
ap.add_argument('--ref-jsonl', nargs='*', default=[])
ap.add_argument('--md', default='')
ap.add_argument('--tsv', default='')
A = ap.parse_args()


def md5(obj):
    return hashlib.md5((obj if isinstance(obj, str) else json.dumps(obj)).encode()).hexdigest()


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


run = [r for r in rows(os.path.join(A.out, 'requests.csv'))]
ref = rows(A.ref)
ref_arms = A.ref_arms.split(',')
arms = list(dict.fromkeys(r['arm'] for r in run))
depths = sorted({int(r['depth']) for r in run if r['kind'] == 'decode'})

ids_by_md5, head_by_md5 = {}, {}
for path in sorted(glob.glob(os.path.join(A.out, '*.jsonl'))):
    for line in open(path):
        r = json.loads(line)
        if r.get('kind') != 'decode':
            continue
        if r.get('ids') and md5(r['ids']) == r['ids_md5']:
            ids_by_md5[r['ids_md5']] = r['ids']
        head_by_md5.setdefault(r['ids_md5'], r.get('content_head', ''))
for path in A.ref_jsonl:
    for line in open(path):
        r = json.loads(line)
        if r.get('kind') == 'decode':
            head_by_md5.setdefault(r['ids_md5'], r.get('content_head', ''))


def reply(rs, arm, depth, pas):
    got = [r for r in rs if r['arm'] == arm and r['kind'] == 'decode' and int(r['depth']) == depth and r['pass'] == pas]
    return got


def ref_reply(arm, depth):
    md5s = {r['ids_md5'] for r in reply(ref, arm, depth, '1') + reply(ref, arm, depth, '2')}
    return md5s.pop() if len(md5s) == 1 else None, len(md5s)


def against(mine, theirs):
    if theirs is None:
        return 'no single reference reply', ''
    if mine == theirs:
        return 'same', 0
    a, b = ids_by_md5.get(mine), ids_by_md5.get(theirs)
    if a and b:
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return f'differs at token {n + 1}', n + 1
    ha, hb = head_by_md5.get(mine, ''), head_by_md5.get(theirs, '')
    if ha and hb:
        n = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
        return ('differs after the first 160 characters' if n is None else
                f'differs at character {n + 1} (no ids of the reference reply)'), ''
    return 'differs (no ids or text to place it)', ''


out = []
for depth in depths:
    for arm in arms:
        p1, p2 = reply(run, arm, depth, '1'), reply(run, arm, depth, '2')
        if not p1:
            out.append({'depth': depth, 'arm': arm, 'load': '', 'ids_md5': 'no reply', 'pass2': ''})
            continue
        m = p1[0]['ids_md5']
        line = {'depth': depth, 'arm': arm, 'load': p1[0]['load'], 'ids_md5': m,
                'pass2': 'same' if p2 and p2[0]['ids_md5'] == m else 'other'}
        for ra in ref_arms:
            theirs, n = ref_reply(ra, depth)
            line[f'vs_{ra}'], line[f'token_{ra}'] = against(m, theirs)
        out.append(line)

checks = []
for depth in depths:
    for kind in ('prime', 'decode'):
        mine = {r['prompt_md5'] for r in run if r['kind'] == kind and int(r['depth']) == depth}
        theirs = {r['prompt_md5'] for r in ref if r['kind'] == kind and int(r['depth']) == depth}
        if mine:
            same = "the reference run's" if mine == theirs else "OTHER than the reference run's"
            checks.append(f'{kind} prompt at {depth}: {same} ({len(mine)} prompt md5 here)')
    mine = {(r['cache_n'], r['prompt_n']) for r in run if r['kind'] == 'prime' and int(r['depth']) == depth}
    theirs = {(r['cache_n'], r['prompt_n']) for r in ref if r['kind'] == 'prime' and int(r['depth']) == depth
              and r['arm'] in ref_arms}
    if mine:
        what = ', '.join(f'{c} cached, {p} read' for c, p in sorted(mine))
        checks.append(f"prime at {depth}: {what} in every load here; "
                      f"{'the same as' if mine == theirs else 'NOT the same as'} the reference run's "
                      f"({', '.join(f'{c} cached, {p} read' for c, p in sorted(theirs))})")
for ra in ref_arms:
    for depth in depths:
        theirs, n = ref_reply(ra, depth)
        if theirs is None:
            checks.append(f'reference arm {ra} at {depth}: {n} different replies')

cols = ['depth', 'arm', 'load', 'ids_md5', 'pass2'] + [f'{k}_{ra}' for ra in ref_arms for k in ('vs', 'token')]
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for line in out:
            f.write('\t'.join(str(line.get(c, '')) for c in cols) + '\n')
md = ['| Depth (tokens) | Arm | Reply (md5 of the token ids) | Pass 2 | ' +
      ' | '.join(f'Against the reference `{ra}`' for ra in ref_arms) + ' |',
      '|---:|---|---|---|' + '---|' * len(ref_arms)]
for line in out:
    md.append(f"| {line['depth']:,} | `{line['arm']}` | `{line['ids_md5']}` | {line['pass2']} | " +
              ' | '.join(str(line.get(f'vs_{ra}', '')) for ra in ref_arms) + ' |')
md += ['', 'Checks:', ''] + [f'- {c}' for c in checks]
text = '\n'.join(md) + '\n'
if A.md:
    with open(A.md, 'w') as f:
        f.write(text)
print(text)
