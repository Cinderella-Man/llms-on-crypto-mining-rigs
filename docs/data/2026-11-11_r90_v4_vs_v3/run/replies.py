#!/usr/bin/env python3
"""replies.py: one deep-lane run's replies compared between loads and arms, per request.

    python3 -I replies.py OUT [--control v3] [--tsv replies.tsv] [--md replies.md]

OUT is a deep-lane results folder (bench/deep_lane: requests.csv). Per kind (decode: the greedy requests at temperature
0, pass 1 and pass 2; sample: the sampler pass's request with its seed), depth and pass, every timed load's reply as
its token ids' md5 (ids_md5, else the text's), shortened to 8 characters, then per arm whether its loads wrote one reply
and whether that reply is the control's. A load whose reply differs from its own arm's other load shows the arm's own
spread; an arm whose loads agree with each other and not with the control's changes the reply.
Standard library only.
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--control', default='v3')
ap.add_argument('--tsv', default='')
ap.add_argument('--md', default='')
A = ap.parse_args()

with open(os.path.join(A.src, 'requests.csv'), newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['load'] != 'warm' and r['kind'] in ('decode', 'sample')]
loads = list(dict.fromkeys(r['load'] for r in rows))
arm_of = {r['load']: r['arm'] for r in rows}
arms = list(dict.fromkeys(arm_of[n] for n in loads))
keys = list(dict.fromkeys((r['kind'], int(r['depth']), r['pass']) for r in rows))
reply = {(r['load'], r['kind'], int(r['depth']), r['pass']): (r['ids_md5'] or r['text_md5'])[:8] for r in rows}

out, counts = [], {a: [0, 0, 0] for a in arms}  # per arm: requests, one reply over its loads, the control's reply
for k in keys:
    row = {'kind': k[0], 'depth': k[1], 'pass': k[2]}
    per_arm = {}
    for n in loads:
        row[n] = reply.get((n,) + k, '')
        per_arm.setdefault(arm_of[n], set()).add(row[n])
    ctl = per_arm.get(A.control, set())
    for a in arms:
        one = len(per_arm[a]) == 1
        same = one and per_arm[a] == ctl
        row[f'{a}_one'] = 'yes' if one else 'no'
        row[f'{a}_as_{A.control}'] = 'yes' if same else 'no'
        counts[a][0] += 1
        counts[a][1] += one
        counts[a][2] += same
    out.append(row)

fields = ['kind', 'depth', 'pass'] + loads + [f'{a}_one' for a in arms] + [f'{a}_as_{A.control}' for a in arms]
if A.tsv:
    with open(A.tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t')
        w.writeheader()
        w.writerows(out)
md = ['| Kind | Depth | Pass | ' + ' | '.join(loads) + ' |', '|---|---:|---|' + '---|' * len(loads)]
for row in out:
    md.append(f'| {row["kind"]} | {row["depth"]:,} | {row["pass"]} | ' + ' | '.join(f'`{row[n]}`' for n in loads) + ' |')
md += ['', f'| Arm | Requests | One reply over its loads | The same reply as {A.control} |', '|---|---:|---:|---:|']
for a in arms:
    md.append(f'| {a} | {counts[a][0]} | {counts[a][1]} | {counts[a][2]} |')
text = '\n'.join(md) + '\n'
if A.md:
    with open(A.md, 'w') as f:
        f.write(text)
print(text)
