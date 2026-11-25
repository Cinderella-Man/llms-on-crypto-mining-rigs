#!/usr/bin/env python3
"""rare.py: where Flash-Next's draft shortlist meets tokens new to it, and the draft's acceptance on those requests.

    python3 -I rare.py OUT [--control c1] [--arm s0] [--k-low 32768] [--tsv rare.tsv] > rare.md

OUT is a deep lane's results folder (bench/deep_lane) run with KEEP_REPLY=1. The draft's shortlist holds the ids below
k_low, the special tokens and every token the server has read or written since it started (common/speculative.cpp:
the context's tokens join the list, and so does each target token). This script replays each timed load's requests in
the order they ran (OUT/<load>.jsonl: the warm-up, then per depth the prime, the greedy passes 1 and 2 and the sampled
request; the prompts rebuilt from OUT/prompts.json as deep_client.py cuts them, checked against each record's
prompt_md5) and counts, per request, the reply's tokens at or above k_low that no earlier prompt or reply of that load
and not its own prompt held: the target tokens outside the list, where the setting's full-head rounds start. A
special token above k_low is counted too (the list holds them from the start), so a count can exceed the server's by
the specials a reply writes; the per-load table counts the tokens written `<…>` (`</think>`, `<tool_call>`) apart.
Tables: per request type and depth, the new tokens (the same in every load when the replies are the same), and each
load's accepted of drafted tokens, steps and step in ms; per load, the new tokens of all its replies summed against the server's own
count ("K target tokens outside it", OUT/server-<load>.log); and the new tokens themselves with their place in the
reply (the first eight per request). Standard library only.
"""
import argparse
import hashlib
import json
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out')
ap.add_argument('--control', default='c1')
ap.add_argument('--arm', default='s0')
ap.add_argument('--k-low', type=int, default=32768)
ap.add_argument('--tsv', default='')
A = ap.parse_args()


def md5(obj):
    return hashlib.md5((obj if isinstance(obj, str) else json.dumps(obj)).encode()).hexdigest()


pack = json.load(open(os.path.join(A.out, 'prompts.json')))
bos, tail, stream = pack['bos'], pack['tail'], pack['stream']
tail_ids = pack['meta'].get('tail_ids', len(tail))


def prompt_of(rec):
    d = int(rec['depth'])
    if rec['kind'] == 'warmup':
        return stream[:64]
    base = bos + stream[:max(d - len(bos) - len(tail), 0)]
    if rec['kind'] == 'prime':
        return base
    return (base + tail) or stream[:tail_ids]


loads = {A.control: [], A.arm: []}
recs = {}
for name in sorted(os.listdir(A.out)):
    if not name.endswith('.jsonl'):
        continue
    rows = [json.loads(line) for line in open(os.path.join(A.out, name))]
    rows = [r for r in rows if r.get('kind') != 'meta']
    if not rows or rows[0]['load'] == 'warm' or rows[0]['arm'] not in loads:
        continue
    loads[rows[0]['arm']].append(rows[0]['load'])
    recs[rows[0]['load']] = rows
order = {A.control: 0, A.arm: 1}
for arm in loads:  # the loads in the order they ran: by the first request's time
    loads[arm].sort(key=lambda ld: recs[ld][0]['time'])

SPECIAL = re.compile(r'^<\|?/?[A-Za-z_]+\|?>$')
KEY = lambda r: (r['kind'], str(r['pass']), int(r['depth']))
per = {}      # (kind, pass, depth) -> load -> dict
per_load = {}  # load -> new tokens in replies
detail = {}   # (kind, pass, depth, load) -> [(pos, id, piece)]
bad = []
for arm in loads:
    for ld in loads[arm]:
        seen = set()
        per_load[ld] = [0, 0]
        for r in recs[ld]:
            p = prompt_of(r)
            if md5(p) != r['prompt_md5']:
                bad.append(f"{ld} {r['kind']} {r['pass']} {r['depth']}")
            seen.update(p)
            new = []
            for i, t in enumerate(r.get('ids') or []):
                if t >= A.k_low and t not in seen:
                    pieces = r.get('pieces') or []
                    new.append((i, t, pieces[i] if i < len(pieces) else ''))
                seen.add(t)
            per_load[ld][0] += len(new)
            per_load[ld][1] += sum(1 for _, _, pc in new if SPECIAL.match(pc))
            if r['kind'] in ('decode', 'sample'):
                gen, acc = int(r['gen_n']), int(r['draft_accepted'] or 0)
                steps = gen - acc
                per.setdefault(KEY(r), {})[ld] = {
                    'new': len(new), 'acc': acc, 'dn': int(r['draft_n'] or 0), 'steps': steps,
                    'ms': float(r['gen_ms']) / steps if steps else float('nan'), 'ids': r['ids_md5'][:8]}
                detail[KEY(r) + (ld,)] = new
if bad:
    print('prompt rebuilt with another md5 for: ' + '; '.join(bad))

NAMES = {('decode', '1'): 'greedy, pass 1', ('decode', '2'): 'greedy, pass 2', ('sample', 's'): 'sampled'}
md = ['| Request | Depth | New tokens per load | ' + f'`{A.control}` loads: accepted/drafted, steps, ms a step | '
      f'`{A.arm}` loads | Same reply in all loads |', '|---|---:|---|---|---|---|']
tsv = ['request\tdepth\tload\tarm\tnew_tokens\taccepted\tdrafted\tsteps\tstep_ms\tids_md5']
for key in sorted(per, key=lambda k: (k[2], ['decode', 'sample'].index(k[0]), k[1])):
    kind, ps, d = key
    cells = per[key]
    news = ', '.join(str(cells[ld]['new']) for arm in loads for ld in loads[arm] if ld in cells)
    fmt = lambda arm: '; '.join(f"{ld} {cells[ld]['acc']}/{cells[ld]['dn']}, {cells[ld]['steps']}, "
                                f"{cells[ld]['ms']:.2f}" for ld in loads[arm] if ld in cells)
    same = 'yes' if len({c['ids'] for c in cells.values()}) == 1 else 'no'
    md.append(f'| {NAMES[(kind, ps)]} | {d:,} | {news} | {fmt(A.control)} | {fmt(A.arm)} | {same} |')
    for arm in loads:
        for ld in loads[arm]:
            if ld in cells:
                c = cells[ld]
                tsv.append(f"{NAMES[(kind, ps)]}\t{d}\t{ld}\t{arm}\t{c['new']}\t{c['acc']}\t{c['dn']}\t{c['steps']}\t"
                           f"{c['ms']:.3f}\t{c['ids']}")

md += ['', '| Load | Arm | New tokens in all replies (warm-up and primes too) | Of them `<…>` tokens | Server: target tokens outside the '
       'list | Server: drafts with the full head |', '|---|---|---:|---:|---:|---:|']
for arm in loads:
    for ld in loads[arm]:
        rep = None
        path = os.path.join(A.out, f'server-{ld}.log')
        if os.path.exists(path):
            for line in open(path, errors='replace'):
                m = re.search(r'draft head shortlist: (\d+) drafts with it, (\d+) with the full head, (\d+) target', line)
                if m:
                    rep = m.groups()
        md.append(f"| {ld} | {arm} | {per_load[ld][0]} | {per_load[ld][1]} | {rep[2] if rep else 'none'} | "
                  f"{rep[1] if rep else 'none'} |")

md += ['', '| Request | Depth | Load | New tokens: place in the reply and text (the first eight) |', '|---|---:|---|---|']
for key in sorted(detail, key=lambda k: (k[2], ['decode', 'sample'].index(k[0]), k[1], order[
        next(a for a in loads if k[3] in loads[a])], k[3])):
    kind, ps, d, ld = key
    if detail[key]:
        md.append(f'| {NAMES[(kind, ps)]} | {d:,} | {ld} | ' + ', '.join(f'{i}: `{json.dumps(p)[1:-1]}`'
                                                                       for i, t, p in detail[key][:8]) + ' |')
print('\n'.join(md))
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write('\n'.join(tsv) + '\n')
