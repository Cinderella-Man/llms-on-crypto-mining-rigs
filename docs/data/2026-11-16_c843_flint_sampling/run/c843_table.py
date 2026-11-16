#!/usr/bin/env python3
"""c843_table.py: C843's tables from a collected run folder (c843.sh collect): every arm's replies against the control's
(v3), request by request (the same request and seed): the generated token ids the same, or the first token where they
differ; the reply's length; how it ended; the prompt as the server rendered and counted it. Also each load's repeats
against its own first reply, and the server's per-request sampling line (tokens drawn on the device, under the grammar,
rejected, forced) and pipelined guesses, read from server-<arm>.log in request order.
  python3 -I c843_table.py <run folder> --control v3 --out table.md --tsv streams.tsv
Writes counts, md5s and token positions only: no text and no token ids.
"""
import argparse, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--control', default='v3')
ap.add_argument('--arms', default='v3 v4 v4n')
ap.add_argument('--out', required=True)
ap.add_argument('--tsv', required=True)
A = ap.parse_args()
arms = A.arms.split()

SAMP = re.compile(r'sampling = +(\d+) tokens, (\d+) drawn on the device, (\d+) while the grammar applied, (\d+) rejected '
                  r'\((\d+) after a device draw\), (\d+) forced')
PIPE = re.compile(r'pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong')


def server_lines(arm):
    s, p = [], []
    f = os.path.join(A.dir, f'server-{arm}.log')
    if os.path.exists(f):
        for line in open(f, errors='replace'):
            m = SAMP.search(line)
            if m:
                s.append(tuple(int(x) for x in m.groups()))
            m = PIPE.search(line)
            if m:
                p.append(tuple(int(x) for x in m.groups()))
    return s, p


recs = {}
for arm in arms:
    f = os.path.join(A.dir, f'{arm}.jsonl')
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    s, p = server_lines(arm)
    for j, r in enumerate(rows):
        r['samp'] = s[j] if len(s) == len(rows) else None
        r['pipe'] = p[j] if len(p) == len(rows) else None
    recs[arm] = rows
    if len(s) != len(rows):
        print(f'{arm}: {len(s)} sampling lines for {len(rows)} requests: not attached')


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def key(r):
    return (r['label'], r['seed'], r['rep'])


ctl = {key(r): r for r in recs.get(A.control, [])}
order = [key(r) for r in recs.get(A.control, [])]
cols = ['arm', 'label', 'seed', 'repeat', 'prompt_tokens', 'prompt_md5', 'body_md5', 'tokens', 'finish_reason',
        'stop_type', 'tool_calls', 'grammar_md5', 'ids_md5', 'reasoning_md5', 'content_md5', 'tool_calls_md5', 'same_ids_as_control',
        'first_diff_token', 'same_prompt_as_control', 'device_drawn', 'grammar_applied', 'rejected',
        'rejected_after_device', 'forced', 'pipe_guesses', 'pipe_right', 'pipe_wrong']
lines = ['\t'.join(cols)]
summ = {}
for arm in arms:
    if arm not in recs:
        continue
    st = summ.setdefault(arm, {'n': 0, 'same': 0, 'tokens': 0, 'same_tokens': 0, 'prompt_same': 0, 'end_same': 0,
                               'len_same': 0, 'diffs': []})
    for r in recs[arm]:
        c = ctl.get(key(r))
        fd = first_diff(r['ids'], c['ids']) if c else None
        same = c is not None and fd is None
        psame = c is not None and r['prompt_md5'] == c['prompt_md5'] and r['tokens_evaluated'] == c['tokens_evaluated']
        s = r['samp'] or ('',) * 6
        p = r['pipe'] or ('',) * 3
        lines.append('\t'.join(str(x) for x in [
            arm, r['label'], r['seed'], r['rep'], r['tokens_evaluated'], r['prompt_md5'], r['body_md5'], r['n_ids'],
            r['finish_reason'], r['stop_type'], len(r['tool_calls']), r['grammar_md5'], r['ids_md5'], r['reasoning_md5'],
            r['content_md5'], r['tool_calls_md5'], int(same) if c else '', '' if same or not c else fd,
            int(psame) if c else '', *s, *p]))
        if arm == A.control or not c:
            continue
        st['n'] += 1
        st['same'] += same
        st['tokens'] += c['n_ids']
        st['same_tokens'] += c['n_ids'] if same else (fd or 0)
        st['prompt_same'] += psame
        st['end_same'] += r['finish_reason'] == c['finish_reason'] and r['stop_type'] == c['stop_type']
        st['len_same'] += r['n_ids'] == c['n_ids']
        if not same:
            st['diffs'].append((key(r), fd, r['n_ids'], c['n_ids']))
open(A.tsv, 'w').write('\n'.join(lines) + '\n')


def endstr(r):
    e = r['finish_reason']
    if r['finish_reason'] == 'tool_calls':
        e += f" ({len(r['tool_calls'])})"
    return e


md = [f'# C843: the arms against `{A.control}`, request by request', '',
      'Each row is one request (a recorded agent request, label, at a fixed seed; `rep` 1 a repeat sent again at the end of '
      'the load). For each arm: the reply\'s length in generated tokens and how it ended; "same" when its token ids '
      f'are those of `{A.control}`\'s reply to the same request, else the first token (0-based) where they differ.', '']
hdr = '| request | seed | rep | prompt tokens | ' + ' | '.join(f'`{a}`' for a in arms if a in recs) + ' |'
md += [hdr, '|' + '---|' * (4 + len([a for a in arms if a in recs]))]
for k in order:
    c = ctl[k]
    cells = []
    for arm in arms:
        if arm not in recs:
            continue
        r = next((x for x in recs[arm] if key(x) == k), None)
        if r is None:
            cells.append('missing')
            continue
        if arm == A.control:
            cells.append(f"{r['n_ids']}, {endstr(r)}")
        else:
            fd = first_diff(r['ids'], c['ids'])
            cells.append(f"{r['n_ids']}, {endstr(r)}, " + ('same' if fd is None else f'**differs at token {fd}**'))
    md.append(f'| {k[0]} | {k[1]} | {k[2]} | {c["tokens_evaluated"]} | ' + ' | '.join(cells) + ' |')
md += ['', '## Totals against the control', '',
       '| arm | requests | the same token ids | tokens compared | tokens the same | same length | same end | same prompt |',
       '|---|---|---|---|---|---|---|---|']
for arm, st in summ.items():
    if arm == A.control:
        continue
    md.append(f"| `{arm}` | {st['n']} | {st['same']} | {st['tokens']} | {st['same_tokens']} | {st['len_same']} | "
              f"{st['end_same']} | {st['prompt_same']} |")
md += ['', '## Repeats within a load', '', '| arm | request | seed | the same token ids as the load\'s first reply |',
       '|---|---|---|---|']
for arm in arms:
    for r in recs.get(arm, []):
        if r['rep']:
            f0 = next(x for x in recs[arm] if x['label'] == r['label'] and x['seed'] == r['seed'] and not x['rep'])
            fd = first_diff(r['ids'], f0['ids'])
            md.append(f"| `{arm}` | {r['label']} | {r['seed']} | {'yes' if fd is None else f'no, from token {fd}'} |")
md += ['', '## The sampler path, from the server log', '',
       '| arm | requests | tokens | drawn on the device | while the grammar applied | rejected (after a device draw) | '
       'forced | pipelined guesses, right, wrong |', '|---|---|---|---|---|---|---|---|']
for arm in arms:
    rows = recs.get(arm, [])
    if not rows or rows[0]['samp'] is None:
        continue
    t = [sum(r['samp'][i] for r in rows) for i in range(6)]
    pp = [sum(r['pipe'][i] for r in rows) for i in range(3)] if rows[0]['pipe'] else ['', '', '']
    md.append(f'| `{arm}` | {len(rows)} | {t[0]} | {t[1]} | {t[2]} | {t[3]} ({t[4]}) | {t[5]} | {pp[0]}, {pp[1]}, {pp[2]} |')
md += ['', '## Settings each reply reports', '', '| arm | seeds echoed | top-k, top-p, min-p, temperature | samplers | '
       'backend sampling | chat format | lazy grammar | grammars |', '|---|---|---|---|---|---|---|---|']
for arm in arms:
    rows = recs.get(arm, [])
    if not rows:
        continue
    ok = all(r['seed_echo'] == r['seed'] for r in rows)
    sp = sorted({(r['top_k'], r['top_p'], r['min_p'], r['temperature']) for r in rows}, key=str)
    smp = sorted({' '.join(r['samplers'] or []) for r in rows})
    md.append(f"| `{arm}` | {'all' if ok else 'NOT all'} | {sp} | {smp} | {sorted({str(r['backend_sampling']) for r in rows})} | "
              f"{sorted({str(r['chat_format']) for r in rows})} | {sorted({str(r['grammar_lazy']) for r in rows})} | "
              f"{len({r['grammar_md5'] for r in rows})} distinct |")
open(A.out, 'w').write('\n'.join(md) + '\n')
for arm, st in summ.items():
    print(arm, {k: v for k, v in st.items() if k != 'diffs'}, st['diffs'][:5])
