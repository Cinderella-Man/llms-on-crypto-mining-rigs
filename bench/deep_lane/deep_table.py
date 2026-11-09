#!/usr/bin/env python3
"""deep_table.py: the tables of a deep-lane run (deep_lane.sh), standard library only.

    python3 -I deep_table.py OUT [--control ARM] [--class-on load|pass1|pass2] [--out table.md] [--tsv classes.tsv]

Reads OUT/requests.csv (deep_client.py's line per request) and OUT/loads.tsv (deep_lane.sh's line per load; without
it every load but `warm` counts as timed) and prints Markdown, also written to --out:
- generation: per depth and arm, tok/s of every load by value (pass 1, pass 2, and the load: its passes' tokens over
  their time), the spread between the arm's loads, the arm against the control and its class;
- prompt reading: the same for the prompt tokens read at each depth (the `prime` requests: a chunk on top of the
  previous depth's cache; with CACHE=0 the decode requests, each a whole prompt read fresh);
- the draft and the step: accepted of drafted tokens, tokens a step and ms a step per load;
- the sampler pass, when the run has one;
- the checks: the same prompt ids on every load, the tokens out, the greedy reply the same on every pass, load and
  arm (its token ids' md5 where the server returned them, else its text's), what each prime read.
The class of an arm against the control (--control, default the first arm): the spread is the larger of the two
arms' differences between their fastest and slowest load; gain when the arm's mean is above the control's by more
than the spread, loss when below by more than it, else negligible; with one load per arm there is no spread and no
class. --class-on picks the speed that is classed: load (default, both passes), pass1 or pass2.
Requests marked valid=0 (deep_lane.sh's QUIET=0) carry times that are not measurements: no speed is printed for a run
that has any, only the draft counters and the checks. --tsv writes the classes as a table (metric, depth, arm, control,
each load's value, spread, delta, percent, class).
"""
import argparse
import csv
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--control', default='')
ap.add_argument('--class-on', choices=('load', 'pass1', 'pass2'), default='load')
ap.add_argument('--out', default='')
ap.add_argument('--tsv', default='')
A = ap.parse_args()
PASSES = {'load': ('1', '2'), 'pass1': ('1',), 'pass2': ('2',)}[A.class_on]
MIN_READ = 32  # prompt tokens a request must have read for its prompt speed to count

with open(os.path.join(A.src, 'requests.csv'), newline='') as f:
    rows = list(csv.DictReader(f))
loads = {}
if os.path.exists(os.path.join(A.src, 'loads.tsv')):
    with open(os.path.join(A.src, 'loads.tsv'), newline='') as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            loads[r['load']] = r
else:
    for r in rows:
        loads.setdefault(r['load'], {'load': r['load'], 'arm': r['arm'], 'timed': '0' if r['load'] == 'warm' else '1'})
R = {(r['load'], r['kind'], r['pass'], int(r['depth'])): r for r in rows}
timed = [n for n, ld in loads.items() if ld['timed'] == '1' and any(r['load'] == n and r['kind'] == 'decode' for r in rows)]
if not timed:
    sys.exit('no timed load with decode requests in requests.csv')
arms = list(dict.fromkeys(loads[n]['arm'] for n in timed))
by_arm = {a: [n for n in timed if loads[n]['arm'] == a] for a in arms}
control = A.control or arms[0]
if control not in arms:
    sys.exit(f'no arm {control} among {", ".join(arms)}')
depths = sorted({k[3] for k in R if k[0] in timed and k[1] == 'decode'})
valid = all(r['valid'] == '1' for r in rows if r['load'] in timed)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def speed(rs, n_key, ms_key):
    rs = [r for r in rs if r]
    n, ms = sum(num(r[n_key]) for r in rs), sum(num(r[ms_key]) for r in rs)
    return n * 1000 / ms if n and ms > 0 else None


def gen(ld, d, passes=PASSES):
    return speed([R.get((ld, 'decode', p, d)) for p in passes], 'gen_n', 'gen_ms')


def read_rows(ld, d):
    rs = [R[(ld, 'prime', '0', d)]] if (ld, 'prime', '0', d) in R else [R.get((ld, 'decode', p, d)) for p in ('1', '2')]
    return [r for r in rs if r and num(r['prompt_n']) >= MIN_READ]


def read(ld, d):
    return speed(read_rows(ld, d), 'prompt_n', 'prompt_ms')


def mean(v):
    return sum(v) / len(v)


def spread(v):
    v = [x for x in v if x is not None]
    return max(v) - min(v) if len(v) > 1 else None


def verdict(ctl, arm):
    """(spread, delta, percent, class) of an arm's load speeds against the control's."""
    ctl, arm = [v for v in ctl if v is not None], [v for v in arm if v is not None]
    if not ctl or not arm:
        return None, None, None, '—'
    delta = mean(arm) - mean(ctl)
    pct = 100 * delta / mean(ctl)
    if len(ctl) < 2 or len(arm) < 2:
        return None, delta, pct, 'no class (one load)'
    s = max(spread(ctl), spread(arm))
    return s, delta, pct, 'gain' if delta > s else 'loss' if delta < -s else 'negligible'


def fmt(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


classes = []


def speed_table(title, note, metric, value, cell, nd, groups):
    """A table of one speed per load: `groups` are (label, key) rows, value(load, key) the classed number."""
    md = ['', f'### {title}', '', note, '',
          f'| Depth (tokens) | Arm | Per load | Spread | Against `{control}` | | Class |', '|---:|---|---|---:|---:|---:|---|']
    for label, key in groups:
        for i, a in enumerate(arms):
            vals = [value(ld, key) for ld in by_arm[a]]
            per = '; '.join(f'{ld}: {cell(ld, key)}' for ld in by_arm[a])
            if a == control:
                tail = 'control | | |'
            else:
                s, delta, pct, cls = verdict([value(ld, key) for ld in by_arm[control]], vals)
                tail = f'{fmt(delta, nd) if delta is None else f"{delta:+.{nd}f}"} tok/s | ' \
                       f'{"—" if pct is None else f"{pct:+.1f} %"} | {cls} |'
                classes.append([metric, label.replace(',', ''), a, control, ' '.join(fmt(v, nd) for v in vals),
                                ' '.join(fmt(value(ld, key), nd) for ld in by_arm[control]), fmt(s, nd), fmt(delta, nd),
                                fmt(pct, 1), cls])
            md.append(f'| {label if i == 0 else ""} | {a} | {per} | {fmt(spread(vals), nd)} | {tail}')
    return md


def gen_cell(ld, d):
    return (', '.join(fmt(gen(ld, d, (p,))) for p in ('1', '2')) + f' → **{fmt(gen(ld, d))}**')


def gen_all(ld, _):
    return speed([R.get((ld, 'decode', p, d)) for d in depths for p in PASSES], 'gen_n', 'gen_ms')


def read_cell(ld, d):
    rs = read_rows(ld, d)
    if not rs:
        return '—'
    return (', '.join(fmt(speed([r], 'prompt_n', 'prompt_ms'), 1) for r in rs)
            + (f' → **{fmt(read(ld, d), 1)}**' if len(rs) > 1 else '')
            + f' ({", ".join(r["prompt_n"] for r in rs)} read on {", ".join(r["cache_n"] or "0" for r in rs)} cached)')


def steps(r):
    return num(r['gen_n']) - num(r['draft_accepted'])


def draft_cell(r):
    if not r:
        return '—'
    s = (f'{r["draft_accepted"]} of {r["draft_n"]} ({num(r["draft_accepted"]) / num(r["draft_n"]):.3f}), '
         if num(r['draft_n']) else 'no draft, ')
    s += f'{num(r["gen_n"]) / steps(r):.2f} a step' if steps(r) > 0 else '— a step'
    if valid and steps(r) > 0:
        s += f', {num(r["gen_ms"]) / steps(r):.1f} ms a step'
    return s


def reply_md5(r):
    return r['ids_md5'] or r['text_md5']


n_predict = sorted({r['gen_n'] for r in rows if r['load'] in timed and r['kind'] == 'decode'})
first = loads[timed[0]]
md = [f'## The deep lane: {", ".join(arms)}', '']
if first.get('model'):
    build = ''
    if os.path.exists(os.path.join(A.src, 'build.txt')):
        with open(os.path.join(A.src, 'build.txt'), errors='replace') as f:
            build = next((line.strip() for line in f if line.startswith('version')), '')
    md += [f'Model file `{first["model"]}`' + (f', llama-server {build}' if build else '') + '.', '']
md += [f'{" or ".join(n_predict)} tokens per request at temperature 0, each request twice per load (pass 1 is the first '
       f'request at a new depth, pass 2 the same request again). Timed loads, in their order: {", ".join(timed)}.', '',
       '| Arm | Loads | Draft | Context | Its own variables and flags |', '|---|---|---|---:|---|']
for a in arms:
    ld = loads[by_arm[a][0]]
    own = ', '.join(f'`{ld[k]}`' for k in ('arm_env', 'arm_args') if ld.get(k))
    md.append(f'| {a}{" (control)" if a == control else ""} | {", ".join(by_arm[a])} | {ld.get("draft", "")} | '
              f'{ld.get("ctx", "")} | {own or "—"} |')

if valid:
    what = {'load': "its two passes' tokens over their time", 'pass1': 'its pass 1', 'pass2': 'its pass 2'}[A.class_on]
    rule = (f'A load\'s speed is {what}. Spread: the difference between the arm\'s fastest and slowest load. Class '
            f'against `{control}`: gain when the mean is above the control\'s by more than the larger of the two '
            'spreads, loss when below by more than it, else negligible.')
    md += speed_table('Generation, tok/s', 'Per load: pass 1, pass 2 → the load. ' + rule, 'generation', gen, gen_cell,
                      2, [(f'{d:,}', d) for d in depths])
    md += speed_table('Generation over all depths, tok/s', 'Every depth\'s tokens over their time, per load.',
                      'generation_all', gen_all, lambda ld, k: f'**{fmt(gen_all(ld, k))}**', 2, [('all', None)])
    md += speed_table('Prompt reading, tok/s',
                      'Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of '
                      'the cache; without primes each decode request reads its prompt), with the tokens read and the '
                      f'tokens reused. Requests that read under {MIN_READ} tokens are left out.',
                      'prompt', read, read_cell, 1, [(f'{d:,}', d) for d in depths])
else:
    md += ['', '**The times of this run are not measurements** (the quiet checks were off, valid=0), so no speed is '
           'printed. The counts below do not depend on the machine.']

md += ['', '### The draft and the step', '',
       'Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a '
       'step being one pass of the main model: generated minus accepted)' + (', and the time of a step.' if valid else '.')
       + ' Pass 2 is listed where its counters differ.', '', '| Depth (tokens) | Arm | Per load |', '|---:|---|---|']
for d in depths:
    for i, a in enumerate(arms):
        cells = []
        for ld in by_arm[a]:
            r1, r2 = R.get((ld, 'decode', '1', d)), R.get((ld, 'decode', '2', d))
            c = f'{ld}: {draft_cell(r1)}'
            if r1 and r2 and (r1['draft_n'], r1['draft_accepted']) != (r2['draft_n'], r2['draft_accepted']):
                c += f' (pass 2: {draft_cell(r2)})'
            cells.append(c)
        md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {"; ".join(cells)} |')

if any(r['kind'] == 'sample' and r['load'] in timed for r in rows):
    md += ['', '### The sampler pass', '',
           'One request per depth and load with the server line\'s own sampler and a fixed seed: '
           + ('tok/s, ' if valid else '') + 'the draft counters and the step.', '',
           '| Depth (tokens) | Arm | Per load |', '|---:|---|---|']
    for d in depths:
        for i, a in enumerate(arms):
            cells = []
            for ld in by_arm[a]:
                r = R.get((ld, 'sample', 's', d))
                v = speed([r], 'gen_n', 'gen_ms') if valid else None
                cells.append(f'{ld}: ' + (f'{fmt(v)} tok/s, ' if v is not None else '') + draft_cell(r))
            md.append(f'| {f"{d:,}" if i == 0 else ""} | {a} | {"; ".join(cells)} |')

md += ['', '### Checks', '']
md.append('- Times: ' + ('every load started on a machine checked quiet (valid=1).' if valid
                         else 'not measurements (valid=0).'))
sizes = {d: sorted({r['prompt_tokens'] for k, r in R.items() if k[0] in timed and k[1] == 'decode' and k[3] == d})
         for d in depths}
same_prompt = all(len({r['prompt_md5'] for k, r in R.items() if k[0] in timed and k[1] == 'decode' and k[3] == d}) == 1
                  for d in depths)
md.append('- Prompts: ' + ', '.join(f'{"/".join(sizes[d])} ids at depth {d:,}' for d in depths) + '; '
          + ('the same ids on every load and pass.' if same_prompt else '**NOT the same ids on every load and pass.**'))
md.append(f'- Tokens out: {n_predict[0]} on every timed request.' if len(n_predict) == 1
          else f'- Tokens out differ between timed requests: {", ".join(n_predict)}.')
for d in depths:
    ref = {reply_md5(R[(ld, 'decode', p, d)]) for ld in by_arm[control] for p in ('1', '2') if (ld, 'decode', p, d) in R}
    line = f'- Greedy reply at depth {d:,}: '
    if len(ref) != 1:
        md.append(line + f'**`{control}` itself wrote {len(ref)} different replies over its passes and loads.**')
        continue
    off = [a for a in arms if a != control and {reply_md5(R[(ld, 'decode', p, d)]) for ld in by_arm[a] for p in ('1', '2')
                                                if (ld, 'decode', p, d) in R} != ref]
    md.append(line + (f'**differs from `{control}` in {", ".join(off)}.**' if off
                      else 'the same on every pass and load' + (' of every arm.' if len(arms) > 1 else '.')))
primes = [(ld, d, R[(ld, 'prime', '0', d)]) for ld in timed for d in depths if (ld, 'prime', '0', d) in R]
if primes:
    seen = sorted({(d, r['prompt_n'], r['cache_n'] or '0', r['cache_src']) for _, d, r in primes})
    md.append('- Primes (read on cached, the cache\'s source): '
              + '; '.join(f'{d:,}: {n} on {c}' + (f' ({s})' if s else '') for d, n, c, s in seen) + '.')
text = '\n'.join(md) + '\n'
if A.out:
    with open(A.out, 'w') as f:
        f.write(text)
if A.tsv:
    with open(A.tsv, 'w') as f:
        f.write('metric\tdepth\tarm\tcontrol\tloads\tcontrol_loads\tspread\tdelta\tpercent\tclass\n')
        for c in classes:
            f.write('\t'.join(c) + '\n')
sys.stdout.write(text)
