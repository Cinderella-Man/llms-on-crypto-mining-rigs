#!/usr/bin/env python3
"""draft_pair.py: a deep-lane run of a line with its draft against the same line without it, read part by part.
Standard library only.

    python3 -I draft_pair.py OUT [--draft draft] [--plain nodraft] [--second FOLDER:ARM] [--second-name NAME]
                             [--yield NAME:DEPTH=TOKENS,...]... [--out draft_pair.md] [--tsv PREFIX]

OUT is a results folder of bench/deep_lane/deep_lane.sh whose arms are the line with the draft (--draft) and the same
line without it (--plain). Tokens a second of the draft line is two numbers: the tokens a step yields (one of the main
model's own and the draft tokens it accepts, which depends on the text written) over the time of a step (which does
not). The line without the draft yields one token a step. Per depth this prints:
- tok/s and ms a token of every load of both arms, each arm's spread, and the class of the draft against no draft by
  bench/deep_lane's rule (gain when the draft arm's mean is above the plain arm's by more than the larger of the two
  spreads, loss when below by more than it, else negligible), on both passes together, on pass 1 and on pass 2;
- the step: its time in each load of the draft arm, the time of a token in each load of the plain arm, their ratio
  (the tokens a step the draft must yield to break even), the share of the drafted tokens that takes, and what each
  load yielded and accepted;
- with --yield (tokens a step per depth measured elsewhere, for example with the line's own sampler), the draft
  against no draft at that yield and this run's step times: an estimate, not a measurement;
- the draft arm's two loads against each other: the difference in tok/s as its two factors, the tokens a step and the
  step's time;
- prompt reading: tok/s and ms a prompt token of every load, the class, and what the draft adds per prompt token;
- a whole turn: the new prompt tokens per generated token at which the draft's cost in the prompt equals what it
  saves in generation, from the first request's times;
- with --second FOLDER:ARM (another run's loads of the draft arm's setting), the two runs side by side.
--tsv PREFIX writes PREFIX_loads.tsv (one line per source, load, depth and pass), PREFIX_classes.tsv (the columns of
deep_table.py's classes.tsv), PREFIX_breakeven.tsv, PREFIX_prompt.tsv and PREFIX_turn.tsv.
"""
import argparse
import csv
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--draft', default='draft')
ap.add_argument('--plain', default='nodraft')
ap.add_argument('--second', default='')
ap.add_argument('--second-name', default='the second run')
ap.add_argument('--yield', dest='yields', action='append', default=[])
ap.add_argument('--out', default='')
ap.add_argument('--tsv', default='')
A = ap.parse_args()
PASS_SETS = (('both', ('1', '2')), ('1', ('1',)), ('2', ('2',)))
PASS_NAME = {'both': 'both passes', '1': 'pass 1', '2': 'pass 2'}
MIN_READ = 32  # prompt tokens a request must have read for its prompt speed to count (deep_table.py's rule)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def read_run(src):
    """A results folder's timed requests: the decode requests {(load, pass, depth): row}, the request that read each
    depth's prompt {(load, depth): row} (the prime, else the first decode request), and the loads per arm in run
    order."""
    with open(os.path.join(src, 'requests.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['kind'] in ('decode', 'prime')]
    path = os.path.join(src, 'loads.tsv')
    if os.path.exists(path):
        with open(path, newline='') as f:
            timed = {r['load'] for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE) if r['timed'] == '1'}
        rows = [r for r in rows if r['load'] in timed]
    else:
        rows = [r for r in rows if r['load'] != 'warm']
    if any(r['valid'] != '1' for r in rows):
        sys.exit(f'{src}: requests marked valid=0 carry no measured time')
    by_arm, decode, prompt = {}, {}, {}
    for r in rows:
        loads = by_arm.setdefault(r['arm'], [])
        if r['load'] not in loads:
            loads.append(r['load'])
        key = (r['load'], int(r['depth']))
        if r['kind'] == 'decode':
            decode[(r['load'], r['pass'], int(r['depth']))] = r
            if r['pass'] == '1' and num(r['prompt_n']) >= MIN_READ:
                prompt.setdefault(key, r)
        elif num(r['prompt_n']) >= MIN_READ:
            prompt[key] = r
    return decode, prompt, by_arm


class Cell:
    """One load at one depth over the given passes."""

    def __init__(self, table, load, depth, passes):
        rs = [table[(load, p, depth)] for p in passes if (load, p, depth) in table]
        self.n = sum(num(r['gen_n']) for r in rs)
        self.ms = sum(num(r['gen_ms']) for r in rs)
        self.drafted = sum(num(r['draft_n']) for r in rs)
        self.accepted = sum(num(r['draft_accepted']) for r in rs)
        self.steps = self.n - self.accepted
        self.ok = bool(rs) and self.n > 0 and self.ms > 0 and self.steps > 0
        self.replies = list(dict.fromkeys(r['ids_md5'] or r['text_md5'] for r in rs))
        if self.ok:
            self.tps = self.n * 1000 / self.ms
            self.ms_token = self.ms / self.n
            self.tok_step = self.n / self.steps
            self.ms_step = self.ms / self.steps
            self.share = self.accepted / self.drafted if self.drafted else None
            self.drafted_step = self.drafted / self.steps


def mean(v):
    return sum(v) / len(v)


def spread(v):
    return max(v) - min(v) if len(v) > 1 else None


def verdict(ctl, arm, lower=False):
    """bench/deep_lane's class of an arm's load values against the control's; lower: a time, less is the gain."""
    delta = mean(arm) - mean(ctl)
    pct = 100 * delta / mean(ctl)
    if len(ctl) < 2 or len(arm) < 2:
        return None, delta, pct, 'no class (one load)'
    s = max(spread(ctl), spread(arm))
    better = -delta if lower else delta
    return s, delta, pct, 'gain' if better > s else 'loss' if better < -s else 'negligible'


def f2(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def depth_label(d):
    return f'{d:,}'


T, P, ARMS = read_run(A.src)
for a in (A.draft, A.plain):
    if a not in ARMS:
        sys.exit(f'no arm {a} among {", ".join(ARMS)}')
DL, PL = ARMS[A.draft], ARMS[A.plain]
DEPTHS = sorted({k[2] for k in T})
T2, L2 = {}, []
if A.second:
    folder, _, arm2 = A.second.rpartition(':')
    T2, _, arms2 = read_run(folder)
    if arm2 not in arms2:
        sys.exit(f'no arm {arm2} in {folder}')
    L2 = arms2[arm2]
YIELDS = []
for y in A.yields:
    name, _, pairs = y.partition(':')
    YIELDS.append((name, {int(k): float(v) for k, v in (p.split('=') for p in pairs.split(','))}))


def cells(table, loads, d, passes):
    return [c for c in (Cell(table, ld, d, passes) for ld in loads) if c.ok]


md = [f'## The draft against no draft: `{A.draft}` ({", ".join(DL)}) and `{A.plain}` ({", ".join(PL)})', '']
classes, breakeven, load_lines, prompt_lines, turn_lines = [], [], [], [], []
BE = {}  # (depth, pass set) -> (mean step of the draft arm, mean token of the plain arm)

# 1. tok/s and ms a token, with the class of the draft against no draft
for name, passes in PASS_SETS:
    md += [f'### Generation on {PASS_NAME[name]}: tok/s (ms a token)', '',
           'Per load' + (': pass 1, pass 2 → the load, its two requests\' tokens over their time' if name == 'both'
                         else '') + '. Spread: the difference between an arm\'s loads. Class of '
           f'`{A.draft}` against `{A.plain}`: gain when its mean is above `{A.plain}`\'s by more than the larger '
           'spread, loss when below by more than it, else negligible.', '',
           '| Depth (tokens) | ' + ' | '.join(f'`{ld}`' for ld in DL) + ' | Spread | '
           + ' | '.join(f'`{ld}`' for ld in PL) + f' | Spread | `{A.draft}` against `{A.plain}` | | Class |',
           '|---:|' + '---|' * len(DL) + '---:|' + '---|' * len(PL) + '---:|---:|---:|---|']
    groups = [(depth_label(d), [d]) for d in DEPTHS] + ([('all depths', DEPTHS)] if name == 'both' else [])
    for label, ds in groups:
        def val(table, ld):
            n = sum(Cell(table, ld, d, passes).n for d in ds)
            ms = sum(Cell(table, ld, d, passes).ms for d in ds)
            return n * 1000 / ms if n and ms > 0 else None

        def show(table, ld):
            v = val(table, ld)
            if v is None:
                return '—'
            lead = ''
            if name == 'both' and len(ds) == 1:
                lead = ', '.join(f2(Cell(table, ld, ds[0], (p,)).tps) for p in passes) + ' → '
            return f'{lead}**{f2(v)}** ({1000 / v:.2f})'
        dv, pv = [val(T, ld) for ld in DL], [val(T, ld) for ld in PL]
        if None in dv or None in pv:
            continue
        s, delta, pct, cls = verdict(pv, dv)
        md.append(f'| {label} | ' + ' | '.join(show(T, ld) for ld in DL) + f' | {f2(spread(dv))} | '
                  + ' | '.join(show(T, ld) for ld in PL) + f' | {f2(spread(pv))} | {delta:+.2f} tok/s | {pct:+.1f} % | {cls} |')
        metric = {'both': 'generation', '1': 'generation_pass1', '2': 'generation_pass2'}[name]
        if len(ds) > 1:
            metric = 'generation_all'
        classes.append([metric, label.replace(',', '').replace('all depths', 'all'), A.draft, A.plain,
                        ' '.join(f2(v) for v in dv), ' '.join(f2(v) for v in pv), f2(s), f2(delta), f2(pct, 1), cls])
    md.append('')

# 2. the step: what it costs, what it must yield, what it yielded
md += ['### The step: what it costs and what it must yield', '',
       f'A step of `{A.draft}` is one pass of the main model over the last token and the drafted ones, plus the '
       f'draft\'s own passes; it yields one token and the draft tokens it accepts. A step of `{A.plain}` is one token. '
       'The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is '
       'the break-even, from the means of each arm\'s loads. The share to break even is the break-even minus one, '
       'over the draft tokens the arm offered a step.', '',
       f'| Depth (tokens) | Pass | Step of `{A.draft}`, ms: ' + ', '.join(f'`{ld}`' for ld in DL)
       + f' | Token of `{A.plain}`, ms: ' + ', '.join(f'`{ld}`' for ld in PL)
       + ' | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: '
       + ', '.join(f'`{ld}`' for ld in DL) + ' | Share accepted: ' + ', '.join(f'`{ld}`' for ld in DL) + ' |',
       '|---:|---|---|---|---:|---:|---|---|']
for d in DEPTHS:
    for name, passes in PASS_SETS[1:] + PASS_SETS[:1]:
        dc, pc = cells(T, DL, d, passes), cells(T, PL, d, passes)
        if len(dc) != len(DL) or len(pc) != len(PL):
            continue
        step, token = mean([c.ms_step for c in dc]), mean([c.ms_step for c in pc])
        BE[(d, name)] = (step, token)
        be = step / token
        per_step = mean([c.drafted_step for c in dc])
        be_share = (be - 1) / per_step if per_step else None
        md.append(f'| {depth_label(d) if name == "1" else ""} | {name} | ' + ', '.join(f2(c.ms_step) for c in dc) + ' | '
                  + ', '.join(f2(c.ms_step) for c in pc) + f' | {be:.2f} | {f2(be_share, 3)} | '
                  + ', '.join(f2(c.tok_step) for c in dc) + ' | ' + ', '.join(f2(c.share, 3) for c in dc) + ' |')
        breakeven.append([d, name, ' '.join(f2(c.ms_step) for c in dc), ' '.join(f2(c.ms_step) for c in pc), f2(be, 3),
                          f2(per_step, 3), f2(be_share, 3), ' '.join(f2(c.tok_step, 3) for c in dc),
                          ' '.join(f2(c.share, 3) for c in dc)])
md.append('')

# 2b. the same step times at a yield measured elsewhere
if YIELDS:
    md += ['### The draft against no draft at another yield (est.)', '',
           'Tokens a step measured elsewhere, at this run\'s step times: the yield over the break-even, minus one. An '
           'estimate, not a measured speed.', '',
           '| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | '
           'Est. on pass 2 |', '|---:|---|---:|---:|---:|---:|---:|']
    for d in DEPTHS:
        first = True
        for name, ys in YIELDS:
            if d not in ys or (d, '1') not in BE or (d, '2') not in BE:
                continue
            b1, b2 = (BE[(d, p)][0] / BE[(d, p)][1] for p in ('1', '2'))
            md.append(f'| {depth_label(d) if first else ""} | {name} | {ys[d]:.2f} | {b1:.2f} | '
                      f'{100 * (ys[d] / b1 - 1):+.1f} % | {b2:.2f} | {100 * (ys[d] / b2 - 1):+.1f} % |')
            first = False
    md.append('')

# 3. the draft arm's loads against each other
if len(DL) == 2:
    a, b = DL
    md += [f'### `{a}` against `{b}`: the text and the step', '',
           'Tok/s is the tokens a step over the step\'s time, so the difference between two loads of one setting is '
           'the product of two factors. Both passes of each load together.', '',
           f'| Depth (tokens) | tok/s: `{a}`, `{b}` | `{a}` against `{b}` | Tokens a step: `{a}`, `{b}` | Factor | '
           f'Step, ms: `{a}`, `{b}` | Factor | Replies |', '|---:|---|---:|---|---:|---|---:|---|']
    for d in DEPTHS:
        ca, cb = Cell(T, a, d, ('1', '2')), Cell(T, b, d, ('1', '2'))
        if not (ca.ok and cb.ok):
            continue
        same = 'the same' if ca.replies == cb.replies and len(ca.replies) == 1 else 'different'
        md.append(f'| {depth_label(d)} | {f2(ca.tps)}, {f2(cb.tps)} | {100 * (ca.tps / cb.tps - 1):+.1f} % | '
                  f'{f2(ca.tok_step)}, {f2(cb.tok_step)} | {100 * (ca.tok_step / cb.tok_step - 1):+.1f} % | '
                  f'{f2(ca.ms_step)}, {f2(cb.ms_step)} | {100 * (cb.ms_step / ca.ms_step - 1):+.1f} % | {same} |')
    md.append('')

# 4. prompt reading, and a whole turn
md += ['### Prompt reading: tok/s (ms a prompt token)', '',
       'Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the '
       'first decode request). The draft adds: the difference between the arms\' mean ms a prompt token, and its '
       f'share of `{A.draft}`\'s.', '',
       '| Depth (tokens) | Read, on cached | ' + ' | '.join(f'`{ld}`' for ld in DL) + ' | '
       + ' | '.join(f'`{ld}`' for ld in PL) + f' | `{A.draft}` against `{A.plain}` | Class | The draft adds, ms a '
       'prompt token | Share of the prompt time |',
       '|---:|---|' + '---|' * (len(DL) + len(PL)) + '---:|---|---:|---:|']
ADD = {}
for d in DEPTHS:
    rs = {ld: P.get((ld, d)) for ld in DL + PL}
    if not all(rs.values()):
        continue
    tps = {ld: num(r['prompt_n']) * 1000 / num(r['prompt_ms']) for ld, r in rs.items()}
    s, delta, pct, cls = verdict([tps[ld] for ld in PL], [tps[ld] for ld in DL])
    md_ms, mp_ms = mean([1000 / tps[ld] for ld in DL]), mean([1000 / tps[ld] for ld in PL])
    ADD[d] = md_ms - mp_ms
    r0 = rs[DL[0]]
    md.append(f'| {depth_label(d)} | {r0["prompt_n"]} on {r0["cache_n"] or "0"} | '
              + ' | '.join(f'**{tps[ld]:.1f}** ({1000 / tps[ld]:.2f})' for ld in DL + PL)
              + f' | {pct:+.1f} % | {cls} | {ADD[d]:+.2f} | {100 * ADD[d] / md_ms:.1f} % |')
    prompt_lines.append([d, r0['prompt_n'], r0['cache_n'] or '0', ' '.join(f'{tps[ld]:.2f}' for ld in DL),
                         ' '.join(f'{tps[ld]:.2f}' for ld in PL), f2(s), f2(delta), f2(pct, 1), cls, f2(md_ms, 3),
                         f2(mp_ms, 3), f2(ADD[d], 3), f2(100 * ADD[d] / md_ms, 1)])
    classes.append(['prompt', str(d), A.draft, A.plain, ' '.join(f'{tps[ld]:.2f}' for ld in DL),
                    ' '.join(f'{tps[ld]:.2f}' for ld in PL), f2(s), f2(delta), f2(pct, 1), cls])
md.append('')

md += ['### A whole turn: the prompt it reads against the tokens it writes', '',
       'A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain '
       'line\'s ms a token minus the draft line\'s, on pass 1, the first request at a depth) and costs time per '
       'prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per '
       'generated token up to which the draft pays in a turn at that depth. For this run\'s loads the saving is '
       'between the arms\' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over '
       'that yield (est.).', '',
       '| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | '
       'Prompt tokens per generated token to break even |', '|---:|---:|---|---:|---:|---:|']
for d in DEPTHS:
    if d not in ADD or (d, '1') not in BE:
        continue
    step, token = BE[(d, '1')]
    dc, pc = cells(T, DL, d, ('1',)), cells(T, PL, d, ('1',))
    rows_ = [('this run\'s loads', mean([c.tok_step for c in dc]),
              mean([c.ms_token for c in pc]) - mean([c.ms_token for c in dc]))]
    rows_ += [(f'{name} (est.)', ys[d], token - step / ys[d]) for name, ys in YIELDS if d in ys]
    for i, (name, tok, saved) in enumerate(rows_):
        if ADD[d] <= 0:
            ratio = 'no cost in the prompt'
        elif saved <= 0:
            ratio = 'none: nothing saved'
        else:
            ratio = f'{saved / ADD[d]:.2f}'
        md.append(f'| {depth_label(d) if i == 0 else ""} | {f"{ADD[d]:+.2f}" if i == 0 else ""} | {name} | {tok:.2f} | '
                  f'{saved:+.2f} | {ratio} |')
        turn_lines.append([d, f2(ADD[d], 3), name, f2(tok, 3), f2(saved, 3), ratio])
md.append('')

# 5. a second run of the draft arm's setting
if L2:
    md += [f'### The same line twice: `{A.draft}` here and `{arm2}` in {A.second_name}', '',
           'Per load: tok/s on pass 1, pass 2 → the load with its ms a token; the share accepted and the tokens a step '
           'on pass 1; the step on pass 1 and on pass 2; the first six characters of the reply\'s hash on pass 1 (loads '
           'with the same characters wrote the same reply).', '',
           '| Depth (tokens) | Load | tok/s (ms a token) | Share accepted | Tokens a step | Step on pass 1, ms | '
           'Step on pass 2, ms | Reply |', '|---:|---|---|---:|---:|---:|---:|---|']
    for d in DEPTHS:
        first = True
        p2, p1, tps = [], [], []
        for table, loads, src in ((T, DL, ''), (T2, L2, f' ({A.second_name})')):
            for ld in loads:
                c, c1, c2 = (Cell(table, ld, d, p) for p in (('1', '2'), ('1',), ('2',)))
                if not (c.ok and c1.ok and c2.ok):
                    continue
                md.append(f'| {depth_label(d) if first else ""} | `{ld}`{src} | {f2(c1.tps)}, {f2(c2.tps)} → **{f2(c.tps)}** '
                          f'({f2(c.ms_token)}) | {f2(c1.share, 3)} | {f2(c1.tok_step)} | {f2(c1.ms_step)} | '
                          f'{f2(c2.ms_step)} | {c1.replies[0][:6]} |')
                first = False
                p1.append(c1.ms_step)
                p2.append(c2.ms_step)
                tps.append(c.tps)
        if len(p2) > 1:
            md.append(f'| | all {len(p2)} loads | fastest − slowest: {f2(spread(tps))} tok/s | | | {f2(spread(p1))} | '
                      f'{f2(spread(p2))} | |')
    md.append('')

for src, table, arms in (('run', T, {A.draft: DL, A.plain: PL}), ('second', T2, {arm2: L2} if L2 else {})):
    for arm, loads in arms.items():
        for ld in loads:
            for d in DEPTHS:
                for name, passes in PASS_SETS:
                    c = Cell(table, ld, d, passes)
                    if c.ok:
                        load_lines.append([src, arm, ld, d, name, f'{c.n:.0f}', f2(c.ms, 3), f2(c.tps, 3), f2(c.ms_token, 3),
                                           f'{c.drafted:.0f}', f'{c.accepted:.0f}', f2(c.share, 4), f'{c.steps:.0f}',
                                           f2(c.tok_step, 4), f2(c.ms_step, 3), ' '.join(c.replies)])

text = '\n'.join(md)
print(text)
if A.out:
    with open(A.out, 'w') as f:
        f.write(text + '\n')
if A.tsv:
    for suffix, head, lines in (
            ('loads', 'source\tarm\tload\tdepth\tpass\tgen_n\tgen_ms\ttok_s\tms_token\tdrafted\taccepted\tshare\tsteps'
                      '\ttokens_step\tms_step\treply_md5', load_lines),
            ('classes', 'metric\tdepth\tarm\tcontrol\tloads\tcontrol_loads\tspread\tdelta\tpercent\tclass', classes),
            ('breakeven', 'depth\tpass\tstep_ms\tplain_token_ms\tbreakeven_tokens_step\tdrafted_step\tbreakeven_share'
                          '\ttokens_step\tshare', breakeven),
            ('prompt', 'depth\tread\tcached\ttok_s\tplain_tok_s\tspread\tdelta\tpercent\tclass\tms_prompt_token'
                       '\tplain_ms_prompt_token\tdraft_adds_ms\tshare_of_prompt_time_pct', prompt_lines),
            ('turn', 'depth\tdraft_adds_ms_prompt_token\tyield\ttokens_step\tsaved_ms_generated_token'
                     '\tprompt_tokens_per_generated_token', turn_lines)):
        with open(f'{A.tsv}_{suffix}.tsv', 'w') as f:
            f.write(head + '\n')
            for line in lines:
                f.write('\t'.join(str(x) for x in line) + '\n')
