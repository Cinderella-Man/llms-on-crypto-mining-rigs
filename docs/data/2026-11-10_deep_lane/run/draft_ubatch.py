#!/usr/bin/env python3
"""draft_ubatch.py: a deep-lane run of a draft line at several settings against the same line without the draft,
read on prompt reading and on the host's memory. Standard library only.

    python3 -I draft_ubatch.py OUT [--control draft] [--floor nodraft] [--out draft_ubatch.md] [--tsv PREFIX]

OUT is a results folder of bench/deep_lane/deep_lane.sh (requests.csv, loads.tsv, fit_<load>.txt and
ram_watchdog-<load>.tsv). --control is the draft line as the preset has it, --floor the same line without the draft;
every other arm is a setting of the draft line. Per depth this prints:
- prompt reading: tok/s and ms a prompt token of every load (the prime request, which reads a chunk on top of the
  previous depth's cache; at depth 0 the first decode request, which reads the tail), each arm's spread, the class
  of every arm against the control by bench/deep_lane's rule (gain when the arm's mean is above the control's by more
  than the larger of the two spreads, loss when below by more than it, else negligible), what the draft adds per
  prompt token over the floor (the arm's mean ms a prompt token minus the floor's), and the share of the control's
  cost an arm takes away (the control's cost minus the arm's, over the control's);
- the same for all primes of a load together (their tokens over their time) and their seconds;
- the host's memory per load, from the RAM watchdog's samples (one a second) and the check after loading:
  MemAvailable after loading, the lowest MemAvailable and usable memory (MemAvailable minus CmaFree), the most swap in
  use, the swap that moved over the load and during its primes, the most written to swap within 10 s (one of the
  watchdog's limits), the highest pressure, and whether the watchdog stopped the server; then each card's memory
  after loading;
- generation as a check: tok/s of both passes, the draft tokens accepted of drafted and the reply's hash per load.
--tsv PREFIX writes PREFIX_prompt.tsv, PREFIX_memory.tsv and PREFIX_generation.tsv.
"""
import argparse
import csv
import datetime
import os
import re
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('src')
ap.add_argument('--control', default='draft')
ap.add_argument('--floor', default='nodraft')
ap.add_argument('--out', default='')
ap.add_argument('--tsv', default='')
A = ap.parse_args()
MIN_READ = 32  # prompt tokens a request must have read for its prompt speed to count (deep_table.py's rule)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def mean(v):
    return sum(v) / len(v)


def spread(v):
    return max(v) - min(v) if len(v) > 1 else None


def verdict(ctl, arm, lower=False):
    """(spread, delta, percent, class) of an arm's load values against the control's; lower: a time, less is a gain."""
    if not ctl or not arm:
        return None, None, None, '—'
    delta = mean(arm) - mean(ctl)
    pct = 100 * delta / mean(ctl)
    if len(ctl) < 2 or len(arm) < 2:
        return None, delta, pct, 'no class (one load)'
    s = max(spread(ctl), spread(arm))
    better = -delta if lower else delta
    return s, delta, pct, 'gain' if better > s else 'loss' if better < -s else 'negligible'


def f2(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def sign(v, nd=2, unit=''):
    return '—' if v is None else f'{v:+.{nd}f}{unit}'.replace('-', '−')


def depth_label(d):
    return f'{d:,}'


with open(os.path.join(A.src, 'requests.csv'), newline='') as f:
    rows = list(csv.DictReader(f))
loads = {}
with open(os.path.join(A.src, 'loads.tsv'), newline='') as f:
    for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
        loads[r['load']] = r
timed = [n for n, ld in loads.items() if ld['timed'] == '1']
arms = list(dict.fromkeys(loads[n]['arm'] for n in timed))
for a in (A.control, A.floor):
    if a not in arms:
        sys.exit(f'no arm {a} among {", ".join(arms)}')
others = [a for a in arms if a not in (A.control, A.floor)]
order = [A.control] + others + [A.floor]
done = {n for n in timed if os.path.exists(os.path.join(A.src, f'load-{n}.done'))}
by_arm = {a: [n for n in timed if loads[n]['arm'] == a] for a in order}
R = {(r['load'], r['kind'], r['pass'], int(r['depth'])): r for r in rows if r['load'] in timed}
depths = sorted({k[3] for k in R})


def prompt_row(ld, d):
    r = R.get((ld, 'prime', '0', d)) or R.get((ld, 'decode', '1', d))
    return r if r and num(r['prompt_n']) >= MIN_READ and num(r['prompt_ms']) > 0 else None


def read_tps(rs):
    n, ms = sum(num(r['prompt_n']) for r in rs), sum(num(r['prompt_ms']) for r in rs)
    return n * 1000 / ms if n and ms else None


md = []
prompt_lines, memory_lines, gen_lines = [], [], []

# 1. prompt reading per depth, and all primes together
md += ['### Prompt reading: tok/s (ms a prompt token)', '',
       f'Per load the tokens its prompt request read over their time. Class of each arm against `{A.control}` by the '
       f'deep lane\'s rule. The draft adds: the arm\'s mean ms a prompt token minus `{A.floor}`\'s. Taken away: the '
       f'share of what the draft adds in `{A.control}` that the arm does not add.', '',
       '| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `' + A.control +
       '` | Class | The draft adds, ms a prompt token | Taken away |',
       '|---:|---|---|---|---|---:|---:|---|---:|---:|']


def prompt_block(label, read_label, per_load):
    """per_load: {load: (tok/s, ms a prompt token)} for the loads that have a value"""
    vals = {a: [per_load[n][0] for n in by_arm[a] if n in per_load] for a in order}
    ms = {a: [per_load[n][1] for n in by_arm[a] if n in per_load] for a in order}
    floor_ms = mean(ms[A.floor]) if ms[A.floor] else None
    ctl_cost = mean(ms[A.control]) - floor_ms if ms[A.control] and floor_ms is not None else None
    first = True
    for a in order:
        cells = [f'{per_load[n][0]:.1f} ({per_load[n][1]:.2f})' if n in per_load else
                 ('stopped' if n not in done else '—') for n in by_arm[a]]
        cells += [''] * (2 - len(cells))
        s = spread(vals[a])
        if a == A.control:
            cls = ('', '', '')
        else:
            sp, delta, pct, c = verdict(vals[A.control], vals[a])
            cls = (sign(pct, 1, ' %'), c, '')
        cost = mean(ms[a]) - floor_ms if ms[a] and floor_ms is not None and a != A.floor else None
        taken = (ctl_cost - cost) / ctl_cost if cost is not None and ctl_cost and a not in (A.control, A.floor) else None
        md.append(f'| {label if first else ""} | {read_label if first else ""} | {a} | {cells[0]} | {cells[1]} | '
                  f'{f2(s, 1)} | {cls[0]} | {cls[1]} | {sign(cost) if cost is not None else ""} | '
                  f'{f"{100 * taken:.1f} %".replace("-", "−") if taken is not None else ""} |')
        prompt_lines.append([label.replace(',', '').replace('**', ''), a,
                             ' '.join(f'{per_load[n][0]:.2f}' if n in per_load else 'stopped' for n in by_arm[a]),
                             ' '.join(f'{per_load[n][1]:.3f}' if n in per_load else 'stopped' for n in by_arm[a]),
                             f2(s), cls[0].replace('−', '-').replace(' %', ''), cls[1] or 'control',
                             f2(cost, 3) if cost is not None else '', f2(100 * taken, 1) if taken is not None else ''])
        first = False


for d in depths:
    per_load, read_label = {}, ''
    for n in timed:
        r = prompt_row(n, d)
        if r:
            per_load[n] = (num(r['prompt_n']) * 1000 / num(r['prompt_ms']), num(r['prompt_ms']) / num(r['prompt_n']))
            read_label = read_label or f'{int(num(r["prompt_n"])):,} on {int(num(r["cache_n"])):,}'
    if per_load:
        prompt_block(depth_label(d), read_label, per_load)
per_load, seconds, tokens = {}, {}, {}
for n in timed:
    rs = [R.get((n, 'prime', '0', d)) for d in depths]
    rs = [r for r in rs if r and num(r['prompt_n']) >= MIN_READ]
    if len(rs) == len([d for d in depths if d > 0]) and rs:
        t = read_tps(rs)
        per_load[n] = (t, 1000 / t)
        seconds[n] = sum(num(r['prompt_ms']) for r in rs) / 1000
        tokens[n] = int(sum(num(r['prompt_n']) for r in rs))
if per_load:
    prompt_block('all primes', f'{next(iter(tokens.values())):,} tokens', per_load)
    md += ['', 'The primes of a load together, seconds: ' + '; '.join(
        f'`{a}` ' + ' and '.join(f'{seconds[n]:.1f}' for n in by_arm[a] if n in seconds) for a in order) + '.']

# 2. the host's memory
md += ['', '### The host\'s memory per load', '',
       'From the RAM watchdog\'s samples, one a second, over the whole load (from the server\'s start to its stop); '
       'usable is MemAvailable minus CmaFree. Swap moved: the change in swap in use over the load, and the sum of '
       'its rises during the prime requests. The most written to swap in 10 s is one of the watchdog\'s own limits (by '
       'default it stops the server above 600 MiB).', '',
       '| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | '
       'Swap moved over the load | Swap moved during the primes | Most written to swap in 10 s | '
       'Highest pressure, some / full | The watchdog |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
cards = {}
for n in [n for a in order for n in by_arm[a]]:
    path = os.path.join(A.src, f'ram_watchdog-{n}.tsv')
    samples, kill = [], ''
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line[:1].isdigit():
                    samples.append([float(x) for x in line.split('\t')[:8]])
                elif 'KILL' in line:
                    kill = line.split('KILL', 1)[1].strip(' :\n')
    avail_fit = ''
    fit = os.path.join(A.src, f'fit_{n}.txt')
    if os.path.exists(fit):
        with open(fit) as f:
            text = f.read()
        m = re.search(r'MemAvailable:\s+(\d+) kB', text)
        avail_fit = f'{int(m.group(1)) / 1024:,.0f}' if m else ''
        cards[n] = [int(x.group(1)) for x in re.finditer(r'^\d+, (\d+), \d+$', text, re.M)]
    if not samples:
        md.append(f'| `{n}` | {loads[n]["arm"]} | {avail_fit} | — | — | — | — | — | — | — | no samples |')
        continue
    prime_moved = 0.0
    for d in depths:
        r = R.get((n, 'prime', '0', d))
        if not r:
            continue
        end = datetime.datetime.strptime(r['time'], '%Y-%m-%dT%H:%M:%S%z').timestamp()
        win = [s for s in samples if end - num(r['wall_s']) - 1 <= s[0] <= end + 1]
        prime_moved += sum(max(0.0, b[4] - a[4]) for a, b in zip(win, win[1:]))
    vals = [min(s[1] for s in samples), min(s[3] for s in samples), max(s[4] for s in samples),
            samples[-1][4] - samples[0][4], prime_moved, max(s[6] for s in samples), max(s[7] for s in samples),
            max(s[5] for s in samples)]
    wd = f'stopped the server: {kill}' if kill else 'did not act'
    md.append(f'| `{n}` | {loads[n]["arm"]} | {avail_fit} | {vals[0]:,.0f} | {vals[1]:,.0f} | {vals[2]:,.0f} | '
              f'{sign(vals[3], 0)} | +{vals[4]:,.0f} | {vals[7]:,.0f} | {vals[5]:.2f} / {vals[6]:.2f} % | {wd} |')
    memory_lines.append([n, loads[n]['arm'], avail_fit.replace(',', ''), f'{vals[0]:.0f}', f'{vals[1]:.0f}',
                         f'{vals[2]:.0f}', f'{vals[3]:+.0f}', f'{vals[4]:.0f}', f'{vals[7]:.0f}', f'{vals[5]:.2f}', f'{vals[6]:.2f}',
                         kill or 'none'])
if cards:
    md += ['', 'Each card\'s memory after loading, MiB:', '',
           '| Load | Arm | ' + ' | '.join(f'Card {i}' for i in range(max(len(c) for c in cards.values()))) + ' | All |',
           '|---|---|' + '---:|' * (max(len(c) for c in cards.values()) + 1)]
    for n, c in cards.items():
        md.append(f'| `{n}` | {loads[n]["arm"]} | ' + ' | '.join(f'{x:,}' for x in c) + f' | {sum(c):,} |')

# 3. generation, a check
md += ['', '### Generation, a check', '',
       'Per load and depth: tok/s on pass 1 and pass 2, the draft tokens accepted of drafted on both, and the first '
       'six characters of the reply\'s hash on pass 1 (a star when pass 2 wrote another reply).', '',
       '| Depth (tokens) | ' + ' | '.join(f'`{n}`' for a in order for n in by_arm[a]) + ' |',
       '|---:|' + '---|' * sum(len(by_arm[a]) for a in order)]
for d in depths:
    cells = []
    for n in [n for a in order for n in by_arm[a]]:
        p1, p2 = R.get((n, 'decode', '1', d)), R.get((n, 'decode', '2', d))
        if not p1 or not p2:
            cells.append('—')
            continue
        t = [num(p['gen_n']) * 1000 / num(p['gen_ms']) if num(p['gen_ms']) else 0 for p in (p1, p2)]
        acc = int(num(p1['draft_accepted']) + num(p2['draft_accepted']))
        dr = int(num(p1['draft_n']) + num(p2['draft_n']))
        h = (p1['ids_md5'] or p1['text_md5'])[:6]
        star = '' if (p1['ids_md5'] or p1['text_md5']) == (p2['ids_md5'] or p2['text_md5']) else '*'
        cells.append(f'{t[0]:.2f}, {t[1]:.2f}; {acc}/{dr}; {h}{star}' if dr else f'{t[0]:.2f}, {t[1]:.2f}; {h}{star}')
        gen_lines.append([d, n, loads[n]['arm'], f'{t[0]:.2f}', f'{t[1]:.2f}', str(acc), str(dr), h, star and 'differs'])
    md.append(f'| {depth_label(d)} | ' + ' | '.join(cells) + ' |')

text = '\n'.join(md) + '\n'
print(text, end='')
if A.out:
    with open(A.out, 'w') as f:
        f.write(text)
if A.tsv:
    for suffix, head, lines in (
            ('prompt', 'depth\tarm\ttok_s\tms_prompt_token\tspread\tpercent_against_control\tclass'
                       '\tdraft_adds_ms_prompt_token\ttaken_away_pct', prompt_lines),
            ('memory', 'load\tarm\tavail_after_load_mib\tmin_avail_mib\tmin_usable_mib\tmax_swap_used_mib'
                       '\tswap_change_load_mib\tswap_moved_primes_mib\tmax_swapout_10s_mib\tmax_psi_some\tmax_psi_full\twatchdog_kill', memory_lines),
            ('generation', 'depth\tload\tarm\ttok_s_pass1\ttok_s_pass2\tdraft_accepted\tdraft_n\treply\tpass2', gen_lines)):
        with open(f'{A.tsv}_{suffix}.tsv', 'w') as f:
            f.write(head + '\n' + ''.join('\t'.join(str(x) for x in l) + '\n' for l in lines))
