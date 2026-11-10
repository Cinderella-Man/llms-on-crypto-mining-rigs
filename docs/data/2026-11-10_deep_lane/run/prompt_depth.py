#!/usr/bin/env python3
"""prompt_depth.py: prompt reading at depth of a deep-lane run's arms against its control, with the loads of other
runs on the same prompts beside them. Standard library only.

    python3 -I prompt_depth.py OUT [--control ARM] [--beside FOLDER:ARM[:NAME]]... [--floor NAME]
        [--out prompt_depth.md] [--tsv PREFIX]

OUT is a results folder of bench/deep_lane/deep_lane.sh (requests.csv, loads.tsv, fit_<load>.txt and
ram_watchdog-<load>.tsv). --control is the arm the others are classed against (default: the first timed arm).
--beside adds the timed loads of ARM from another results folder of the same prompts, under NAME (default ARM). A
beside arm is shown at a depth only where its request read the same chunk on the same cached tokens as this run's
control, and it gets no class: it ran in another run. --floor names one beside arm, the line the arms are measured
toward (for example the line without the draft): at each depth above 0 where it is shown, the share of the gap
between the control and the floor that an arm closes, (control's ms a prompt token - arm's) / (control's - floor's),
from each arm's mean over its loads. Depth 0 reads only the 64-token tail and gets no gap.

Per depth this prints:
- prompt reading: tok/s and ms a prompt token of every load (the prime request, which reads a chunk on top of the
  previous depth's cache; at depth 0 the first decode request, which reads the tail), each arm's spread, the arm
  against the control in percent and its class by bench/deep_lane's rule (a gain when the arm's mean is above the
  control's by more than the larger of the two spreads, a loss when below by more than it, else negligible); the
  tokens read and the tokens cached are printed again for an arm that read another chunk than the control (another
  micro-batch moves the checkpoint the next prime continues from);
- all primes of a load together: their tokens over their time, and their seconds;
- the host's memory per load, from the RAM watchdog's samples (one a second) and the check after loading: MemAvailable
  after loading, the lowest MemAvailable and usable memory (MemAvailable minus CmaFree), the most swap in use, the swap
  moved during the primes, the most written to swap within 10 s (one of the watchdog's limits), the highest pressure,
  and whether the watchdog stopped the server; then each card's memory after loading;
- generation as a check: per load and depth tok/s on pass 1 and pass 2, the ms of a step on each pass (the
  generation time over the steps, a step being one token of the main model's own plus the draft tokens it accepts),
  the draft tokens accepted of drafted on both passes, and the reply's hash.
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
ap.add_argument('--control', default='')
ap.add_argument('--beside', action='append', default=[])
ap.add_argument('--floor', default='')
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


def verdict(ctl, arm):
    """(spread, percent, class) of an arm's load values (tok/s) against the control's"""
    if not ctl or not arm:
        return None, None, '—'
    pct = 100 * (mean(arm) - mean(ctl)) / mean(ctl)
    if len(ctl) < 2 or len(arm) < 2:
        return None, pct, 'no class (one load)'
    s = max(spread(ctl), spread(arm))
    delta = mean(arm) - mean(ctl)
    return s, pct, 'gain' if delta > s else 'loss' if delta < -s else 'negligible'


def f2(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def sign(v, nd=1, unit=''):
    return '—' if v is None else f'{v:+.{nd}f}{unit}'.replace('-', '−')


def read_run(src):
    """(rows by (load, kind, pass, depth), loads.tsv rows by load, timed loads in order, finished loads)"""
    with open(os.path.join(src, 'requests.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    loads = {}
    with open(os.path.join(src, 'loads.tsv'), newline='') as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            loads[r['load']] = r
    timed = [n for n, ld in loads.items() if ld['timed'] == '1']
    done = {n for n in timed if os.path.exists(os.path.join(src, f'load-{n}.done'))}
    R = {(r['load'], r['kind'], r['pass'], int(r['depth'])): r for r in rows if r['load'] in timed}
    return R, loads, timed, done


def prompt_row(R, ld, d):
    r = R.get((ld, 'prime', '0', d)) or R.get((ld, 'decode', '1', d))
    return r if r and num(r['prompt_n']) >= MIN_READ and num(r['prompt_ms']) > 0 else None


R, loads, timed, done = read_run(A.src)
arms = list(dict.fromkeys(loads[n]['arm'] for n in timed))
control = A.control or arms[0]
if control not in arms:
    sys.exit(f'no arm {control} among {", ".join(arms)}')
order = [control] + [a for a in arms if a != control]
by_arm = {a: [n for n in timed if loads[n]['arm'] == a] for a in order}
depths = sorted({k[3] for k in R})

beside = []  # (name, folder, arm, R, its loads of the arm)
for spec in A.beside:
    parts = spec.split(':')
    if len(parts) < 2:
        sys.exit(f'--beside {spec}: FOLDER:ARM[:NAME]')
    folder, arm = parts[0], parts[1]
    name = parts[2] if len(parts) > 2 and parts[2] else arm
    bR, bloads, btimed, _ = read_run(folder)
    lds = [n for n in btimed if bloads[n]['arm'] == arm]
    if not lds:
        sys.exit(f'--beside {spec}: no timed load of {arm} in {folder}')
    beside.append((name, folder, arm, bR, lds))
if A.floor and A.floor not in [b[0] for b in beside]:
    sys.exit(f'--floor {A.floor}: not a --beside name')

md = []
prompt_lines, memory_lines, gen_lines = [], [], []

# 1. prompt reading per depth
md += ['### Prompt reading: tok/s (ms a prompt token)', '',
       f'Per load, the tokens its prompt request read over their time. Each arm of this run is classed against '
       f'`{control}` by the deep lane\'s rule.' +
       (' Beside: the loads of another run that read the same chunk on the same cached tokens, not classed.'
        if beside else '') +
       (f' Gap closed: the share of the gap between `{control}` and `{A.floor}` in ms a prompt token that the arm '
        f'closes.' if A.floor else ''), '',
       '| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `' + control + '` | Class |'
       + (' Gap closed |' if A.floor else ''),
       '|---:|---|---|---|---|---:|---:|---|' + ('---:|' if A.floor else '')]
skipped = []
for d in depths:
    ctl_rows = [prompt_row(R, n, d) for n in by_arm[control]]
    ctl_rows = [r for r in ctl_rows if r]
    if not ctl_rows:
        continue
    key = (ctl_rows[0]['prompt_n'], ctl_rows[0]['cache_n'] or '0')
    label = f'{int(num(key[0])):,} on {int(num(key[1])):,}'
    series = []  # (name, kind, [(load, tok/s, ms)], read label)
    for a in order:
        vals, own = [], ''
        for n in by_arm[a]:
            r = prompt_row(R, n, d)
            vals.append((n, num(r['prompt_n']) * 1000 / num(r['prompt_ms']), num(r['prompt_ms']) / num(r['prompt_n']))
                        if r else (n, None, None))
            if r and not own:
                own = f'{int(num(r["prompt_n"])):,} on {int(num(r["cache_n"])):,}'
        series.append((a, 'arm', vals, own or label))
    for name, folder, arm, bR, lds in beside:
        vals, other = [], None
        for n in lds:
            r = prompt_row(bR, n, d)
            if r and (r['prompt_n'], r['cache_n'] or '0') == key:
                vals.append((n, num(r['prompt_n']) * 1000 / num(r['prompt_ms']), num(r['prompt_ms']) / num(r['prompt_n'])))
            elif r:
                other = f'{int(num(r["prompt_n"])):,} on {int(num(r["cache_n"])):,}'
        if vals:
            series.append((name, 'beside', vals, label))
        elif other:
            skipped.append(f'{name} at {d:,} ({other})')
    ctl_tps = [v[1] for v in series[0][2] if v[1] is not None]
    ctl_ms = [v[2] for v in series[0][2] if v[2] is not None]
    floor = next((s for s in series if s[0] == A.floor and s[1] == 'beside'), None)
    floor_ms = mean([v[2] for v in floor[2]]) if floor else None
    first = True
    for name, kind, vals, own in series:
        tps = [v[1] for v in vals if v[1] is not None]
        ms = [v[2] for v in vals if v[2] is not None]
        cells = [f'{v[1]:.1f} ({v[2]:.2f})' if v[1] is not None else ('stopped' if kind == 'arm' and v[0] not in done
                                                                         else '—') for v in vals]
        cells += [''] * (2 - len(cells))
        extra = f' ({", ".join(cells[2:])} more)' if len(cells) > 2 else ''
        s = spread(tps)
        if name == control and kind == 'arm':
            pct, cls = None, 'control'
        elif kind == 'arm':
            _, pct, cls = verdict(ctl_tps, tps)
        else:
            pct, cls = (100 * (mean(tps) - mean(ctl_tps)) / mean(ctl_tps) if tps and ctl_tps else None), 'beside'
        gap = None
        if (d > 0 and floor_ms is not None and ms and ctl_ms and kind == 'arm' and name != control
                and mean(ctl_ms) != floor_ms):
            gap = (mean(ctl_ms) - mean(ms)) / (mean(ctl_ms) - floor_ms)
        shown = name if kind == 'arm' else f'{name} (beside)'
        md.append(f'| {f"{d:,}" if first else ""} | {own if first or own != label else ""} | {shown} | {cells[0]} | '
                  f'{cells[1]}{extra} | '
                  f'{f2(s, 1)} | {"" if pct is None else sign(pct, 1, " %")} | {"" if cls == "control" else cls} |'
                  + (f' {"" if gap is None else f"{100 * gap:.1f} %".replace("-", "−")} |' if A.floor else ''))
        prompt_lines.append([d, own.replace(',', ''), name, kind,
                             ' '.join(f'{v[1]:.2f}' if v[1] is not None else 'none' for v in vals),
                             ' '.join(f'{v[2]:.3f}' if v[2] is not None else 'none' for v in vals),
                             f2(s), '' if pct is None else f'{pct:.2f}', cls, '' if gap is None else f'{100 * gap:.1f}'])
        first = False
if skipped:
    md += ['', 'Not shown, as they read another chunk: ' + '; '.join(skipped) + '.']

# all primes of a load together
per_load = {}
for a in order:
    for n in by_arm[a]:
        rs = [R.get((n, 'prime', '0', d)) for d in depths if d > 0]
        rs = [r for r in rs if r and num(r['prompt_n']) >= MIN_READ]
        if rs and len(rs) == len([d for d in depths if d > 0]):
            tok = sum(num(r['prompt_n']) for r in rs)
            ms = sum(num(r['prompt_ms']) for r in rs)
            per_load[n] = (tok * 1000 / ms, ms / 1000, int(tok))
if per_load:
    md += ['', 'All primes of a load together: the tokens they read, and tok/s (seconds):', '',
           '| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `' + control + '` | Class |',
           '|---|---:|---|---|---:|---:|---|']
    ctl_tps = [per_load[n][0] for n in by_arm[control] if n in per_load]
    for a in order:
        tps = [per_load[n][0] for n in by_arm[a] if n in per_load]
        tokens = ' and '.join(dict.fromkeys(f'{per_load[n][2]:,}' for n in by_arm[a] if n in per_load))
        cells = [f'{per_load[n][0]:.1f} ({per_load[n][1]:.1f} s)' if n in per_load else
                 ('stopped' if n not in done else '—') for n in by_arm[a]]
        cells += [''] * (2 - len(cells))
        if a == control:
            pct, cls = None, ''
        else:
            _, pct, cls = verdict(ctl_tps, tps)
        md.append(f'| {a} | {tokens} | {cells[0]} | {cells[1]} | {f2(spread(tps), 1)} | '
                  f'{"" if pct is None else sign(pct, 1, " %")} | {cls} |')
        prompt_lines.append(['all', tokens.replace(',', ''), a, 'arm', ' '.join(f'{per_load[n][0]:.2f}' for n in by_arm[a] if n in per_load),
                             ' '.join(f'{per_load[n][1]:.1f}' for n in by_arm[a] if n in per_load), f2(spread(tps)),
                             '' if pct is None else f'{pct:.2f}', cls or 'control', ''])

# 2. the host's memory
md += ['', '### The host\'s memory per load', '',
       'From the RAM watchdog\'s samples, one a second, over the whole load (from the server\'s start to its stop); '
       'usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a '
       'prime request ran. The most written to swap in 10 s is one of the watchdog\'s own limits (by default it stops '
       'the server above 600 MiB).', '',
       '| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | '
       'Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
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
        md.append(f'| `{n}` | {loads[n]["arm"]} | {avail_fit} | — | — | — | — | — | — | no samples |')
        continue
    prime_moved = 0.0
    for d in depths:
        r = R.get((n, 'prime', '0', d))
        if not r:
            continue
        end = datetime.datetime.strptime(r['time'], '%Y-%m-%dT%H:%M:%S%z').timestamp()
        win = [s for s in samples if end - num(r['wall_s']) - 1 <= s[0] <= end + 1]
        prime_moved += sum(max(0.0, b[4] - a[4]) for a, b in zip(win, win[1:]))
    vals = [min(s[1] for s in samples), min(s[3] for s in samples), max(s[4] for s in samples), prime_moved,
            max(s[6] for s in samples), max(s[7] for s in samples), max(s[5] for s in samples)]
    wd = f'stopped the server: {kill}' if kill else 'did not act'
    md.append(f'| `{n}` | {loads[n]["arm"]} | {avail_fit} | {vals[0]:,.0f} | {vals[1]:,.0f} | {vals[2]:,.0f} | '
              f'+{vals[3]:,.0f} | {vals[6]:,.0f} | {vals[4]:.2f} / {vals[5]:.2f} % | {wd} |')
    memory_lines.append([n, loads[n]['arm'], avail_fit.replace(',', ''), f'{vals[0]:.0f}', f'{vals[1]:.0f}',
                         f'{vals[2]:.0f}', f'{vals[3]:.0f}', f'{vals[6]:.0f}', f'{vals[4]:.2f}', f'{vals[5]:.2f}',
                         kill or 'none'])
if cards:
    width = max(len(c) for c in cards.values())
    md += ['', 'Each card\'s memory after loading, MiB:', '',
           '| Load | Arm | ' + ' | '.join(f'Card {i}' for i in range(width)) + ' | All |',
           '|---|---|' + '---:|' * (width + 1)]
    for n, c in cards.items():
        md.append(f'| `{n}` | {loads[n]["arm"]} | ' + ' | '.join(f'{x:,}' for x in c) + f' | {sum(c):,} |')

# 3. generation, a check
md += ['', '### Generation, a check', '',
       'Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of '
       'drafted on both passes; the first six characters of the reply\'s hash on pass 1 (a star when pass 2 wrote '
       'another reply).', '',
       '| Depth (tokens) | ' + ' | '.join(f'`{n}`' for a in order for n in by_arm[a]) + ' |',
       '|---:|' + '---|' * sum(len(by_arm[a]) for a in order)]


def step_ms(p):
    steps = num(p['gen_n']) - num(p['draft_accepted'])
    return num(p['gen_ms']) / steps if steps > 0 else 0.0


for d in depths:
    cells = []
    for n in [n for a in order for n in by_arm[a]]:
        p1, p2 = R.get((n, 'decode', '1', d)), R.get((n, 'decode', '2', d))
        if not p1 or not p2:
            cells.append('—')
            continue
        t = [num(p['gen_n']) * 1000 / num(p['gen_ms']) if num(p['gen_ms']) else 0 for p in (p1, p2)]
        st = [step_ms(p) for p in (p1, p2)]
        acc = int(num(p1['draft_accepted']) + num(p2['draft_accepted']))
        dr = int(num(p1['draft_n']) + num(p2['draft_n']))
        h1, h2 = p1['ids_md5'] or p1['text_md5'], p2['ids_md5'] or p2['text_md5']
        star = '' if h1 == h2 else '*'
        cells.append(f'{t[0]:.2f}, {t[1]:.2f}; {st[0]:.1f}, {st[1]:.1f} ms; {acc}/{dr}; {h1[:6]}{star}')
        gen_lines.append([d, n, loads[n]['arm'], f'{t[0]:.2f}', f'{t[1]:.2f}', f'{st[0]:.2f}', f'{st[1]:.2f}', str(acc),
                          str(dr), h1[:6], star and 'differs'])
    md.append(f'| {d:,} | ' + ' | '.join(cells) + ' |')

text = '\n'.join(md) + '\n'
print(text, end='')
if A.out:
    with open(A.out, 'w') as f:
        f.write(text)
if A.tsv:
    for suffix, head, lines in (
            ('prompt', 'depth\tread_on_cached\tarm\tkind\ttok_s\tms_prompt_token\tspread\tpercent_against_control\tclass'
                       '\tgap_closed_pct', prompt_lines),
            ('memory', 'load\tarm\tavail_after_load_mib\tmin_avail_mib\tmin_usable_mib\tmax_swap_used_mib'
                       '\tswap_moved_primes_mib\tmax_swapout_10s_mib\tmax_psi_some\tmax_psi_full\twatchdog_kill',
             memory_lines),
            ('generation', 'depth\tload\tarm\ttok_s_pass1\ttok_s_pass2\tms_step_pass1\tms_step_pass2\tdraft_accepted'
                           '\tdraft_n\treply\tpass2', gen_lines)):
        with open(f'{A.tsv}_{suffix}.tsv', 'w') as f:
            f.write(head + '\n' + ''.join('\t'.join(str(x) for x in l) + '\n' for l in lines))
