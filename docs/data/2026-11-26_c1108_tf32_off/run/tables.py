#!/usr/bin/env python3
"""C1108's tables from run.sh's WORK folder (standard library only).

    python3 -I tables.py <data folder> <WORK> [<P106 logits.tsv>]

<P106 logits.tsv>: docs/data/2026-11-24_c912_p106_check/logits.tsv, whose card_vs_cpu_nmse_ref column is each random
model's card-against-CPU nmse on a P106 (af00bb13b's build; 2d839da1d's CPU logits are af00bb13b's bit for bit).
Writes into the data folder:
  archs-runs.tsv     per test-llama-archs run: setting, repeat, exit status, summary, card rows OK / FAIL / SKIP, CPU
                     logits files written, compared and differing (against the first run of its pair)
  archs-nmse.tsv     per random model: the card-against-CPU nmse ("NMSE vs. CPU") and the mixed-batch nmse of every
                     run, the P106's card-against-CPU nmse, and whether each setting's repeat gives its first number
  archs-summary.tsv  per setting: models with a card number, the largest nmse and its model, the median, the models
                     at or below 1e-12, and against the P106 per model: the geometric mean of the ratio, the largest
                     ratio and its model, the models within 10 times the P106's number
  replay.tsv         per MUL_MAT replay (tags before, tf32, after): exit status, passed line, cases, OK, FAIL, NOREC,
                     the log's "not supported" lines, seconds
  replay-cases.tsv   the cases whose verdict differs between the replays (header only when none does)
  replay-errors.tsv  the cases by weight type, activation type and token count (1-8 or 9+): how many, and per tag the
                     median and largest error (the largest of a case's compared outputs) and how many cases' errors
                     differ from the before replay's
  pipe.tsv           per pipe check verdict line: arm, layout, arch, check, the verdict of each pass, whether the
                     verdicts and the plain run's tokens are the same in all, the plain run's logits hash of each pass
                     and whether it is the before pass's
"""
import math
import os
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
SETTINGS = ['stock', 'notf32', 'before', 'after', 'f32']
RUNS = [f'{s}-{r}' for r in (1, 2) for s in SETTINGS]
CMP = re.compile(r'^(\S+): logits against \S+: nmse (\S+), max abs diff (\S+), (\d+) of (\d+) differ')
WROTE = re.compile(r'logits_dump_or_compare: wrote \d+ logits to \S+/(?:plain|forced)-(\S+)\.f32')
# the cells after "Config": NMSE vs. CPU | Roundtrip | Mixed batch
VAL = re.compile(r'^\s*(OK|FAIL|SKIP)\s*(?:\(([0-9.e+-]+)\))?\s*\|\s*(\S+)\s*\|\s*(OK|FAIL|SKIP)\s*(?:\(([0-9.e+-]+)\))?')


def lines(path):
    if not os.path.exists(path):
        return []
    return ANSI.sub('', open(path, errors='replace').read()).splitlines()


def card_rows(path):
    """(architecture, Dense|MoE) -> (nmse vs CPU, mixed batch nmse), each a number string, 'SKIP' or 'FAIL <n>'.

    A row whose cells a log line interrupted continues on a later line (C1039's parser)."""
    out, pending = {}, None

    def val(v, n):
        return 'SKIP' if v == 'SKIP' else (n if v == 'OK' else f'FAIL {n}')

    for line in lines(path):
        if line.startswith('|') and 'CUDA0' in line:
            cells = line.split('|')
            key = (cells[1].strip(), cells[3].strip())
            m = VAL.match('|'.join(cells[4:]))
            if m:
                out.setdefault(key, (val(m.group(1), m.group(2)), val(m.group(4), m.group(5))))
                pending = None
            else:
                pending = key
            continue
        m = VAL.match(line)
        if m and pending:
            out.setdefault(pending, (val(m.group(1), m.group(2)), val(m.group(4), m.group(5))))
            pending = None
    return out


def dump_name(key, keys):
    """test-llama-archs' logits file name: '<arch>', or '<arch>-moe' for the MoE model of an architecture with both"""
    arch, kind = key
    return arch + '-moe' if kind == 'MoE' and (arch, 'Dense') in keys else arch


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write(dest, name, header, rows):
    with open(os.path.join(dest, name), 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(c) for c in r) + '\n')


def archs(dest, work, p106_path):
    a = os.path.join(work, 'archs')
    rc = {}
    for line in lines(os.path.join(a, 'rc.txt')):
        n, _, v = line.partition(' rc=')
        rc[n] = v
    rows, cards = [], {}
    for r in RUNS:
        log = os.path.join(a, 'logs', r + '.log')
        ls = lines(log)
        summ = next((x for x in ls if x.startswith('Summary')), '')
        c = card_rows(log)
        cards[r] = c
        vals = [v[0] for v in c.values()]
        cmp_ = [CMP.match(x.strip()) for x in ls]
        cmp_ = [m for m in cmp_ if m]
        wrote = sum(1 for x in ls if WROTE.search(x))
        rows.append([r, r.rsplit('-', 1)[0], r.rsplit('-', 1)[1], rc.get(r, '-'), summ,
                     sum(1 for v in vals if v not in ('SKIP',) and not v.startswith('FAIL')),
                     sum(1 for v in vals if v.startswith('FAIL')), sum(1 for v in vals if v == 'SKIP'),
                     wrote, len(cmp_), sum(1 for m in cmp_ if int(m.group(4)) > 0)])
    write(dest, 'archs-runs.tsv', ['run', 'setting', 'repeat', 'rc', 'summary', 'card_ok', 'card_fail', 'card_skip',
                                   'cpu_logits_written', 'cpu_logits_compared', 'cpu_logits_differing'], rows)

    p106 = {}
    if p106_path and os.path.exists(p106_path):
        hdr = None
        for line in open(p106_path):
            f = line.rstrip('\n').split('\t')
            if hdr is None:
                hdr = f
                continue
            p106[f[0]] = f[hdr.index('card_vs_cpu_nmse_ref')]

    keys = sorted(set().union(*[set(c) for c in cards.values()]))
    rows = []
    per = {s: [] for s in SETTINGS}  # (model, nmse, p106) per setting, first run
    same = {s: [0, 0] for s in SETTINGS}
    for k in keys:
        name = dump_name(k, keys)
        nm = [cards[r].get(k, ('-', '-'))[0] for r in RUNS]
        mx = [cards[r].get(k, ('-', '-'))[1] for r in RUNS]
        if all(v in ('SKIP', '-') for v in nm):
            continue
        rep = []
        for s in SETTINGS:
            v1, v2 = cards[f'{s}-1'].get(k, ('-',))[0], cards[f'{s}-2'].get(k, ('-',))[0]
            ok = v1 == v2
            rep.append('yes' if ok else 'no')
            same[s][0] += ok
            same[s][1] += 1
            if num(v1) is not None:
                per[s].append((name, num(v1), num(p106.get(name))))
        rows.append([name, k[0], k[1]] + nm + mx + [p106.get(name, '-')] + rep)
    write(dest, 'archs-nmse.tsv', ['model', 'arch', 'config'] + [f'nmse_{r}' for r in RUNS] +
          [f'mixed_{r}' for r in RUNS] + ['p106_nmse_c912'] + [f'repeat_same_{s}' for s in SETTINGS], rows)

    rows = []
    for s in SETTINGS:
        v = per[s]
        xs = sorted(x[1] for x in v)
        big = max(v, key=lambda x: x[1]) if v else ('-', 0)
        med = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2 if xs else 0
        rat = [(x[0], x[1] / x[2]) for x in v if x[2]]
        gm = math.exp(sum(math.log(r[1]) for r in rat) / len(rat)) if rat else 0
        rmax = max(rat, key=lambda r: r[1]) if rat else ('-', 0)
        rows.append([s, len(v), f'{big[1]:.3g}', big[0], f'{med:.3g}', sum(1 for x in xs if x <= 1e-12), len(rat),
                     f'{gm:.3g}', f'{rmax[1]:.3g}', rmax[0], sum(1 for r in rat if r[1] <= 10),
                     f'{same[s][0]} of {same[s][1]}'])
    write(dest, 'archs-summary.tsv', ['setting', 'models', 'largest_nmse', 'largest_model', 'median_nmse',
                                      'models_at_or_below_1e-12', 'models_with_p106', 'ratio_to_p106_geomean',
                                      'ratio_to_p106_largest', 'ratio_largest_model', 'models_within_10x_p106',
                                      'repeat_gives_first_number'], rows)


TAGS = ['before', 'tf32', 'after']


def replay(dest, work):
    """the MUL_MAT replays from their --replay-log files: index, case, verdict, each output's error"""
    o = os.path.join(work, 'bops')
    rc = {}
    for line in lines(os.path.join(o, 'rc.txt')):
        n, _, v = line.partition(' rc=')
        rc[n] = v
    rows, cases = [], {}
    for tag in TAGS:
        tsv = os.path.join(o, f'mul_mat-replay-{tag}.tsv')
        if not os.path.exists(tsv):
            continue
        c = {}
        for line in open(tsv):
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            errs = [num(x) for x in (f[3].split(',') if len(f) > 3 and f[3] else [])]
            errs = [x for x in errs if x is not None]
            c[f[0]] = (f[2], max(errs) if errs else None, f[1])
        cases[tag] = c
        ls = lines(os.path.join(o, f'mul_mat-replay-{tag}.log'))
        passed = next((x.strip() for x in ls if 'tests passed' in x), '')
        sec = (lines(os.path.join(o, f'mul_mat-replay-{tag}.sec')) or ['-'])[0]
        v = [x[0] for x in c.values()]
        rows.append([tag, rc.get(f'mul_mat-replay-{tag}', '-'), passed, len(c), v.count('OK'), v.count('FAIL'),
                     v.count('NOREC'), sum(1 for x in ls if 'not supported' in x), sec])
    write(dest, 'replay.tsv', ['tag', 'rc', 'passed_line', 'cases', 'ok', 'fail', 'norec', 'not_supported_lines',
                               'seconds'], rows)
    tags = [t for t in TAGS if t in cases]
    if len(tags) < 2:
        return
    keys = sorted(set().union(*[set(cases[t]) for t in tags]))
    # keyed by the case's index (a case string can repeat in the suite)
    diff = []
    for k in keys:
        v = [cases[t].get(k, ('-',))[0] for t in tags]
        if len(set(v)) > 1:
            name = next(cases[t][k][2] for t in tags if k in cases[t])
            diff.append([k, name] + v)
    write(dest, 'replay-cases.tsv', ['index', 'case'] + tags, diff)
    # errors by the weight's type and the token count (1-8: the matrix-vector kernels; 9+: cuBLAS or the quantized
    # kernels), per tag; error_changed counts the cases whose error differs from the first tag's
    groups = {}
    for k in keys:
        if not all(k in cases[t] and cases[t][k][1] is not None for t in tags):
            continue
        m = re.search(r'type_a=([a-z0-9_]+),type_b=([a-z0-9_]+),m=\d+,n=(\d+)', cases[tags[0]][k][2])
        if not m:
            continue
        g = (m.group(1), m.group(2), '1-8' if int(m.group(3)) <= 8 else '9+')
        groups.setdefault(g, []).append([cases[t][k][1] for t in tags])
    rows = []
    for g in sorted(groups):
        v = groups[g]
        row = list(g) + [len(v)]
        for i, t in enumerate(tags):
            xs = sorted(x[i] for x in v)
            row += [f'{xs[len(xs) // 2]:.3g}', f'{xs[-1]:.3g}']
            if i:
                row.append(sum(1 for x in v if x[i] != x[0]))
        rows.append(row)
    hdr = ['type_a', 'type_b', 'tokens', 'cases']
    for i, t in enumerate(tags):
        hdr += [f'median_{t}', f'largest_{t}'] + ([f'changed_{t}'] if i else [])
    write(dest, 'replay-errors.tsv', hdr, rows)


def pipe(dest, work):
    tabs = {}
    for tag in TAGS:
        p = os.path.join(work, f'pipe-{tag}', 'summary.tsv')
        if not os.path.exists(p):
            continue
        hdr, t = None, {}
        for line in open(p):
            f = line.rstrip('\n').split('\t')
            if hdr is None:
                hdr = f
                continue
            d = dict(zip(hdr, f))
            t[(d['arm'], d['layout'], d['arch'], d['check'])] = d
        tabs[tag] = t
    tags = [t for t in TAGS if t in tabs]
    if len(tags) < 2:
        return
    rows = []
    for k in sorted(set().union(*[set(tabs[t]) for t in tags])):
        d = [tabs[t].get(k, {}) for t in tags]
        row = list(k) + [x.get('verdict', '-') for x in d]
        row.append('yes' if len(set(x.get('verdict') for x in d)) == 1 else 'no')
        row.append('yes' if len(set(x.get('plain tokens hash') for x in d)) == 1 else 'no')
        row += [x.get('plain logits hash', '-') for x in d]
        row += ['yes' if d[i].get('plain logits hash') == d[0].get('plain logits hash') else 'no'
                for i in range(1, len(d))]
        rows.append(row)
    write(dest, 'pipe.tsv', ['arm', 'layout', 'arch', 'check'] + [f'verdict_{t}' for t in tags] +
          ['verdicts_same', 'plain_tokens_same'] + [f'plain_logits_{t}' for t in tags] +
          [f'logits_{t}_same_as_{tags[0]}' for t in tags[1:]], rows)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dest, work = sys.argv[1], sys.argv[2]
    archs(dest, work, sys.argv[3] if len(sys.argv) > 3 else None)
    replay(dest, work)
    pipe(dest, work)


if __name__ == '__main__':
    main()
