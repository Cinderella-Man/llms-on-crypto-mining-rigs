#!/usr/bin/env python3
"""C1157's tables from run.sh's outputs (or a copy of them).

Usage: python3 -I tables.py <W> <GW> <out folder>
  W   run.sh's work folder (rig/md5.txt, rig/bin-md5-*.txt, testonly/md5.txt, checks/sass/)
  GW  the GPU steps' folder (test/, again/, mm/bops/, fa/bops/, pipe/, flash/)
Writes to <out>: binaries.tsv, test.tsv, again.tsv, again_runs.tsv, replays.tsv, pipe.tsv, pipe_plain_hashes.tsv,
flash.tsv, sass.txt, and
prints one summary line per table.
"""
import csv
import json
import os
import re
import shutil
import sys


def read_tsv(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(p, header, rows):
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def md5s(p):
    out = {}
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            parts = line.split()
            if len(parts) == 2:
                out[parts[1].lstrip('*')] = parts[0]
    return out


def binaries(W, out):
    b = md5s(os.path.join(W, 'rig', 'bin-md5-base.txt'))
    h = md5s(os.path.join(W, 'rig', 'bin-md5-head.txt'))
    rows = []
    for n in sorted(set(b) | set(h)):
        same = 'yes' if b.get(n) == h.get(n) else ('only in branch' if n not in b else ('only in base' if n not in h else 'no'))
        rows.append([n, b.get(n, '-'), h.get(n, '-'), same])
    t = md5s(os.path.join(W, 'testonly', 'md5.txt'))
    for n, m in t.items():
        rows.append([n + ' (test-only build)', '-', m, 'test-only build'])
    write_tsv(os.path.join(out, 'binaries.tsv'), ['binary', 'base md5', 'branch md5', 'same'], rows)
    differ = sum(1 for r in rows if r[3] == 'no')
    print(f'binaries: {len(b)} in the base, {len(h)} in the branch, {differ} with other bytes, '
          f'only in the branch: {", ".join(r[0] for r in rows if r[3] == "only in branch") or "none"}')


def test(GW, out):
    p = os.path.join(GW, 'test', 'test.tsv')
    if not os.path.exists(p):
        print('test: no table')
        return
    rows = read_tsv(p)
    shutil.copy(p, os.path.join(out, 'test.tsv'))
    groups = {}
    for r in rows:
        k = (r['build'], r['setting'])
        g = groups.setdefault(k, [0, 0, set()])
        g[0] += 1
        g[1] += r['verdict'] == 'PASS'
        g[2].add(f"{r['right']} of {r['evaluations']}")
    print('test: ' + '; '.join(f'{b} {s} {g[1]} of {g[0]} runs pass ({", ".join(sorted(g[2]))} evaluations right)'
                               for (b, s), g in groups.items()))


def again(GW, out):
    p = os.path.join(GW, 'again', 'again.tsv')
    if not os.path.exists(p):
        print('again: no table')
        return
    rows = read_tsv(p)
    write_tsv(os.path.join(out, 'again_runs.tsv'), list(rows[0].keys()) if rows else [],
              [list(r.values()) for r in rows])
    groups = {}
    for r in rows:
        k = (r['build'], r['arm'])
        g = groups.setdefault(k, {'runs': 0, 'clean': 0, 'fails': {}, 'errs': set(), 'wc': set(), 'wr': set()})
        g['runs'] += 1
        if r['failing'] == 'none':
            g['clean'] += 1
        else:
            for c in r['failing'].split(';'):
                g['fails'][c] = g['fails'].get(c, 0) + 1
        if r['errors']:
            g['errs'].update(e for e in r['errors'].split(';') if e)
        g['wc'].add(r['warmups_complete'])
        g['wr'].add(r['warmups_reset'])
    tab = []
    for (b, a), g in groups.items():
        fails = '; '.join(f'{c} ({n} runs)' for c, n in sorted(g['fails'].items())) or 'none'
        tab.append([b, a, g['runs'], g['clean'], fails, ';'.join(sorted(g['errs'])) or '-',
                    ','.join(sorted(g['wc'])), ','.join(sorted(g['wr']))])
    write_tsv(os.path.join(out, 'again.tsv'), ['build', 'arm', 'runs', 'runs without a failing case', 'failing cases',
                                               'ERR values', 'warm-ups complete per run', 'warm-up resets per run'], tab)
    print('again: ' + '; '.join(f'{r[0]} {r[1]} {r[3]} of {r[2]} without a failing case' for r in tab))


SUM_RE = re.compile(r'^(?P<tag>[^:]+): (?P<rc>[^;]*); (?P<passed>[^;]*); OK (?P<ok>\d+); FAIL (?P<fail>\d+); '
                    r'NOREC (?P<norec>\d+); not supported (?P<ns>\d+); illegal (?P<ill>\d+); (?P<sec>.*) s$')


def replays(GW, out):
    rows = []
    # mm and fa: run.sh's own; *-rerun*: single runs again in their own folders (a run cut short has no rc)
    for part in ('mm', 'fa', 'mm-rerun', 'fa-rerun', 'fa-rerun2'):
        p = os.path.join(GW, part, 'bops', 'summary.txt')
        if not os.path.exists(p):
            if not part.endswith(('rerun', 'rerun2')):
                print(f'replays: no {part} summary')
            continue
        last = None
        for line in open(p, encoding='utf-8'):
            line = line.rstrip('\n')
            m = SUM_RE.match(line)
            if m:
                last = [part, m['tag'], m['rc'].strip(), m['passed'].strip(), m['ok'], m['fail'], m['norec'], m['ns'],
                        m['ill'], '', '' if m['rc'].strip() else 'cut short (no exit code): run again in a rerun folder']
                rows.append(last)
            elif last is not None and 'cpu-check' in line:
                last[9] = line.strip()
    write_tsv(os.path.join(out, 'replays.tsv'), ['driver', 'run', 'rc', 'tests passed', 'OK', 'FAIL', 'NOREC',
                                                 'not supported', 'illegal access', 'cpu-check', 'note'], rows)
    bad = [r for r in rows if r[5] != '0' or r[6] != '0' or r[8] != '0']
    cut = [r for r in rows if r[10]]
    print(f'replays: {len(rows)} runs, {len(bad)} with a FAIL, NOREC or illegal access'
          + (': ' + ', '.join(r[1] for r in bad) if bad else '') + f'; cut short: {len(cut)}'
          + (' (' + ', '.join(r[1] for r in cut) + ')' if cut else ''))
    for r in rows:
        if r[9]:
            print(f'  {r[1]}: {r[9]}')


def pipe(GW, out):
    p = os.path.join(GW, 'pipe', 'summary.tsv')
    if not os.path.exists(p):
        print('pipe: no table')
        return
    shutil.copy(p, os.path.join(out, 'pipe.tsv'))
    rows = read_tsv(p)
    counts = {}
    hashes = {}
    for r in rows:
        b = r.get('tree', r.get('build', '?'))
        v = r.get('verdict', '?')
        counts.setdefault(b, {}).setdefault(v, 0)
        counts[b][v] += 1
        k = (r.get('arm'), r.get('layout'), r.get('arch'), r.get('check'))
        hashes.setdefault(k, {})[b] = (r.get('plain tokens hash', '-'), r.get('plain logits hash', '-'))
    print('pipe: ' + '; '.join(f'{b} ' + ', '.join(f'{v} {n}' for v, n in sorted(c.items())) for b, c in counts.items()))
    # the plain run's hashes, base against branch, for every check that records them
    tab, same, diff = [], 0, 0
    for k, h in sorted(hashes.items(), key=lambda x: tuple(str(y) for y in x[0])):
        if 'base' not in h or 'branch' not in h or h['base'][0] in ('', '-'):
            continue
        eq = h['base'] == h['branch']
        same += eq
        diff += not eq
        tab.append(list(k) + [h['base'][0], h['branch'][0], h['base'][1], h['branch'][1], 'yes' if eq else 'no'])
    write_tsv(os.path.join(out, 'pipe_plain_hashes.tsv'), ['arm', 'layout', 'arch', 'check', 'base tokens hash',
              'branch tokens hash', 'base logits hash', 'branch logits hash', 'same'], tab)
    print(f'pipe plain hashes: {same} checks the same on both builds, {diff} different')


def flash(GW, out):
    base = os.path.join(GW, 'flash')
    if not os.path.isdir(base):
        print('flash: no folder')
        return

    def reply(build, req):
        try:
            d = json.load(open(os.path.join(base, build, f'reply-{req}.json'), encoding='utf-8'))
        except Exception:
            return None
        t = d.get('timings', {})
        return d.get('tokens'), d.get('content', ''), t.get('draft_n'), t.get('draft_n_accepted')

    def mem(build, name):
        p = os.path.join(base, build, name)
        if not os.path.exists(p):
            return '-'
        return ' '.join(line.split(',')[1].strip() for line in open(p, encoding='utf-8') if line.strip())

    rows = []
    for req in ('code', 'prose'):
        a, b = reply('branch', req), reply('base', req)
        if a is None or b is None:
            rows.append([req, '-', '-', 'a reply is missing', '-', '-', '-'])
            continue
        same = a[0] == b[0]
        n = next((i for i, (x, y) in enumerate(zip(a[0], b[0])) if x != y), min(len(a[0]), len(b[0])))
        rows.append([req, len(a[0]), len(b[0]), 'the same token ids' if same else f'first difference at token {n}',
                     'the same text' if a[1] == b[1] else 'other text', f'{a[2]} and {a[3]}', f'{b[2]} and {b[3]}'])
    rows.append(['MiB per card after the load', mem('branch', 'mem.csv'), mem('base', 'mem.csv'), '', '', '', ''])
    rows.append(['MiB per card after the requests', mem('branch', 'mem-after.csv'), mem('base', 'mem-after.csv'),
                 '', '', '', ''])
    write_tsv(os.path.join(out, 'flash.tsv'), ['request', 'branch tokens', 'base tokens', 'token ids', 'text',
                                               'branch drafted and accepted', 'base drafted and accepted'], rows)
    print('flash: ' + '; '.join(f'{r[0]}: {r[3]}, drafts {r[5]} / {r[6]}' for r in rows[:2]))


def sass(W, out):
    p = os.path.join(W, 'checks', 'sass', 'pairs.txt')
    if not os.path.exists(p):
        print('sass: no summary')
        return
    shutil.copy(p, os.path.join(out, 'sass.txt'))
    print('sass: ' + ' '.join(line.strip() for line in open(p, encoding='utf-8')))


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    W, GW, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    for f in (lambda: binaries(W, out), lambda: test(GW, out), lambda: again(GW, out), lambda: replays(GW, out),
              lambda: pipe(GW, out), lambda: flash(GW, out), lambda: sass(W, out)):
        f()


if __name__ == '__main__':
    main()
