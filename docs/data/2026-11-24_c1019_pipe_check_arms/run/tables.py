#!/usr/bin/env python3
"""C1019's tables from run.sh's output folder (one or more, later folders' rounds added after earlier ones').

Usage: python3 -I tables.py [<label>=]<out folder> [[<label>=]<out folder> ...] <data folder>
A label (e.g. t2=out2) goes before each of that folder's round names (t2/jobsall), to tell rounds of the same name in
two folders apart.
Writes into the data folder:
  rounds.tsv    run.sh's rounds.tsv, each folder's lines in order, other jobs' containers as a count
  verdicts.tsv  per round, tree, arm, layout and check: how many architectures gave each verdict
  checks.tsv    every check: its verdict, the pipelined runs differing (and with other tokens), the plain hashes
  race.tsv      every race check: the stop token, its position in sample 1, the stop text, the winner, the race's tokens
  times.tsv     every check's seconds (gpu_pipe_check.sh's times.tsv, with the round)
  jobs.tsv      every container's seconds and each build's total (gpu_pipe_check.sh's jobs.tsv, with the round)
  checktime.tsv per round and check: the seconds summed over every container, and the mean per architecture
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(path, header, data):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(data)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    outs, dest = sys.argv[1:-1], sys.argv[-1]
    rnd, vd, ck, rc, tm, jb = [], [], [], [], [], []
    ct = defaultdict(lambda: [0.0, 0])
    for arg in outs:
        label, out = arg.split('=', 1) if '=' in arg else ('', arg)
        for r in rows(os.path.join(out, 'rounds.tsv')):
            name = f"{label}/{r['round']}" if label else r['round']
            # other jobs' containers by count only (their names are other projects' business)
            o = r['other containers at start']
            rnd.append([name] + [r[k] for k in ('script', 'builds', 'settings', 'rc', 'seconds', 'verdicts')] +
                       ['0' if o in ('', 'none') else str(len(o.split()))])
            base = os.path.join(out, r['round'])
            count = defaultdict(Counter)
            for s in rows(os.path.join(base, 'summary.tsv')):
                key = (s['tree'], s['arm'], s['layout'], s['check'])
                count[key][s['verdict']] += 1
                d = s['detail']
                m = re.search(r'differing (\d+) of (\d+)(?: \((\d+) with other tokens\))?', d)
                nd, nr, nt = (m.group(1), m.group(2), m.group(3) or '0') if m else ('-', '-', '-')
                ck.append([name, s['tree'], s['arm'], s['layout'], s['arch'], s['check'], s['verdict'], nd, nr, nt,
                           s['plain tokens hash'], s['plain logits hash']])
                if s['check'] == 'race':
                    st = re.search(r'STOP_TOKEN=(\S+)', d)
                    tx = re.search(r'RACE_STOP=(\S+)', d)
                    w = re.search(r'winner=(-?\d+) race_tokens=(\d+)', d)
                    tok, pos = (st.group(1).split('@') + ['-'])[:2] if st and st.group(1) != 'none' else ('none', '-')
                    rc.append([name, s['tree'], s['arm'], s['layout'], s['arch'], tok, pos, tx.group(1) if tx else '-',
                               w.group(1) if w else '-', w.group(2) if w else '-', s['verdict'], nd, nr, nt])
            for (t, a, l, c), n in count.items():
                vd.append([name, t, a, l, c, n['OK'], n['DIFFERS'], n['FAIL'], n['SKIP'], n['INFO']])
            for s in rows(os.path.join(base, 'times.tsv')):
                tm.append([name, s['tree'], s['arm'], s['layout'], s['arch'], s['check'], s['seconds']])
                c = ct[(name, s['check'])]
                c[0] += float(s['seconds'])
                c[1] += 1
            for s in rows(os.path.join(base, 'jobs.tsv')):
                jb.append([name, s['tree'], s['job'], s['cards'], s['rc'], s['seconds']])
    os.makedirs(dest, exist_ok=True)
    write(os.path.join(dest, 'rounds.tsv'), ['round', 'script', 'builds', 'settings', 'rc', 'seconds', 'verdicts',
                                             "other jobs' containers at start"], rnd)
    write(os.path.join(dest, 'verdicts.tsv'), ['round', 'tree', 'arm', 'layout', 'check', 'OK', 'DIFFERS', 'FAIL',
                                               'SKIP', 'INFO'], vd)
    write(os.path.join(dest, 'checks.tsv'), ['round', 'tree', 'arm', 'layout', 'arch', 'check', 'verdict',
                                             'runs differing', 'pipelined runs', 'with other tokens',
                                             'plain tokens hash', 'plain logits hash'], ck)
    write(os.path.join(dest, 'race.tsv'), ['round', 'tree', 'arm', 'layout', 'arch', 'stop token',
                                           "its position in sample 1", 'stop text', 'plain winner',
                                           "plain race tokens", 'verdict', 'runs differing', 'pipelined runs',
                                           'with other tokens'], rc)
    write(os.path.join(dest, 'times.tsv'), ['round', 'tree', 'arm', 'layout', 'arch', 'check', 'seconds'], tm)
    write(os.path.join(dest, 'jobs.tsv'), ['round', 'tree', 'job', 'cards', 'rc', 'seconds'], jb)
    write(os.path.join(dest, 'checktime.tsv'), ['round', 'check', 'seconds summed', 'checks', 'mean seconds'],
          [[r, c, f'{v[0]:.1f}', v[1], f'{v[0] / v[1]:.1f}'] for (r, c), v in ct.items()])


if __name__ == '__main__':
    main()
