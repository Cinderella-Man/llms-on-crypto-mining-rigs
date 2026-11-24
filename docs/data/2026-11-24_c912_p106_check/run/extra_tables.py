#!/usr/bin/env python3
"""C912's P106 check (Q1649): three tables beside cand_tables.py's and load_tables.py's (standard library only).

Usage: extra_tables.py LOAD SUITE OTHER_COMPARE DEST
  LOAD           cand_load.sh's OUT: <model>/load-<label>/{server-<model>-<label>.log,console.txt}
  SUITE          rig_suite.sh's out/: off/archs-off.log (the candidate's CPU logits against the reference's), and
                 v4kern/unrecorded.tsv, ro8kt/unrecorded.tsv (replay_checks.sh: run, case id, case)
  OTHER_COMPARE  the same comparison's log from another machine (test-llama-archs with LLAMA_ARCHS_LOGITS comparing
                 the candidate against the reference's files), or '-' to leave that table out
  DEST           a folder for the tables:
                 load-start.tsv               per model and load: the server log's time of its first line and of
                                              'load_model: loading model' (seconds since the log's start), and the
                                              seconds to load from rig_server.sh's console ('up after N s')
                 unrecorded.tsv               per case the candidate lists and the record sets lack: its id, the case,
                                              how many runs selected it, and those runs
                 logits-vs-other-machine.tsv  per random model: nmse and values differing on this rig and on the
                                              other machine, and whether both agree
"""
import collections
import csv
import glob
import os
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
STAMP = re.compile(r'^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ')
CMP = re.compile(r'^([A-Za-z0-9_.-]+): logits against [^:]+: nmse ([0-9.e+-]+), max abs diff ([0-9.e+-]+), '
                 r'(\d+) of (\d+) differ', re.M)


def stamp(line):
    m = STAMP.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return round(mi * 60 + s + ms / 1e3 + us / 1e6, 6)


def write(dest, name, header, rows):
    with open(os.path.join(dest, name), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def load_start(load):
    rows = []
    for log in sorted(glob.glob(os.path.join(load, '*', 'load-*', 'server-*.log'))):
        ldir = os.path.dirname(log)
        model = os.path.basename(os.path.dirname(ldir))
        label = os.path.basename(ldir)[5:]
        first = load_at = None
        for line in open(log, errors='replace'):
            t = stamp(line)
            if t is None:
                continue
            if first is None:
                first = t
            if load_at is None and 'load_model: loading model' in line:
                load_at = t
                break
        con = os.path.join(ldir, 'console.txt')
        m = re.search(r'up after (\d+) s', open(con, errors='replace').read()) if os.path.exists(con) else None
        rows.append([model, label, '' if first is None else f'{first:.3f}', '' if load_at is None else f'{load_at:.3f}',
                     m.group(1) if m else ''])
    return rows


def unrecorded(suite):
    runs = collections.defaultdict(list)
    names = {}
    for s in ('v4kern', 'ro8kt'):
        p = os.path.join(suite, s, 'unrecorded.tsv')
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p), delimiter='\t'):
            runs[r['id']].append(f"{s}/{r['run']}")
            names[r['id']] = r['case']
    return [[i, names[i], len(runs[i]), ' '.join(runs[i])] for i in sorted(names, key=lambda k: names[k])]


def compares(path):
    out = {}
    for m in CMP.finditer(ANSI.sub('', open(path, errors='replace').read())):
        out.setdefault(m.group(1), []).append((m.group(2), int(m.group(4)), int(m.group(5))))
    return out


def vs_other(suite, other):
    here = compares(os.path.join(suite, 'off', 'archs-off.log'))
    there = compares(other)
    rows = []
    for model in sorted(set(here) | set(there)):
        a, b = here.get(model, []), there.get(model, [])
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else ('', '', '')
            y = b[i] if i < len(b) else ('', '', '')
            rows.append([model, x[0], x[1], y[0], y[1], x[2] or y[2], 'same' if x[:2] == y[:2] else 'differs'])
    return rows


def main():
    load, suite, other, dest = sys.argv[1:5]
    os.makedirs(dest, exist_ok=True)
    write(dest, 'load-start.tsv', ['model', 'load', 'first_log_line_s', 'load_model_line_s', 'load_s'], load_start(load))
    write(dest, 'unrecorded.tsv', ['id', 'case', 'runs_selecting_it', 'runs'], unrecorded(suite))
    if other != '-':
        write(dest, 'logits-vs-other-machine.tsv', ['model', 'nmse_here', 'values_differing_here', 'nmse_other',
                                                    'values_differing_other', 'values', 'agree'], vs_other(suite, other))


if __name__ == '__main__':
    main()
