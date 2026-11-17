#!/usr/bin/env python3
"""The P106 check of the v5 candidate (Q1525): the tables of rig_suite.sh's and cand_load.sh's output (standard
library only; ../../2026-11-13_c831_upstream/run/r153_load_tables.py makes the memory and reply tables).

Usage: cand_tables.py SUITE LOAD DEST [NEW_TAG REF_TAG]
  SUITE  rig_suite.sh's out/: v4kern/ and ro8kt/ (replay_checks.sh's results.tsv, unrecorded.tsv), archs/logs/*.log,
         off/archs-off-base.log and off/archs-off.log (the reference build writes the CPU logits, the candidate
         compares)
  LOAD   cand_load.sh's OUT: <model>/load-<label>/prof.csv (GGML_CUDA_PROFILE_MODE=graph), <model>/greedy/greedy-*.tsv
  DEST   a folder for the tables:
         replays.tsv        every replayed run of both sets: what it must do, cases passed, failed, unrecorded, verdict
         archs.tsv          check.sh's three test-llama-archs runs on the card: tests passed, FAIL lines, exit status,
                            card-built masks identical to the host's
         logits.tsv         per random model: the CPU logits of the candidate against the reference's (nmse, values
                            differing) and each build's own card-against-CPU error from the same runs
         launch-groups.tsv  per release file and card: evaluations read on each build (and the profiler's unread ones),
                            whether the evaluations' (graph, launch groups) pairs are the same with their counts, and
                            the decode-sized evaluations' launch groups (value x count) on each build (decode-sized is
                            a GPU time under 60 ms, so an evaluation near that line can count on one build only)
         greedy-vs-v4.tsv   rig_greedy.sh's text on each build run here against v4.0's text of R153 (same command)
  NEW_TAG, REF_TAG  the loads' labels (cand, ref)
"""
import collections
import csv
import glob
import os
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
# v4.0's tree (llama.cpp-20261113-eb312b7bc-...-c828) gave these texts in R153 part (3), the same command and files
V4_GREEDY = {'flashnext': '4384ebc02147be1ee6f38d1ce4a43f21', 'flintq8': 'c5b2d8f20654728edfe38c37458b4c5c',
             'flintq5': 'd72ca1b968a8c1eb2b1ebeea22458ff9', 'ornith': '35f987c059c7d2f521404e964995cd2d'}
DECODE_US = 60000  # prof_summary.py's default: an evaluation under 60 ms on one card is decode-sized


def write(dest, name, header, rows):
    with open(os.path.join(dest, name), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def replays(suite):
    rows = []
    for s in ('v4kern', 'ro8kt'):
        p = os.path.join(suite, s, 'results.tsv')
        if not os.path.exists(p):
            rows.append([s, '(no results)', '', '', '', '', '', ''])
            continue
        for r in csv.DictReader(open(p), delimiter='\t'):
            rows.append([s, r['run'], r['expect'], r['cases'], r['fail_lines'], r['exit'], r['verdict'], r['unrecorded']])
    return rows


def archs(suite):
    rows = []
    for tag in ('archs', 'archs-on', 'archs-mask'):
        p = os.path.join(suite, 'archs', 'logs', tag + '.log')
        if not os.path.exists(p):
            rows.append([tag, '(no log)', '', '', '', ''])
            continue
        t = ANSI.sub('', open(p, errors='replace').read())
        summ = re.findall(r'Summary: .*', t)
        ok = sum(1 for line in t.splitlines() if re.search(r'\|\s*OK\b', line))
        fail = sum(1 for line in t.splitlines() if 'FAIL' in line)
        ex = re.findall(r'^exit (\d+)', t, re.M)
        masks = len(re.findall(r'mask.*identical|identical to the host', t))
        rows.append([tag, summ[-1] if summ else '', ok, fail, ex[-1] if ex else '', masks])
    return rows


def card_nmse(path):
    """(architecture, Dense|MoE) -> the first nmse column (card against CPU) of test-llama-archs' table."""
    out = {}
    if not os.path.exists(path):
        return out
    pending = None  # a row whose cells the logits lines interrupted: its rest starts a later line
    cont = re.compile(r'^\s*(OK|FAIL)\s*\(([0-9.e+-]+)\)\|')
    for line in ANSI.sub('', open(path, errors='replace').read()).splitlines():
        cells = [c.strip() for c in line.split('|')]
        if line.startswith('|') and len(cells) >= 5 and 'CUDA0' in cells[2]:
            m = re.match(r'(OK|FAIL)\s*\(([0-9.e+-]+)\)', cells[4])
            if m:
                out.setdefault((cells[1], cells[3]), m.group(2))
                pending = None
            elif cells[4].startswith('SKIP'):
                pending = None
            else:
                pending = (cells[1], cells[3])
            continue
        m = cont.match(line)
        if m and pending:
            out.setdefault(pending, m.group(2) if m.group(1) == 'OK' else 'FAIL ' + m.group(2))
            pending = None
    return out


def nmse_for(table, name):
    """the logits file's name (logits_dump_name: '<arch>', or '<arch>-moe' for an architecture with both models)"""
    if name.endswith('-moe') and (name[:-4], 'MoE') in table:
        return table[(name[:-4], 'MoE')]
    return table.get((name, 'Dense'), table.get((name, 'MoE'), ''))


def logits(suite):
    base = os.path.join(suite, 'off', 'archs-off-base.log')
    new = os.path.join(suite, 'off', 'archs-off.log')
    rows = []
    if not os.path.exists(new):
        return rows
    cmp = re.compile(r'^(\S+): logits against \S+: nmse (\S+), max abs diff (\S+), (\d+) of (\d+) differ')
    nb, nn = card_nmse(base), card_nmse(new)
    seen = set()
    for line in open(new, errors='replace'):
        m = cmp.match(line.strip())
        if m:
            name = m.group(1)
            seen.add(name)
            rows.append([name, m.group(2), m.group(3), m.group(4), m.group(5), nmse_for(nb, name), nmse_for(nn, name)])
    for line in open(new, errors='replace'):
        m = re.search(r'wrote \d+ logits to \S*off-(\S+)\.f32', line)
        if m and m.group(1) not in seen:
            rows.append([m.group(1), '(only the candidate has it: written, not compared)', '', '', '', '',
                         nmse_for(nn, m.group(1))])
    return rows


def read_prof(path):
    evals = []  # (device, graph, eval_groups, eval_us) of every read evaluation
    unread = collections.Counter()
    if not os.path.exists(path):
        return None, None
    for r in csv.DictReader(open(path)):
        if r.get('group') == '-1':
            unread[int(r['device'])] += 1
            continue
        try:
            evals.append((int(r['device']), int(r['graph']), int(r['eval_groups']), float(r['eval_us'])))
        except (KeyError, ValueError):
            continue
    return evals, unread


def launch_groups(load, new_tag, ref_tag):
    rows = []
    for mdir in sorted(glob.glob(os.path.join(load, '*'))):
        if not os.path.isdir(mdir):
            continue
        model = os.path.basename(mdir)
        en, un = read_prof(os.path.join(mdir, 'load-' + new_tag, 'prof.csv'))
        er, ur = read_prof(os.path.join(mdir, 'load-' + ref_tag, 'prof.csv'))
        if en is None or er is None:
            rows.append([model, '(no profile)'] + [''] * 7)
            continue
        for dev in sorted({e[0] for e in en} | {e[0] for e in er}):
            pn = collections.Counter((g, n) for d, g, n, _ in en if d == dev)
            pr = collections.Counter((g, n) for d, g, n, _ in er if d == dev)
            dn = collections.Counter(n for d, g, n, us in en if d == dev and us < DECODE_US)
            dr = collections.Counter(n for d, g, n, us in er if d == dev and us < DECODE_US)

            def fmt(c):
                return ' '.join(f'{k}x{v}' for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:6])
            rows.append([model, dev, sum(1 for e in en if e[0] == dev), un.get(dev, 0),
                         sum(1 for e in er if e[0] == dev), ur.get(dev, 0),
                         'same' if pn == pr else
                         f'differs: cand only {sorted((pn - pr).items())[:4]}, ref only {sorted((pr - pn).items())[:4]}',
                         fmt(dn), fmt(dr), 'same' if set(dn) == set(dr) else 'differs'])
    return rows


def greedy(load):
    rows = []
    for g in sorted(glob.glob(os.path.join(load, '*', 'greedy*', 'greedy-*.tsv'))):
        model = g.split(os.sep)[-3]
        for r in list(csv.reader(open(g), delimiter='\t'))[1:]:
            exp = V4_GREEDY.get(model, '')
            rows.append([model, r[0], r[1], r[2], exp, 'same' if r[1] == exp else 'differs', r[3]])
    return rows


def main():
    suite, load, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    new_tag = sys.argv[4] if len(sys.argv) > 4 else 'cand'
    ref_tag = sys.argv[5] if len(sys.argv) > 5 else 'ref'
    os.makedirs(dest, exist_ok=True)
    write(dest, 'replays.tsv', ['set', 'run', 'expect', 'cases_passed', 'fail_lines', 'exit', 'verdict', 'unrecorded'],
          replays(suite))
    write(dest, 'archs.tsv', ['run', 'summary', 'ok_rows', 'fail_lines', 'exit', 'mask_lines'], archs(suite))
    write(dest, 'logits.tsv', ['model', 'cpu_nmse_cand_vs_ref', 'cpu_max_abs_diff', 'values_differing', 'values',
                               'card_vs_cpu_nmse_ref', 'card_vs_cpu_nmse_cand'], logits(suite))
    write(dest, 'launch-groups.tsv', ['model', 'card', 'evals_cand', 'unread_cand', 'evals_ref', 'unread_ref',
                                      'graph_and_groups_pairs', 'decode_groups_cand', 'decode_groups_ref',
                                      'decode_values'], launch_groups(load, new_tag, ref_tag))
    write(dest, 'greedy-vs-v4.tsv', ['model', 'build', 'md5', 'bytes', 'v4_md5_r153', 'vs_v4', 'eval'], greedy(load))


if __name__ == '__main__':
    main()
