#!/usr/bin/env python3
"""C1039's tables from run.sh's WORK folder (standard library only).

    python3 -I tables.py <data folder> <set>=<WORK> [<set>=<WORK> ...]

One set is one run.sh call (its WORK folder); C1039 ran two, "plain" and "f32" (EXTRA_ENV: cuBLAS in F32 and a P106's
kernels forced). Writes into the data folder, every row led by its set:
  runs.tsv   per run: build, fusions, exit status, test-llama-archs' summary, card rows OK / FAIL / SKIP, and how many
             random models' CPU logits the run wrote or compared with the set's first run's, and how many differ
  nmse.tsv   per random model: whether the two builds' CPU logits are the same bits ("no logits file": the test
             writes none for that model, or only one build has it), then the card-against-CPU nmse of every run, and
             three verdicts: the builds agree with the fusions on, with them off, and each repeat gives its first
             run's number
  groups.tsv the verdicts counted, apart for the models whose CPU logits are the same on both builds (the only ones
             whose card results can be compared across the builds)
"""
import os
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
RUNS = ['ref-on', 'cand-on', 'ref-off', 'cand-off', 'ref-on2', 'cand-on2']
CMP = re.compile(r'^(\S+): logits against \S+: nmse (\S+), max abs diff (\S+), (\d+) of (\d+) differ')


def card_nmse(path):
    """(architecture, Dense|MoE) -> the first nmse column (card against CPU) of test-llama-archs' table.

    The same parser as docs/data/2026-11-17_v5_cand_p106_check/run/cand_tables.py: a row whose cells the logits
    lines interrupted continues on a later line."""
    out = {}
    if not os.path.exists(path):
        return out
    pending = None
    cont = re.compile(r'^\s*(OK|FAIL)\s*\(([0-9.e+-]+)\)\|')
    for line in ANSI.sub('', open(path, errors='replace').read()).splitlines():
        cells = [c.strip() for c in line.split('|')]
        if line.startswith('|') and len(cells) >= 5 and 'CUDA0' in cells[2]:
            m = re.match(r'(OK|FAIL)\s*\(([0-9.e+-]+)\)', cells[4])
            if m:
                out.setdefault((cells[1], cells[3]), m.group(2) if m.group(1) == 'OK' else 'FAIL ' + m.group(2))
                pending = None
            elif cells[4].startswith('SKIP'):
                out.setdefault((cells[1], cells[3]), 'SKIP')
                pending = None
            else:
                pending = (cells[1], cells[3])
            continue
        m = cont.match(line)
        if m and pending:
            out.setdefault(pending, m.group(2) if m.group(1) == 'OK' else 'FAIL ' + m.group(2))
            pending = None
    return out


def compares(path):
    """logits file name -> (nmse, max abs diff, differing, values) of the run's CPU logits against the first run's"""
    out = {}
    if os.path.exists(path):
        for line in open(path, errors='replace'):
            m = CMP.match(ANSI.sub('', line).strip())
            if m:
                out[m.group(1)] = m.groups()[1:]
    return out


def dump_name(key, keys):
    """test-llama-archs' logits file name: '<arch>', or '<arch>-moe' for the MoE model of an architecture with both"""
    arch, kind = key
    return arch + '-moe' if kind == 'MoE' and (arch, 'Dense') in keys else arch


def write(dest, name, header, rows):
    with open(os.path.join(dest, name), 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(c) for c in r) + '\n')


def one_set(label, work):
    logs = {r: os.path.join(work, 'logs', r + '.log') for r in RUNS}
    rcs = {}
    if os.path.exists(os.path.join(work, 'rc.txt')):
        for line in open(os.path.join(work, 'rc.txt')):
            m = re.match(r'(\S+) rc=(\d+)', line)
            if m:
                rcs[m.group(1)] = m.group(2)
    tables = {r: card_nmse(logs[r]) for r in RUNS}
    cmps = {r: compares(logs[r]) for r in RUNS}

    runs = []
    for r in RUNS:
        t = ANSI.sub('', open(logs[r], errors='replace').read()) if os.path.exists(logs[r]) else ''
        summ = re.findall(r'^Summary.*$', t, re.M)
        vals = tables[r].values()
        wrote = len(re.findall(r'wrote \d+ logits to', t))
        runs.append([label, r, 'af00bb13b' if r.startswith('ref') else 'eee615cdb',
                     'off (GGML_CUDA_DISABLE_FUSION=1)' if '-off' in r else 'on (default)', rcs.get(r, ''),
                     summ[-1] if summ else '', sum(1 for v in vals if not v.startswith(('FAIL', 'SKIP'))),
                     sum(1 for v in vals if v.startswith('FAIL')), sum(1 for v in vals if v == 'SKIP'),
                     wrote, len(cmps[r]), sum(1 for c in cmps[r].values() if c[2] != '0')])
    keys = []
    for r in RUNS:
        for k in tables[r]:
            if k not in keys:
                keys.append(k)
    rows = []
    groups = {}
    for k in keys:
        name = dump_name(k, keys)
        c = cmps['cand-on'].get(name)
        cpu = 'same' if c and c[2] == '0' else ('differ' if c else 'no logits file')
        v = [tables[r].get(k, '') for r in RUNS]
        ref_on, cand_on, ref_off, cand_off, ref_on2, cand_on2 = v
        num = all(x and not x.startswith(('FAIL', 'SKIP')) for x in v)
        on_same = 'yes' if num and ref_on == cand_on else ('no' if num else '')
        off_same = 'yes' if num and ref_off == cand_off else ('no' if num else '')
        rep_same = 'yes' if num and ref_on == ref_on2 and cand_on == cand_on2 else ('no' if num else '')
        rows.append([label, k[0], k[1], name, cpu] + v + [on_same, off_same, rep_same])
        if num:
            g = groups.setdefault(cpu, [0, 0, 0, 0])
            g[0] += 1
            g[1] += on_same == 'yes'
            g[2] += off_same == 'yes'
            g[3] += rep_same == 'yes'
    return runs, rows, groups


def main():
    dest = sys.argv[1]
    runs, rows, groups = [], [], []
    for arg in sys.argv[2:]:
        label, work = arg.split('=', 1)
        a, b, g = one_set(label, work)
        runs += a
        rows += b
        groups += [[label, k] + g[k] for k in sorted(g)]
    write(dest, 'runs.tsv', ['set', 'run', 'build', 'fusions', 'exit', 'summary', 'card_rows_ok', 'card_rows_fail',
                             'card_rows_skip', 'cpu_logits_written', 'cpu_logits_compared_with_ref_on',
                             'cpu_logits_differing'], runs)
    write(dest, 'nmse.tsv', ['set', 'arch', 'model', 'logits_file', 'cpu_logits_cand_vs_ref'] +
          ['card_vs_cpu_nmse_' + r.replace('-', '_') for r in RUNS] +
          ['builds_same_fusions_on', 'builds_same_fusions_off', 'repeats_same'], rows)
    write(dest, 'groups.tsv', ['set', 'cpu_logits_cand_vs_ref', 'models_with_a_number_in_every_run',
                               'builds_same_fusions_on', 'builds_same_fusions_off', 'repeats_same'], groups)


if __name__ == '__main__':
    main()
