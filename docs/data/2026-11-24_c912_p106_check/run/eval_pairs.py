#!/usr/bin/env python3
"""C912's P106 check (Q1649): the launch groups of each evaluation, matched by its index on both builds (standard
library only). cand_tables.py compares the multiset of (graph, launch groups) pairs per card; when the light profiler
could not read an evaluation's events on one build ('not ready', group -1) and could on the other, the multisets differ
by that evaluation alone. This table matches evaluations by index instead.

Usage: eval_pairs.py LOAD DEST [NEW_TAG REF_TAG]
  LOAD  cand_load.sh's OUT: <model>/load-<label>/prof.csv (GGML_CUDA_PROFILE_MODE=graph)
  DEST  a folder for eval-pairs.tsv: per model and card, the evaluations on each build, those read on both, how many of
        them have the same (graph, launch groups) on both, the evaluations not ready on each build (index:graph), and
        the (graph, launch groups) the other build read for them
"""
import csv
import glob
import os
import sys


def read(path):
    ev, nr = {}, {}
    for r in csv.DictReader(open(path)):
        d, e = int(r['device']), int(r['eval'])
        if r['group'] == '-1':
            nr.setdefault(d, {})[e] = r['graph']
            continue
        ev.setdefault(d, {}).setdefault(e, (r['graph'], r['eval_groups']))
    return ev, nr


def main():
    load, dest = sys.argv[1], sys.argv[2]
    new_tag = sys.argv[3] if len(sys.argv) > 3 else 'cand'
    ref_tag = sys.argv[4] if len(sys.argv) > 4 else 'ref'
    rows = []
    for mdir in sorted(glob.glob(os.path.join(load, '*'))):
        pn, pr = (os.path.join(mdir, f'load-{t}', 'prof.csv') for t in (new_tag, ref_tag))
        if not (os.path.exists(pn) and os.path.exists(pr)):
            continue
        (en, nn), (er, nrr) = read(pn), read(pr)
        for d in sorted(set(en) | set(er)):
            a, b = en.get(d, {}), er.get(d, {})
            na, nb = nn.get(d, {}), nrr.get(d, {})
            both = sorted(set(a) & set(b))
            same = sum(1 for e in both if a[e] == b[e])
            other_n = ' '.join(f'{e}:{"x".join(b[e])}' for e in sorted(na) if e in b)
            other_r = ' '.join(f'{e}:{"x".join(a[e])}' for e in sorted(nb) if e in a)
            rows.append([os.path.basename(mdir), d, len(a) + len(na), len(b) + len(nb), len(both), same,
                         ' '.join(f'{e}:{g}' for e, g in sorted(na.items())),
                         ' '.join(f'{e}:{g}' for e, g in sorted(nb.items())), other_n, other_r])
    with open(os.path.join(dest, 'eval-pairs.tsv'), 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['model', 'card', f'evals_{new_tag}', f'evals_{ref_tag}', 'read_on_both', 'same_graph_and_groups',
                    f'not_ready_{new_tag}', f'not_ready_{ref_tag}', f'{ref_tag}_read_for_{new_tag}_not_ready',
                    f'{new_tag}_read_for_{ref_tag}_not_ready'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
