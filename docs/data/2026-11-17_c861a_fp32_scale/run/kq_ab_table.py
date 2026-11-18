#!/usr/bin/env python3
"""kq_ab_table.py: the tables of rig_kq_ab.sh's llama-bench rounds (C861 (a), Q1549).

Reads bench-<arm>-<round>.jsonl (llama-bench -o jsonl) from OUT and prints, as Markdown:
  1. per micro-batch, prompt size, arm and round: llama-bench's avg_ts and every repetition's tok/s (from samples_ns);
  2. per micro-batch and prompt size: each arm's change against off in each round, and each arm's round-to-round
     spread (round 2 against round 1), all in per cent;
  3. the means of the rounds per arm and the change of on and all against off on those means;
  4. with REF (a load as <arm>-<round>, e.g. off-2): every load against that one load, in per cent, and per
     repetition (the same repetition's index) the change in tok/s.
Usage: python3 -I kq_ab_table.py OUT [ARMS (default "off on all")] [ROUNDS (default 2)] [REF]
"""
import json
import os
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            r = json.loads(line)
            if r.get('n_gen', 0) != 0:
                continue
            key = (int(r['n_ubatch']), int(r['n_prompt']))
            reps = [r['n_prompt'] * 1e9 / ns for ns in r['samples_ns']]
            rows[key] = (float(r['avg_ts']), reps)
    return rows


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    out = sys.argv[1]
    arms = (sys.argv[2] if len(sys.argv) > 2 else 'off on all').split()
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ref = sys.argv[4] if len(sys.argv) > 4 else None
    data = {}
    for arm in arms:
        for r in range(1, rounds + 1):
            p = os.path.join(out, f'bench-{arm}-{r}.jsonl')
            if os.path.exists(p):
                data[(arm, r)] = load(p)
    keys = sorted({k for d in data.values() for k in d})

    print('## Every load: tok/s (llama-bench avg_ts) and each repetition\n')
    print('| -ub | prompt | arm | round | avg_ts | repetitions |')
    print('|---|---|---|---|---|---|')
    for k in keys:
        for arm in arms:
            for r in range(1, rounds + 1):
                v = data.get((arm, r), {}).get(k)
                if v is None:
                    continue
                print(f'| {k[0]} | {k[1]} | {arm} | {r} | {v[0]:.2f} | ' + ', '.join(f'{x:.2f}' for x in v[1]) + ' |')

    print('\n## Change against off per round, and each arm\'s round-to-round spread (%)\n')
    head = ['-ub', 'prompt']
    for arm in arms[1:]:
        head += [f'{arm} vs off, r{r}' for r in range(1, rounds + 1)]
    if rounds >= 2:
        head += [f'{arm} r2 vs r1' for arm in arms]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for k in keys:
        cells = [str(k[0]), str(k[1])]
        for arm in arms[1:]:
            for r in range(1, rounds + 1):
                a = data.get((arm, r), {}).get(k)
                b = data.get((arms[0], r), {}).get(k)
                cells.append(f'{pct(a[0], b[0]):+.2f}' if a and b else '-')
        if rounds >= 2:
            for arm in arms:
                a = data.get((arm, 2), {}).get(k)
                b = data.get((arm, 1), {}).get(k)
                cells.append(f'{pct(a[0], b[0]):+.2f}' if a and b else '-')
        print('| ' + ' | '.join(cells) + ' |')

    print('\n## Means of the rounds (tok/s) and the change against off (%)\n')
    head = ['-ub', 'prompt'] + [f'{arm} mean' for arm in arms] + [f'{arm} vs off' for arm in arms[1:]]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for k in keys:
        means = {}
        for arm in arms:
            vals = [data[(arm, r)][k][0] for r in range(1, rounds + 1) if k in data.get((arm, r), {})]
            means[arm] = sum(vals) / len(vals) if len(vals) == rounds else None
        cells = [str(k[0]), str(k[1])] + [f'{means[a]:.2f}' if means[a] else '-' for a in arms]
        for arm in arms[1:]:
            cells.append(f'{pct(means[arm], means[arms[0]]):+.2f}' if means[arm] and means[arms[0]] else '-')
        print('| ' + ' | '.join(cells) + ' |')

    if ref:
        rarm, rr = ref.rsplit('-', 1)
        rdata = data[(rarm, int(rr))]
        loads = [(a, r) for r in range(1, rounds + 1) for a in arms if (a, r) in data and (a, r) != (rarm, int(rr))]
        print(f'\n## Every load against {ref} (%), and each repetition against {ref}\'s same repetition (tok/s)\n')
        head = ['-ub', 'prompt'] + [f'{a}-{r}' for a, r in loads]
        print('| ' + ' | '.join(head) + ' |')
        print('|' + '---|' * len(head))
        for k in keys:
            cells = [str(k[0]), str(k[1])]
            for a, r in loads:
                v = data[(a, r)].get(k)
                b = rdata.get(k)
                if not v or not b:
                    cells.append('-')
                    continue
                reps = ', '.join(f'{x - y:+.2f}' for x, y in zip(v[1], b[1]))
                cells.append(f'{pct(v[0], b[0]):+.2f} ({reps})')
            print('| ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
