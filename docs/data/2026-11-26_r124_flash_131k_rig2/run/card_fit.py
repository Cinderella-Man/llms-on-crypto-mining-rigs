#!/usr/bin/env python3
"""card_fit.py: memory per card of a server line, projected and real, as one table. Standard library only.

    python3 -I card_fit.py [--probe fit_probe.md] [--usable 6068] OUT [OUT ...] > card_fit.tsv

docs/data/2026-11-10_deep_lane/run/card_fit.py for the deep lane's newer card log: the highest MiB in a load come from
cards.csv (scripts/card_log.py, every card every 0.5 s through NVML) instead of gpu.csv.
--probe is the output of scripts/fit_probe.sh: llama-server's projection per card for each labelled line, made
without loading the weights. Each OUT is a results folder of bench/deep_lane/deep_lane.sh: fit_<load>.txt (the cards'
memory in use right after the load, from nvidia-smi), cards.csv or cards.csv.gz, and loads.tsv (each load's start,
context, micro-batch and draft). One line per source:
- `projected`: the probe's MiB per card for a label;
- `after loading`: fit_<load>.txt;
- `highest in the load`: the most each card held between the load's start and the next load's start, from cards.csv.
  NVML counts the driver's reserve as used (75 MiB a card on an idle P106-090 under driver 580, where nvidia-smi shows
  0), so each card's reading in the log's first sample (taken with nothing loaded) is subtracted: the column is on
  nvidia-smi's scale, as fit_<load>.txt.
Columns: source, run (the probe's label, or the folder's name), load, context, ub, draft, card0 ... cardN (MiB in
use), the fullest card, its MiB, and the MiB left on it of --usable: what one card offers a process, the card's
total minus the driver's reserve (6,144 - 76 = 6,068 MiB on a P106-090 under driver 580, which is also the total
llama.cpp prints for the card).
"""
import argparse
import csv
import datetime
import gzip
import os
import re
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--probe', default='')
ap.add_argument('--usable', type=int, default=6068)
ap.add_argument('out', nargs='*')
A = ap.parse_args()
rows = []


def add(source, run, load, ctx, ub, draft, used):
    if not used:
        return
    cards = [used[i] for i in sorted(used)]
    top = max(range(len(cards)), key=lambda i: cards[i])
    rows.append([source, run, load, ctx, ub, draft] + [str(c) for c in cards]
                + [str(top), str(cards[top]), str(A.usable - cards[top])])


if A.probe:
    label, line, used = '', {}, {}
    with open(A.probe) as f:
        for raw in f:
            m = re.match(r'== ([^:]+): (.*)', raw)
            if m:
                if used:
                    add('projected', label, '', line.get('ctx', ''), line.get('ub', ''), line.get('draft', ''), used)
                label, used = m.group(1), {}
                c = re.search(r' -c (\d+)', m.group(2))
                u = re.search(r' -ub (\d+)', m.group(2))
                line = {'ctx': c.group(1) if c else '', 'ub': u.group(1) if u else '',
                        'draft': 'on' if re.search(r' (-md|--model-draft) ', m.group(2)) else 'off'}
                continue
            m = re.search(r'common_params_fit_impl:\s+- CUDA(\d+) .*?(\d+) total,\s+(\d+) used', raw)
            if m:
                used[int(m.group(1))] = int(m.group(3))
    if used:
        add('projected', label, '', line.get('ctx', ''), line.get('ub', ''), line.get('draft', ''), used)


def card_samples(out):  # [(time, {card: MiB on nvidia-smi's scale})] from cards.csv(.gz)
    for name in ('cards.csv', 'cards.csv.gz'):
        path = os.path.join(out, name)
        if os.path.exists(path):
            break
    else:
        return []
    opener = gzip.open if path.endswith('.gz') else open
    samples, idle = [], None
    with opener(path, 'rt', newline='') as f:
        rd = csv.reader(row for row in f if not row.startswith('#'))
        head = next(rd)
        cols = {int(m.group(1)): i for i, h in enumerate(head) for m in [re.match(r'g(\d+)_used_mib$', h)] if m}
        ti = head.index('time')
        for p in rd:
            try:
                t = datetime.datetime.strptime(p[ti], '%Y-%m-%d %H:%M:%S.%f')
                used = {c: int(float(p[i])) for c, i in cols.items()}
            except (ValueError, IndexError):
                continue
            if idle is None:
                idle = used
            samples.append((t, {c: used[c] - idle[c] for c in used}))
    return samples


for out in A.out:
    run = os.path.basename(os.path.normpath(out))
    with open(os.path.join(out, 'loads.tsv'), newline='') as f:
        loads = list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))
    starts = [datetime.datetime.fromisoformat(ld['started']).replace(tzinfo=None) for ld in loads]
    samples = card_samples(out)
    for n, ld in enumerate(loads):
        used = {}
        path = os.path.join(out, f"fit_{ld['load']}.txt")
        if os.path.exists(path):
            with open(path) as f:
                for raw in f:
                    m = re.match(r'(\d+), (\d+), (\d+)\s*$', raw)
                    if m:
                        used[int(m.group(1))] = int(m.group(2))
        add('after loading', run, ld['load'], ld['ctx'], ld['ub'], ld['draft'], used)
        end = starts[n + 1] if n + 1 < len(starts) else datetime.datetime.max
        high = {}
        for t, cards in samples:
            if starts[n] <= t < end:
                for card, mib in cards.items():
                    high[card] = max(high.get(card, 0), mib)
        add('highest in the load', run, ld['load'], ld['ctx'], ld['ub'], ld['draft'], high)

if not rows:
    sys.exit('nothing read: pass --probe or a results folder')
cards = max(len(r) for r in rows) - 9
print('\t'.join(['source', 'run', 'load', 'context', 'ub', 'draft'] + [f'card{i}_mib' for i in range(cards)]
                + ['fullest_card', 'fullest_mib', 'left_on_fullest_mib']))
for r in rows:
    print('\t'.join(r))
