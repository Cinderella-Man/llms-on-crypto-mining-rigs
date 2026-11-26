#!/usr/bin/env python3
"""probe_table.py: scripts/fit_probe.sh's output as one table, without the server lines (they hold the rig's paths).

    python3 -I probe_table.py fit_probe.md > fit_probe.tsv

One line per labelled line, model (main, then the draft where the line has one) and card: llama.cpp's memory breakdown
(the card's total and free MiB as llama.cpp read them, the model's own MiB = weights + context + compute), then the
card's projected use of both models and what that leaves free against the fit step's target. Standard library only.
"""
import re
import sys

print('\t'.join(['line', 'model', 'card', 'total_mib', 'free_mib', 'self_mib', 'weights_mib', 'context_mib',
                 'compute_mib', 'projected_used_mib', 'projected_free_mib', 'target_free_mib']))
label, seen, used = '', {}, {}
rows = []
for raw in open(sys.argv[1]):
    m = re.match(r'== ([^:]+):', raw)
    if m:
        label, seen = m.group(1), {}
        continue
    m = re.search(r'\|\s+- CUDA(\d+) \([^)]*\)\s+\|\s+(\d+) = (\d+) \+ \(\s*(\d+) =\s+(\d+) \+\s+(\d+) \+\s+(\d+)\)', raw)
    if m:
        card = int(m.group(1))
        if int(m.group(4)) == 0:
            continue
        model = 'draft' if card in seen else 'main'
        seen[card] = 1
        rows.append([label, model, card] + [int(m.group(i)) for i in range(2, 8)])
        continue
    m = re.search(r'common_params_fit_impl:\s+- CUDA(\d+) .*?(\d+) total,\s+(\d+) used,\s+(-?\d+) free vs\. target of\s+(\d+)', raw)
    if m:
        used[(label, int(m.group(1)))] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
for r in rows:
    u = used.get((r[0], r[2]), ('', '', ''))
    print('\t'.join(str(x) for x in r + list(u)))
