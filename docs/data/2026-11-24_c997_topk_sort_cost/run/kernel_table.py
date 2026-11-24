#!/usr/bin/env python3
# C997: test-backend-ops perf lines of TOP_K from perf-<run>.txt (runs u1 s1 u2 s2: switch unset, set) into one table,
# each case's us/run by run, and each pair's difference (s1 - u1, s2 - u2). Usage: kernel_table.py <folder> > table.md
import re, sys
from pathlib import Path
d = Path(sys.argv[1])
runs = ['u1', 's1', 'u2', 's2']
pat = re.compile(r'TOP_K\((type=[^)]*)\):\s+(\d+) runs -\s+([\d.]+) us/run')
rows, order = {}, []
for r in runs:
    for line in (d / f'perf-{r}.txt').read_text(errors='replace').splitlines():
        m = pat.search(line)
        if not m:
            continue
        case = m.group(1).replace('type=f32,', '')
        if case not in rows:
            rows[case] = {}
            order.append(case)
        rows[case][r] = float(m.group(3))
print('| case | unset 1 | set 1 | unset 2 | set 2 | set 1 - unset 1 | set 2 - unset 2 |')
print('|---|---:|---:|---:|---:|---:|---:|')
for c in order:
    v = rows[c]
    if not all(r in v for r in runs):
        print(f'| `{c}` | ' + ' | '.join(f"{v[r]:.2f}" if r in v else '-' for r in runs) + ' | - | - |')
        continue
    print(f'| `{c}` | ' + ' | '.join(f'{v[r]:.2f}' for r in runs) +
          f" | {v['s1'] - v['u1']:+.2f} | {v['s2'] - v['u2']:+.2f} |")
