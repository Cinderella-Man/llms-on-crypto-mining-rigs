#!/usr/bin/env python3
# C997: the card's own time of each TOP_K op from the fork's profiler (gp-<run>.csv, GGML_CUDA_PROFILE group mode, one
# group = one op) under test-backend-ops perf (perf-<run>.txt names the cases in order). A case's ops are the evals with
# more than one group (perf mode's timed graphs); the one-group evals before them are its warm-up. Prints each case's
# median and mean us per op by run (pu1 ps1 pu2 ps2: switch unset, set) and each pair's difference of the medians.
# The mean holds the ops where the card waited for the host's launches; the median is the card-bound op.
# Usage: kernel_prof_table.py <folder> > table.md
import csv, re, statistics as st, sys
from pathlib import Path
d = Path(sys.argv[1])
runs = ['pu1', 'ps1', 'pu2', 'ps2']
pat = re.compile(r'TOP_K\((type=[^)]*)\):')
res = {}
names = None
for r in runs:
    cases = [m.group(1).replace('type=f32,', '') for m in pat.finditer((d / f'perf-{r}.txt').read_text(errors='replace'))]
    names = names or cases
    ev = {}
    for row in csv.DictReader(open(d / f'gp-{r}.csv')):
        ev.setdefault(int(row['eval']), []).append(float(row['us']))
    segs, prev_multi = [], False
    for e in sorted(ev):
        multi = len(ev[e]) > 1
        if multi and not prev_multi:
            segs.append([])
        if multi:
            segs[-1].extend(ev[e])
        prev_multi = multi
    for c, s in zip(cases, segs):
        res.setdefault(c, {})[r] = (st.median(s), st.mean(s), len(s))
print('| case | unset 1 median (mean) | set 1 median (mean) | unset 2 median (mean) | set 2 median (mean) | set 1 - unset 1 | set 2 - unset 2 | ops a run |')
print('|---|---:|---:|---:|---:|---:|---:|---|')
for c in names:
    v = res[c]
    cells = ' | '.join(f'{v[r][0]:.2f} ({v[r][1]:.2f})' for r in runs)
    n = ', '.join(str(v[r][2]) for r in runs)
    print(f"| `{c}` | {cells} | {v['ps1'][0] - v['pu1'][0]:+.2f} | {v['ps2'][0] - v['pu2'][0]:+.2f} | {n} |")
