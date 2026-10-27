#!/usr/bin/env python3
"""Main-loop cost of the fork's single-token Pascal kernels (mul_mat_vec_q_pascal) from per-function sm_61 SASS files
(sass_cmp.sh output): warp instructions in the largest backward-branch loop, per weight byte. Types: Q4_K, Q5_K, Q6_K,
IQ4_XS.

Usage: forkloops.py $WORK/sass/new-*.sass"""
import re
import sys

bsz = {12: 144, 13: 176, 14: 210, 23: 136}
for path in sys.argv[1:]:
    m = re.search(r'mul_mat_vec_q_pascalIL9ggml_type(\d+)ELi(\d)ELi(\d)ELb([01])ELi(\d)E(?:Li(\d+)E)?E', path)
    if not m or int(m.group(1)) not in bsz:
        continue
    t, R, C, st, fk = (int(x) for x in m.groups()[:5])
    ins = []
    for l in open(path):
        mm = re.match(r'\s+/\*([0-9a-f]{4,})\*/\s+(.*?);', l)
        if mm:
            ins.append((int(mm.group(1), 16), mm.group(2)))
    if not ins:  # offsets were stripped by sass_cmp.sh: number the instructions instead
        k = 0
        for l in open(path):
            s = l.strip()
            if s and not s.startswith(('/*', 'Function', '.')) and ';' in s:
                ins.append((k * 8, s.split(';')[0]))
                k += 1
    best = None
    for a, op in ins:
        mb = re.search(r'BRA\b.*?0x([0-9a-f]+)\s*$', op)
        if mb and int(mb.group(1), 16) < a:
            span = [x for x in ins if int(mb.group(1), 16) <= x[0] <= a]
            if best is None or len(span) > len(best):
                best = span
    if not best:
        print(f"type {t} R{R} C{C} staged{st} fk{fk}: no loop found")
        continue
    n = len([op for _, op in best if not op.startswith('NOP')])
    wb = 32 * R * (2 if fk == 2 else 1) * C * bsz[t]
    print(f"type {t} R{R} C{C} staged{st} fk{fk}: loop {n} warp-instr, {32 * n / wb:.2f} lane-instr/B")
