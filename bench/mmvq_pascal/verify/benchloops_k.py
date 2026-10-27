#!/usr/bin/env python3
"""Main-loop cost of the K-quant kernels in bench/mmvq_pascal's sm_61 SASS (build/sm61.sass): warp instructions in the
largest backward-branch loop of each staged (st_) and direct-load (p_) kernel, per weight byte, and its instruction mix.

Usage: benchloops_k.py bench/mmvq_pascal/build/sm61.sass   (written by bench/mmvq_pascal/build.sh)"""
import collections
import re
import sys

names = {'5': 'q5_K', '6': 'iq4_xs', '7': 'q6_K', '8': 'q4_K'}
bsz = {'q5_K': 176, 'iq4_xs': 136, 'q6_K': 210, 'q4_K': 144}
txt = open(sys.argv[1]).read()
for f in re.split(r'\n\s+Function : ', txt)[1:]:
    name, body = f.split('\n', 1)
    name = name.strip()
    m = re.match(r'_ZN2mb6pascal11mmvq_stagedILi(\d)ELi(\d)ELi(\d)ELi4ELb([01])ELi(\d+)EEEvNS0_4argsE', name)
    kind = 'st'
    if not m:
        m = re.match(r'_ZN2mb6pascal9mmvq_rowsILi(\d)ELi(\d)ELi(\d)ELi4ELb([01])EEEvNS0_4argsE', name)
        kind = 'p'
    if not m or m.group(1) not in names:
        continue
    t, R, C, G = names[m.group(1)], int(m.group(2)), int(m.group(3)), int(m.group(4))
    ins = []
    for l in body.split('\n'):
        mm = re.match(r'\s+/\*([0-9a-f]{4,})\*/\s+(.*?);', l)
        if mm:
            ins.append((int(mm.group(1), 16), mm.group(2)))
    best = None
    for a, op in ins:
        mb = re.search(r'BRA\b.*?0x([0-9a-f]+)\s*$', op)
        if mb and int(mb.group(1), 16) < a:
            span = [x for x in ins if int(mb.group(1), 16) <= x[0] <= a]
            if best is None or len(span) > len(best):
                best = span
    if not best:
        continue
    span = [op for _, op in best if not op.startswith('NOP')]
    c = collections.Counter((op.split()[1] if op.startswith('@') else op.split()[0]).split('.')[0] for op in span)
    wb = 32 * R * (2 if G else 1) * C * bsz[t]
    print(f"{t:6s} {kind}_r{R}c{C} gate{G}: loop {len(span)} warp-instr, {32 * len(span) / wb:.2f} lane-instr/B; "
          f"IDP {c['IDP']} LDS {c['LDS']} LDG {c['LDG']} STS {c['STS']} LOP {c['LOP'] + c['LOP3']} "
          f"SHF {c['SHF'] + c['SHL'] + c['SHR']} PRMT {c['PRMT']} I2F {c['I2F']} XMAD {c['XMAD']}")
