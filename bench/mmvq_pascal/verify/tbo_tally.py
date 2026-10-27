#!/usr/bin/env python3
"""Tally test-backend-ops logs: OK/FAIL per (op, type, tokens 1-4 or more, view). A mutation build must fail exactly the
cases its kernel takes (Q149: all 258 Q4_K cases of 1-4 tokens) and pass the rest.

Usage: tbo_tally.py $WORK/logs/<tag>-mm.log $WORK/logs/<tag>-mmid.log ...   (logs of tbo.sh / run_suite.sh)"""
import re, sys
from collections import Counter
c = Counter()
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        m = re.search(r"([A-Z_]+)\((type(?:_a)?)=([A-Za-z0-9_]+),(.*)\): (OK|FAIL)$", line)
        if not m:
            continue
        op, _, ty, rest, st = m.groups()
        # tokens: n= for MUL_MAT/MUL_MAT_ID/large_act/shared, m= for MUL_MAT_VEC_FUSION
        key = "m" if op == "MUL_MAT_VEC_FUSION" else "n"
        t = re.search(r"(?:^|,)" + key + r"=(\d+)", rest)
        n = int(t.group(1)) if t else 0
        view = bool(re.search(r"(k_v|m_v|pad)=[1-9]", rest))
        c[(op, ty, "1-4" if 1 <= n <= 4 else ">4", "view" if view else "", st)] += 1
for k in sorted(c):
    print(f"{c[k]:5d}  {' '.join(x for x in k if x)}")
