#!/usr/bin/env python3
"""Splits cuobjdump -sass output into one file per function whose name matches a regex: <out>/<n>_<name>.sass
(the first copy of each name only). Usage: split_sass.py <all.sass> <out dir> <regex>"""
import os, re, sys
src, out, rx = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
os.makedirs(out, exist_ok=True)
cur, lines, seen, n = None, [], set(), 0
def flush():
    global n
    if cur and rx.search(cur) and cur not in seen:
        seen.add(cur)
        n += 1
        open(os.path.join(out, f"{n:02d}_{cur[:150]}.sass"), "w").write("".join(lines))
for line in open(src, errors="replace"):
    m = re.search(r"Function : (\S+)", line)
    if m:
        flush()
        cur, lines = m.group(1), [line]
    elif cur:
        lines.append(line)
flush()
print(n, "functions written")
