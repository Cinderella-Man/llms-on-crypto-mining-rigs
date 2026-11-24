#!/usr/bin/env python3
"""C1024: the PTX of the head's 64-column prompt kernels against the reference's, with block labels renumbered.

Usage: python3 -I ptx_j64.py <head ptx dump> <reference ptx dump>
A dump is `cuobjdump -ptx <llama-server>`. Takes every `.entry mul_mat_q<type, 64, ...>` of the 11 types the switch
covers (each copy), renames the basic-block labels ($L__BB<function index>_<block>), whose function index is the
entry's place in its module and so differs between translation units, and counts the reference's entries that one of
the head's copies matches. ptx_functions.py (docs/data/2026-11-24_c985_topk_order/run/) compares whole functions without
the renaming, so it lists these copies as different.
"""
import collections
import hashlib
import re
import sys

LBL = re.compile(r"\$L__BB\d+_")
ENTRY = re.compile(r"^\s*(?:\.visible\s+|\.weak\s+)*\.entry\s+(\S+?)\(")
TYPES = {"2", "3", "6", "7", "8", "16", "18", "20", "23", "41", "42"}  # ggml_type values of the 11 types


def funcs(path):
    out = collections.defaultdict(list)
    cur, name = None, None
    for ln in open(path, errors="replace"):
        if cur is None:
            m = ENTRY.match(ln)
            if m and "mul_mat_q" in m.group(1) and "ELi64E" in m.group(1):
                cur, name = [ln], m.group(1)
        else:
            cur.append(ln)
            if ln.startswith("}"):
                out[name].append(hashlib.sha1(LBL.sub("$L__BB_", "".join(cur)).encode()).hexdigest())
                cur = None
    return out


H, R = funcs(sys.argv[1]), funcs(sys.argv[2])
same = tot = 0
for n, rs in sorted(R.items()):
    t = re.match(r"_Z\d+mul_mat_qIL9ggml_type(\d+)E", n)
    if not t or t.group(1) not in TYPES:
        continue
    tot += 1
    hit = any(h in H.get(n, []) for h in rs)
    same += hit
    if not hit:
        print("no match:", n[:90])
print(f"J=64 mul_mat_q entries of the 11 types in the reference: {tot}; one of the head's copies the same PTX once "
      f"block labels are renumbered: {same}")
