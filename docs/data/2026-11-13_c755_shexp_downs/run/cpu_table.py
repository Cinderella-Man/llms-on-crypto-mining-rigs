#!/usr/bin/env python3
"""multitoken_bitid shdown's output as a table (C755): one row per product, its result and its blocks past 2^13.

Usage: cpu_table.py <shdown output> > cpu_table.tsv
"""
import re
import sys

INPUT = {"MUL_MAT": "normal", "MUL_MAT/unif": "uniform", "MUL_MAT/big": "big"}
lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
print("type\tm\tk\ttokens\tinput\tresult\tvalues_differ\tvalues\tblocks_I_ge_2^13_est\tblocks")
for i, line in enumerate(lines):
    m = re.match(r"(\S+)\s+(\S+)\s+n=(\d+) m=(\d+)\s+k=(\d+)\s*: (bit-identical|DIFFERENT) \((\d+) of (\d+) values", line)
    if not m:
        continue
    b = re.search(r": (\d+) of (\d+) blocks", lines[i + 1])
    print("\t".join([m[2], m[4], m[5], m[3], INPUT[m[1]], m[6], m[7], m[8], b[1], b[2]]))
