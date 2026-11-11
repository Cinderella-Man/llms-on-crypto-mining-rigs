#!/usr/bin/env python3
"""FN74 (Q1316): the replay's per-case errors, run against run.

Reads replay_checks.sh's OUT/<run>.tsv (per case: id, case string, verdict, each node's error) for every run of the run
list and prints one table: per run the cases, those passing and failing, and, split into one-token calls (nb=1) and
calls of two or more query tokens, how many cases have the same errors (every node, to the digits printed) as in
qu-stock (the dense kernels), qu-m2 (every call on the per-token sparse kernel) and qu-m2-union (multi-token calls on
the union kernel). A kernel that ran gives other errors than the one it replaced; the same kernel at the same split
count gives the same ones. Then every failing case of every run.

Usage: errors.py <OUT folder of replay_checks.sh> <run list .tsv>
"""
import re
import sys
from pathlib import Path


def load(path):
    cases = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split("\t")
        if len(f) >= 3:
            cases[f[1]] = (f[2], f[3] if len(f) > 3 else "")
    return cases


def nb(case):
    m = re.search(r"\bnb=(\d+)", case)
    return int(m.group(1)) if m else 0


def main():
    out, runs_file = Path(sys.argv[1]), Path(sys.argv[2])
    runs = [l.split("\t")[0] for l in runs_file.read_text().splitlines() if l.strip()]
    expect = {l.split("\t")[0]: l.split("\t")[4] for l in runs_file.read_text().splitlines() if l.strip()}
    data = {r: load(out / f"{r}.tsv") for r in runs if (out / f"{r}.tsv").exists()}
    refs = [r for r in ("qu-stock", "qu-m2", "qu-m2-union") if r in data]
    head = ["run", "expect", "cases", "pass", "fail"]
    for ref in refs:
        head += [f"same as {ref}: 1 token", f"same as {ref}: 2+ tokens"]
    print("\t".join(head))
    for r, cases in data.items():
        ok = sum(1 for v, _ in cases.values() if v == "OK")
        row = [r, expect.get(r, ""), str(len(cases)), str(ok), str(len(cases) - ok)]
        for ref in refs:
            for one in (True, False):
                keys = [c for c in cases if c in data[ref] and (nb(c) == 1) == one]
                same = sum(1 for c in keys if cases[c][1] == data[ref][c][1])
                row.append(f"{same}/{len(keys)}")
        print("\t".join(row))
    print()
    print("run\tnb\tfailing case")
    for r, cases in data.items():
        for c, (v, _) in cases.items():
            if v != "OK":
                print(f"{r}\t{nb(c)}\t{c}")


if __name__ == "__main__":
    main()
