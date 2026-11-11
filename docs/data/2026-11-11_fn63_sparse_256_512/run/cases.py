#!/usr/bin/env python3
"""FN63 (Q1245): the replay's cases side by side, qp-stock (the dense kernels) against qp-m2 (the sparse kernel at any
size, prompt calls up to 512 query tokens).

Reads replay_checks.sh's OUT/<run>.tsv (per case: id, case string, verdict, each node's error) for both runs and prints
two tables: per number of query tokens (nb), the cases, those passing in each run, and how many have other errors in
qp-m2 than in qp-stock (a kernel that ran gives other errors than the one it replaced); then one line per case of more
than 128 query tokens, with its key fields, both verdicts and both errors (the first node's: the attention itself).

Usage: cases.py <OUT folder of replay_checks.sh> [stock run] [sparse run]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    cases = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split("\t")
        if len(f) >= 3:
            cases[f[1]] = (f[0], f[2], f[3] if len(f) > 3 else "")
    return cases


def field(case, name):
    m = re.search(r"\b" + name + r"=(\[[^\]]*\]|[^,)]+)", case)
    return m.group(1) if m else ""


def main():
    out = Path(sys.argv[1])
    stock_run = sys.argv[2] if len(sys.argv) > 2 else "qp-stock"
    sparse_run = sys.argv[3] if len(sys.argv) > 3 else "qp-m2"
    stock, sparse = load(out / f"{stock_run}.tsv"), load(out / f"{sparse_run}.tsv")
    groups = defaultdict(list)
    for c in sorted(set(stock) | set(sparse), key=lambda c: int(field(c, "nb") or 0)):
        groups[int(field(c, "nb") or 0)].append(c)
    print(f"nb\tcases\tpass {stock_run}\tpass {sparse_run}\tother errors in {sparse_run}")
    for nb, cs in sorted(groups.items()):
        ps = sum(1 for c in cs if c in stock and stock[c][1] == "OK")
        pm = sum(1 for c in cs if c in sparse and sparse[c][1] == "OK")
        diff = sum(1 for c in cs if c in stock and c in sparse and stock[c][2] != sparse[c][2])
        print(f"{nb}\t{len(cs)}\t{ps}\t{pm}\t{diff}")
    tot = [c for cs in groups.values() for c in cs]
    print(f"all\t{len(tot)}\t{sum(1 for c in tot if c in stock and stock[c][1] == 'OK')}\t"
          f"{sum(1 for c in tot if c in sparse and sparse[c][1] == 'OK')}\t"
          f"{sum(1 for c in tot if c in stock and c in sparse and stock[c][2] != sparse[c][2])}")
    print()
    print(f"id\tnb\tkv\tn_kv_max\tnh\tnr23\ttype_K\tpermute\t{stock_run}\t{sparse_run}\terror {stock_run}\t"
          f"error {sparse_run}\tsame errors")
    for nb, cs in sorted(groups.items()):
        if nb <= 128:
            continue
        for c in cs:
            s = stock.get(c, ("", "missing", ""))
            m = sparse.get(c, ("", "missing", ""))
            print("\t".join([s[0] or m[0], str(nb), field(c, "kv"), field(c, "n_kv_max"), field(c, "nh"),
                             field(c, "nr23"), field(c, "type_K"), field(c, "permute"), s[1], m[1],
                             s[2].split(",")[0], m[2].split(",")[0], "yes" if s[2] == m[2] else "no"]))
    print()
    print("run\tnb\tfailing case")
    for r, data in ((stock_run, stock), (sparse_run, sparse)):
        for c, (_, v, _) in data.items():
            if v != "OK":
                print(f"{r}\t{field(c, 'nb')}\t{c}")


if __name__ == "__main__":
    main()
