#!/usr/bin/env python3
"""R88 (Q1315): which kernel took each multi-token case of the qsa-union replay, against the rule.

Reads replay_checks.sh's OUT/<run>.tsv (per case: id, case string, verdict, each node's error) and the rule's table
(rule.txt, its last table: per shape the kernel each replay run gives, d the dense kernels, S the per-token sparse
kernel, U the union kernel). A case is "as stock" in a run when every node's error, to the digits printed, equals its
error in qu-stock. The rule holds for a case when it is as stock where the table says d and not as stock where it says
S or U. The mode-2 runs (qu-m2-union-s1, -s32, qu-fault1) take qu-m2-union's column. Cases of two or more query
tokens the table does not list are counted apart, by what they did. With a second OUT folder (REF, e.g. the compute_61
PTX replay of the same set), each run's per-case "as stock" pattern is compared with that folder's for the runs both
hold.

Usage: kernel_classes.py <OUT> <rule.txt> [<REF OUT>]
Prints: a table per run (cases by rule class, those that hold, those that do not), then every case where the rule does
not hold, then the unlisted multi-token cases, then the comparison with REF.
"""
import re
import sys
from pathlib import Path

COLUMN = {"qu-sparse1": "qp-sparse1", "qu-union": "qu-union", "qu-uprompt": "qu-uprompt", "qu-ur05": "qu-ur05",
          "qu-m2": "qu-m2", "qu-m2-union": "qu-m2-union", "qu-m2-union-s1": "qu-m2-union",
          "qu-m2-union-s32": "qu-m2-union", "qu-fault1": "qu-m2-union"}
RUNS = ["qu-sparse1", "qu-union", "qu-uprompt", "qu-ur05", "qu-m2", "qu-m2-union", "qu-m2-union-s1",
        "qu-m2-union-s32", "qu-fault1", "qu-fault2", "qu-fault3", "qu-fault4"]


def load(path):
    cases = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split("\t")
        if len(f) >= 3 and f[1].startswith("FLASH_ATTN_EXT"):
            # upstream b12360 added q_head_padding to the case string; at 0 it is the case recorded before
            cases[re.sub(r",q_head_padding=0\)$", ")", f[1])] = (f[2], f[3] if len(f) > 3 else "")
    return cases


def key(case):
    g = lambda k: re.search(rf"\b{k}=(\d+)", case).group(1)
    nr = re.search(r"nr23=\[(\d+),(\d+)\]", case)
    return (int(nr.group(1)), int(g("nh")), int(g("nb")), int(nr.group(2)), int(g("kv")), int(g("n_kv_max")))


def rule_table(path):
    lines = path.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("G, KV heads"))
    cols = lines[head].split("n_kv_max", 1)[1].split()
    table = {}
    for l in lines[head + 1:]:
        m = re.match(r"^(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\s+(.*)$", l)
        if not m:
            break
        table[tuple(int(x) for x in m.groups()[:6])] = dict(zip(cols, m.group(7).split()))
    return table


def main():
    out, table = Path(sys.argv[1]), rule_table(Path(sys.argv[2]))
    ref = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    stock = load(out / "qu-stock.tsv")
    multi = [c for c in stock if key(c)[2] >= 2]
    listed = [c for c in multi if key(c) in table]
    print(f"qu-stock: {len(stock)} cases, {len(multi)} of two or more query tokens, {len(listed)} of them in the rule's table")
    print("run\trule column\td: as stock\tS: not as stock\tU: not as stock\tholds\tdoes not hold")
    bad, extra = [], {}
    for r in RUNS:
        p = out / f"{r}.tsv"
        if not p.exists():
            continue
        cases = load(p)
        col = COLUMN.get(r)
        if col:
            n = {"d": [0, 0], "S": [0, 0], "U": [0, 0]}
            for c in listed:
                if c not in cases:
                    continue
                k = table[key(c)][col]
                same = cases[c][1] == stock[c][1]
                ok = same if k == "d" else not same
                n[k][0] += ok
                n[k][1] += 1
                if not ok:
                    bad.append((r, k, "as stock" if same else "not as stock", c))
            held = sum(v[0] for v in n.values())
            tot = sum(v[1] for v in n.values())
            print(f"{r}\t{col}\t" + "\t".join(f"{n[k][0]}/{n[k][1]}" for k in "dSU") + f"\t{held}\t{tot - held}")
        for c in multi:
            if c not in listed and c in cases:
                extra.setdefault(c, {})[r] = "stock" if cases[c][1] == stock[c][1] else "other"
    print()
    print("run\trule\tobserved\tcase where the rule does not hold")
    for b in bad:
        print("\t".join(b))
    print(f"({len(bad)} cases)")
    print()
    print("multi-token cases outside the rule's table: as stock (s) or other errors (o) per run")
    runs = [r for r in RUNS if (out / f"{r}.tsv").exists()]
    print("case\t" + "\t".join(runs))
    for c, v in extra.items():
        print(c + "\t" + "\t".join({"stock": "s", "other": "o"}.get(v.get(r, ""), "-") for r in runs))
    if ref:
        print()
        rstock = load(ref / "qu-stock.tsv")
        print("run\tcases in both\tsame as-stock pattern as REF\tdiffer")
        for r in ["qu-stock"] + RUNS:
            if not (ref / f"{r}.tsv").exists() or not (out / f"{r}.tsv").exists():
                continue
            a, b = load(out / f"{r}.tsv"), load(ref / f"{r}.tsv")
            both = [c for c in a if c in b and c in stock and c in rstock]
            diff = [c for c in both if (a[c][1] == stock[c][1]) != (b[c][1] == rstock[c][1])]
            print(f"{r}\t{len(both)}\t{len(both) - len(diff)}\t{len(diff)}")
            for c in diff:
                print(f"  differs: {c}")


if __name__ == "__main__":
    main()
