#!/usr/bin/env python3
# C983 (2026-11-24): the data folder's tables from a copy of run.sh's work folder (without sources, builds and models).
# Writes into <data dir>: checks.tsv (cpu_tests.sh's verdicts), ctest.tsv, types.tsv, cli.tsv, override.tsv, sweep.tsv;
# prints the counts the README quotes.
# Usage: python3 -I c983_tables.py <data dir> <copy of the work folder>
import csv
import os
import re
import sys


def read_tsv(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(rows)


def main():
    out, raw = sys.argv[1], sys.argv[2]
    ck = os.path.join(raw, "checks")

    summary = read_tsv(os.path.join(raw, "cpu", "summary.tsv"))
    write_tsv(os.path.join(out, "checks.tsv"), summary)
    print("cpu_tests.sh:")
    for r in summary[1:]:
        print("  " + " | ".join(c[:150] for c in r))

    ctest = read_tsv(os.path.join(ck, "ctest.tsv"))
    write_tsv(os.path.join(out, "ctest.tsv"), ctest)
    body = ctest[1:]
    print(f"ctest: {len(body)} tests; passed base {sum(r[1] == 'Passed' for r in body)}, "
          f"head {sum(r[2] == 'Passed' for r in body)}; differ: {[r for r in body if r[3] != 'yes']}")
    print("  not passed in either: " + "; ".join(f"{r[0]} {r[1]}/{r[2]}" for r in body if r[1] != "Passed" or r[2] != "Passed"))

    types = read_tsv(os.path.join(ck, "types.tsv"))
    write_tsv(os.path.join(out, "types.tsv"), types)
    tb = types[1:]
    differ = [r for r in tb if r[7] != "0" or r[2] != r[3]]
    print(f"types: {len(tb)} files quantized by both builds; rc != 0: base {sum(r[2] != '0' for r in tb)}, "
          f"head {sum(r[3] != '0' for r in tb)}; tensors {sum(int(r[4]) for r in tb)}; differ in {len(differ)} files:")
    for r in differ:
        print(f"  {r[0]}/{r[1]}: rc {r[2]}/{r[3]}, quantized {r[5]}/{r[6]} of {r[4]}, {r[7]} differ: {r[8][:300]}")
    for name in ("qwen4exp-moe", "qwen35moe-moe", "qwen35moe-dense", "flint-dense", "flint-moe"):
        for r in tb:
            if r[1] == name:
                print(f"  {r[0]}/{name}: {r[5]}/{r[6]} of {r[4]} quantized (base/head), {r[7]} differ")

    cli = read_tsv(os.path.join(ck, "cli.tsv"))
    write_tsv(os.path.join(out, "cli.tsv"), [["build", "model", "rc", "first error line"]] + cli)
    print("cli:")
    for r in cli:
        print("  " + " | ".join(r))
    override = read_tsv(os.path.join(ck, "override.tsv"))
    write_tsv(os.path.join(out, "override.tsv"), override)
    print("override:")
    for r in override:
        print("  " + " | ".join(r))

    sweep = {}
    for t in ("base", "head"):
        for r in read_tsv(os.path.join(ck, f"sweep-{t}.tsv"))[1:]:
            rows = r[2] if len(r) > 2 else ""
            sweep.setdefault(r[0], {})[t] = (r[1], rows)
    rows = [["arch", "rc_base", "rc_head", "base", "head"]]
    for a in sorted(sweep):
        b = sweep[a].get("base", ("missing", ""))
        h = sweep[a].get("head", ("missing", ""))
        rows.append([a, b[0], h[0], b[1], h[1]])
    write_tsv(os.path.join(out, "sweep.tsv"), rows)
    ran = [r for r in rows[1:] if re.search(r"\|(OK|FAIL|ERROR)", r[3] + r[4]) or r[1] != "0" or r[2] != "0"]
    print(f"sweep: {len(rows) - 1} architectures; with a model tested: {len(ran)}; rc 0 base {sum(r[1] == '0' for r in rows[1:])}, "
          f"head {sum(r[2] == '0' for r in rows[1:])}")
    print("  not rc 0 in a build:")
    for r in rows[1:]:
        if r[1] != "0" or r[2] != "0":
            print(f"    {r[0]}: base rc {r[1]} {r[3][:160]} | head rc {r[2]} {r[4][:160]}")


if __name__ == "__main__":
    main()
