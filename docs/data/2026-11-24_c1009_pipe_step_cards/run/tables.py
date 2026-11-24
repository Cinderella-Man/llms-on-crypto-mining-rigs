#!/usr/bin/env python3
"""C1009: tables from run.sh's rounds (WORK/out/<round>/summary.tsv and WORK/out/<round>.log).

Usage: python3 -I tables.py <WORK/out> <folder for the tables>
Writes pipe.tsv (every verdict of every round), hashes.tsv (each host-sampled identity check's plain hashes),
times.tsv (each tree's time in each round) and prints the Markdown tables of the README: the verdicts per tree, arm and
layout; whether a setting changed the plain run's logits against another on the same model and layout (which kernels
computed them); how many plain logits hashes each arm gave over the trees and layouts; the times.
"""
import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROUNDS = ["step", "reach", "virt"]
ARCHS = ["qwen4exp", "qwen35", "qwen35moe", "qwen3next", "flint"]
CHECKS = ["host", "fed", "recurrent"]


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def short(v):
    return {"OK": "OK", "DIFFERS": "**DIFFERS**", "FAIL": "**FAIL**", "SKIP": "skip", "INFO": "info"}.get(v, v)


def main():
    out, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in ROUNDS:
        p = out / r / "summary.tsv"
        if p.exists():
            for row in read_summary(p):
                row["round"] = r
                rows.append(row)

    with open(dest / "pipe.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["round", "tree", "arm", "layout", "arch", "check", "verdict", "detail"])
        for x in rows:
            w.writerow([x["round"], x["tree"], x["arm"], x["layout"], x["arch"], x["check"], x["verdict"], x["detail"]])

    host = [x for x in rows if x["check"] == "host"]
    with open(dest / "hashes.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["round", "tree", "arm", "layout", "arch", "verdict", "plain tokens hash", "plain logits hash"])
        for x in host:
            w.writerow([x["round"], x["tree"], x["arm"], x["layout"], x["arch"], x["verdict"],
                        x["plain tokens hash"], x["plain logits hash"]])

    # verdicts: one table per round, rows arch x check, columns tree / arm / layout
    for r in ROUNDS:
        rr = [x for x in rows if x["round"] == r]
        if not rr:
            continue
        cols = []
        for x in rr:
            k = (x["tree"], x["arm"], x["layout"])
            if k not in cols and x["arch"] != "all":
                cols.append(k)
        print(f"\n#### Round `{r}`\n")
        print("| model | check | " + " | ".join(f"{t} {a} {l}" for t, a, l in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for a in ARCHS:
            for c in CHECKS:
                cells = []
                for t, arm, l in cols:
                    m = [x for x in rr if (x["tree"], x["arm"], x["layout"], x["arch"], x["check"]) == (t, arm, l, a, c)]
                    if not m:
                        cells.append("—")
                        continue
                    x = m[0]
                    cell = short(x["verdict"])
                    mm = re.search(r"differing (\d+) of (\d+)", x["detail"])
                    if mm and c != "recurrent":
                        cell += f" ({mm.group(1)}/{mm.group(2)})"
                    if "refused by design" in x["detail"]:
                        cell += " ¹"
                    cells.append(cell)
                print(f"| {a} | {c} | " + " | ".join(cells) + " |")
        kern = [x for x in rr if x["check"] == "kernels"]
        for x in kern:
            print(f"\nkernels, {x['tree']} {x['arm']} {x['layout']}: {x['detail']}")
        other = [x for x in rr if x["arch"] == "all" and x["check"] != "kernels"]
        for x in other:
            print(f"\n{x['tree']} {x['arm']} {x['layout']} {x['check']}: {x['verdict']} {x['detail']}")

    # which kernels: on the same tree, models, layout and architecture, a setting's plain logits hash against another's;
    # other bits mean other kernels computed some product (each kernel sums in its own order)
    print("\n#### Plain logits: setting against setting (host sampling)\n")
    base = {}
    for x in host:
        kind, setting = x["arm"].split("-", 1)
        base.setdefault((x["tree"], kind, x["layout"], x["arch"]), {})[setting] = x["plain logits hash"]
    pairs = [("p106", "stock"), ("p106all", "p106")]
    print("| tree | models | layout | " + " | ".join(ARCHS) + " |")
    print("|---|---|---|" + "---|" * len(ARCHS))
    for t, k, l in sorted({(t, k, l) for t, k, l, _ in base}):
        cells = []
        for a in ARCHS:
            d = base.get((t, k, l, a), {})
            parts = [f"{x} vs {y}: " + ("same" if d[x] == d[y] else "**other**") for x, y in pairs if x in d and y in d]
            cells.append("; ".join(parts) or "—")
        if any(c != "—" for c in cells):
            print(f"| {t} | {k} | {l} | " + " | ".join(cells) + " |")

    # the same plain run everywhere: distinct plain logits hashes per models, setting and architecture over the trees
    # and layouts
    print("\n#### Plain logits across trees and layouts (host sampling)\n")
    groups = {}
    for x in host:
        groups.setdefault((x["arm"], x["arch"]), []).append(x)
    print("| arm | " + " | ".join(ARCHS) + " |")
    print("|---|" + "---|" * len(ARCHS))
    for arm in sorted({g[0] for g in groups}):
        cells = []
        for a in ARCHS:
            g = groups.get((arm, a), [])
            hs = {x["plain logits hash"] for x in g}
            cells.append(f"{len(hs)} hash over {len(g)} runs" if g else "—")
        print(f"| {arm} | " + " | ".join(cells) + " |")

    # times: each tree's span in each round's log (from its line to the next tree's line or the done line)
    times = []
    for r in ROUNDS:
        p = out / f"{r}.log"
        if not p.exists():
            continue
        marks = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"(\d\d:\d\d:\d\d) (\w+): /", line)
            if m:
                marks.append((m.group(2), m.group(1)))
            m = re.match(r"(\d\d:\d\d:\d\d) done:", line)
            if m:
                marks.append(("done", m.group(1)))
        for (t, s), (_, e) in zip(marks, marks[1:]):
            d = datetime.strptime(e, "%H:%M:%S") - datetime.strptime(s, "%H:%M:%S")
            if d < timedelta(0):
                d += timedelta(days=1)
            times.append((r, t, s, e, int(d.total_seconds())))
    with open(dest / "times.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["round", "tree", "start", "end", "seconds"])
        w.writerows(times)
    print("\n#### Time per tree\n")
    print("| round | tree | start | end | seconds |")
    print("|---|---|---|---|---|")
    for r, t, s, e, d in times:
        print(f"| {r} | {t} | {s} | {e} | {d} |")


if __name__ == "__main__":
    main()
