#!/usr/bin/env python3
# C933 (2026-11-24): the data folder's tables from the two rounds' work folders (run.sh's $W, copied back without the
# builds, models and logits files).
#   round 1: cpu_tests.sh with the seeds only (its comparisons redone with the final compare_hand.py in checks/final/);
#   round 2: the seeds and each hand log written from inside its container.
# Writes into <out dir>: hand_pairs.tsv, hand_round1.tsv, hand_round2.tsv, eval_sums.tsv, models.tsv, ctest.tsv,
# checks.tsv, streams.tsv.
# Usage: python3 -I tables.py <round 1 dir> <round 2 dir> <out dir>
import re
import sys
from pathlib import Path

r1, r2, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PAIRS = [("base-twice", "run1/base", "run2/base"), ("head-twice", "run1/head", "run2/head"),
         ("base-head-run1", "run1/base", "run1/head"), ("base-head-run2", "run2/base", "run2/head")]
OLD = ("old-script-twice", "C880 landing's run of 8a9629942", "C914's run of 8a9629942")


def read_tsv(p: Path) -> list:
    lines = p.read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln]


def write(name: str, header: list, rows: list) -> None:
    with (out / name).open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# the pairs, counted
rows = []
sets = [("before C933", OLD[0], OLD[1], OLD[2], r2 / "checks" / "hand-old-script-twice.tsv")]
sets += [("round 1", n, a, b, r1 / "checks" / "final" / f"hand-{n}.tsv") for n, a, b in PAIRS]
sets += [("round 2", n, a, b, r2 / "checks" / f"hand-{n}.tsv") for n, a, b in PAIRS]
tables = {}
for rnd, n, a, b, p in sets:
    t = read_tsv(p)
    tables[(rnd, n)] = t
    v = [r["verdict"] for r in t]
    differ = [r["item"] for r in t if r["verdict"] == "DIFFERS"]
    short = [r["item"] for r in t if r["verdict"] == "cut short"]
    rows.append((rnd, n, a, b, len(t), v.count("same"), v.count("cut short"), v.count("DIFFERS"),
                 ", ".join(short) or "-", ", ".join(differ) or "-"))
write("hand_pairs.tsv", ["round", "pair", "a", "b", "comparisons", "same", "cut_short", "differ", "cut_short_items",
                         "differing_items"], rows)

# item by item, per round
for rnd, name, with_old in (("round 1", "hand_round1.tsv", False), ("round 2", "hand_round2.tsv", True)):
    cols = [n for n, _, _ in PAIRS] + ([OLD[0]] if with_old else [])
    src = {n: tables[(rnd, n)] for n, _, _ in PAIRS}
    if with_old:
        src[OLD[0]] = tables[("before C933", OLD[0])]
    items = []
    for n in cols:
        for r in src[n]:
            if (r["item"], r["kind"]) not in items:
                items.append((r["item"], r["kind"]))
    out_rows = []
    for item, kind in items:
        cells = []
        for n in cols:
            r = next((x for x in src[n] if x["item"] == item and x["kind"] == kind), None)
            if r is None:
                cells.append("-")
            elif r["verdict"] == "same" and kind.startswith("log"):
                cells.append(f"same ({r['a']})")
            elif r["verdict"] in ("cut short", "DIFFERS") and (kind.startswith("log") or kind == "sums"):
                cells.append(f"{r['verdict']} ({r['a']} / {r['b']})")
            else:
                cells.append(r["verdict"])
        out_rows.append([item, kind] + cells)
    write(name, ["item", "kind"] + cols, out_rows)

# eval-callback's sums, as each run printed them (each round's eval_sums.tsv: tree, count, md5 of the list)
rows = []
for rnd, d in (("round 1", r1), ("round 2", r2)):
    for ln in (d / "checks" / "eval_sums.tsv").read_text().splitlines():
        tree, n, h = ln.split("\t")
        if tree.startswith("BEFORE"):
            if rnd == "round 2":
                continue
            tree = {"BEFORE_A": "before C933: C880 landing's run", "BEFORE_B": "before C933: C914's run"}[tree]
            log = None
        else:
            log = d / tree.replace("-", "/") / "hand" / "eval-callback.log"
        lines = len(log.read_text(errors="replace").splitlines()) if log else "-"
        last = "-"
        if log:
            sums = [m.group(1) for m in re.finditer(r"sum = (\S+)", log.read_text(errors="replace"))]
            last = sums[-1] if sums else ""
        rows.append((rnd if log else "-", tree, n, lines, h, last))
write("eval_sums.tsv", ["round", "tree", "sums_printed", "log_lines", "md5_of_the_sums", "last_sum"], rows)

# the models' checksums, per tree
rows = []
for rnd, d in (("round 1", r1), ("round 2", r2)):
    for tree in ("run1/base", "run1/head", "run2/base", "run2/head"):
        for ln in (d / tree / "hand" / "models.md5").read_text().splitlines():
            h, _, f = ln.partition("  ")
            rows.append((rnd, tree, f, h))
write("models.tsv", ["round", "tree", "model", "md5"], rows)

# ctest and the other checks: round 2's tables as run.sh wrote them
(out / "ctest.tsv").write_text((r2 / "checks" / "ctest.tsv").read_text())
(out / "checks.tsv").write_text((r2 / "checks" / "checks.tsv").read_text())
# the stream test (streams.sh), as it wrote it
(out / "streams.tsv").write_text((r1 / "streams" / "streams.tsv").read_text())
print("written:", ", ".join(sorted(p.name for p in out.glob("*.tsv"))))
