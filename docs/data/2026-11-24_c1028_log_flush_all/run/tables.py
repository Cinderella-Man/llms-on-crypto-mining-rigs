#!/usr/bin/env python3
# C1028 (2026-11-24): the data folder's tables from run.sh's work folder (or a copy of its logs, checks, run1/ and
# repeat/ folders). Writes hand_pairs.tsv, hand_items.tsv, repeat.tsv, repeat_runs.tsv, eval_sums.tsv, ctest.tsv,
# checks.tsv, pipe.tsv and ccache.tsv into <out dir>; no work-folder paths are written.
# Usage: python3 -I tables.py <work dir or its copy> <out dir>
import re
import sys
from pathlib import Path

w, out = Path(sys.argv[1]), Path(sys.argv[2])
c = w / "checks"
BEFORE = "C1007's run 2 base"
PAIRS = [  # name in run.sh, tree a, tree b
    ("before-base", BEFORE, "base"),
    ("base-head", "base", "head"),
]


def tsv(p: Path) -> list:
    return [ln.split("\t") for ln in p.read_text().splitlines() if ln]


def write(name: str, rows: list) -> None:
    (out / name).write_text("".join("\t".join(str(x) for x in r) + "\n" for r in rows))


# the pairs: compare_hand.py's verdicts per pair, and per item across the pairs
pairs, items, order = [["pair", "tree a", "tree b", "comparisons", "same", "cut short", "differ", "not same"]], {}, []
for name, a, b in PAIRS:
    p = c / f"hand-{name}.tsv"
    if not p.exists():
        continue
    rows = tsv(p)[1:]
    n = {v: sum(1 for r in rows if r[4] == v) for v in ("same", "cut short", "DIFFERS")}
    pairs.append([name, a, b, len(rows), n["same"], n["cut short"], n["DIFFERS"],
                  "; ".join(f"{r[0]} {r[4]}" for r in rows if r[4] != "same") or "-"])
    for r in rows:
        if r[0] not in items:
            items[r[0]] = {"kind": r[1]}
            order.append(r[0])
        items[r[0]][name] = r[4]
write("hand_pairs.tsv", pairs)
names = [n for n, _, _ in PAIRS if (c / f"hand-{n}.tsv").exists()]
write("hand_items.tsv", [["item", "kind"] + names] + [[i, items[i]["kind"]] + [items[i].get(n, "-") for n in names]
                                                       for i in order])

# the repeat step: as run.sh wrote it (every number masked in the comparison and the md5)
write("repeat.tsv", tsv(c / "repeat.tsv"))
# (the missing-model runs' last lines name the work folder: written as $W)
WORK = re.compile(r"/[^\s']*?/run[1N]/")  # numbers are masked: run1 reads runN
write("repeat_runs.tsv", [[WORK.sub("$W/run1/", x) for x in r] for r in tsv(c / "repeat_runs.tsv")])

# the hand logs of eval-callback and thread-safety
write("eval_sums.tsv", [["tree", "eval-callback sums", "eval-callback lines", "md5 of the sums",
                         "thread-safety's last line"]]
      + [[BEFORE if r[0] == "before" else r[0]] + r[1:] for r in tsv(c / "eval_sums.tsv")])

# ctest test by test, the other checks' verdicts: as run.sh wrote them, the column "before" named; the pipe step's
# verdicts (base and head only: the earlier run had no pipe step) in a table of their own
for f in ("ctest.tsv", "checks.tsv"):
    rows = tsv(c / f)
    rows[0] = [(BEFORE if x == "before" else x) for x in rows[0]]
    write(f, rows)
rows = tsv(c / "checks.tsv")
hb, hh = rows[0].index("base"), rows[0].index("head")
summary = {}
for t in ("base", "head"):
    for r in tsv(w / "run1" / "summary.tsv")[1:]:
        if r[0] == t and r[1].startswith("pipe "):
            summary[(t, r[1])] = r[3] if len(r) > 3 else ""
pipe = [["model", "arm", "base", "head", "head's detail"]]
for r in rows[1:]:
    if r[0].startswith("pipe "):
        parts = r[0].split()
        pipe.append([parts[1], " ".join(parts[2:]), r[hb], r[hh], summary.get(("head", r[0]), "")[:200]])
write("pipe.tsv", pipe)

# ccache's hits per build round (run 1 builds both trees from the seed cache; each repeat round rebuilds them)
cc = [["build", "hits", "misses"]]
for r in ("run1", "repeat", "repeat2", "repeat3"):
    t = (c / f"ccache-{r}.txt").read_text()
    h = re.search(r"Hits:\s+(\d+) / (\d+)", t)
    m = re.search(r"Misses:\s+(\d+) / (\d+)", t)
    cc.append([r, f"{h.group(1)} of {h.group(2)}" if h else "-", f"{m.group(1)} of {m.group(2)}" if m else "-"])
write("ccache.tsv", cc)
print("\n".join(sorted(p.name for p in out.glob("*.tsv"))))
