#!/usr/bin/env python3
# C1007 (2026-11-24): the data folder's tables from run.sh's work folder (or a copy of its logs, checks, run*/ and
# repeat/ folders). Writes hand_pairs.tsv, hand_items.tsv, repeat.tsv, repeat_runs.tsv, eval_sums.tsv, ctest.tsv,
# checks.tsv and ccache.tsv into <out dir>; no work-folder paths are written.
# Usage: python3 -I tables.py <work dir or its copy> <out dir>
import hashlib
import re
import sys
from pathlib import Path

w, out = Path(sys.argv[1]), Path(sys.argv[2])
c = w / "checks"
PAIRS = [  # name in run.sh, tree a, tree b
    ("before-base-run1", "landing's run of the base", "run 1 base"),
    ("base-twice", "run 1 base", "run 2 base"),
    ("head-twice", "run 1 head", "run 2 head"),
    ("base-head-run1", "run 1 base", "run 1 head"),
    ("base-head-run2", "run 2 base", "run 2 head"),
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

# the repeat step: the summary, and each run with the md5 of its log without time stamps
write("repeat.tsv", tsv(c / "repeat.tsv"))
STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ?")
runs = tsv(c / "repeat_runs.tsv")
rows = [runs[0][:7] + ["log md5 (time stamps dropped)", "last line"]]
for r in runs[1:]:
    log = w / "repeat" / r[0] / f"{r[1]}-{r[2]}.log"
    text = "\n".join(STAMP.sub("", x).rstrip() for x in log.read_text(errors="replace").splitlines())
    if r[1] == "thread-safety":  # several threads write it: its lines as a sorted list
        text = "\n".join(sorted(text.splitlines()))
    rows.append(r[:7] + [hashlib.md5(text.encode()).hexdigest()[:12], r[7].strip() if len(r) > 7 else ""])
write("repeat_runs.tsv", rows)

# the test runs' hand logs of eval-callback and thread-safety
write("eval_sums.tsv", [["tree", "eval-callback sums", "eval-callback lines", "md5 of the sums",
                         "thread-safety's last line"]]
      + [["landing's run of the base" if r[0] == "before" else r[0].replace("-", " ")] + r[1:]
         for r in tsv(c / "eval_sums.tsv")])

# ctest test by test, the other checks' verdicts: as run.sh wrote them, the column "before" named
for f in ("ctest.tsv", "checks.tsv"):
    rows = tsv(c / f)
    rows[0] = [("landing's run of the base" if x == "before" else x) for x in rows[0]]
    write(f, rows)

# ccache's hits per run
cc = [["build", "hits", "misses"]]
for r in ("run1", "run2", "repeat"):
    t = (c / f"ccache-{r}.txt").read_text()
    h = re.search(r"Hits:\s+(\d+) / (\d+)", t)
    m = re.search(r"Misses:\s+(\d+) / (\d+)", t)
    cc.append([r, f"{h.group(1)} of {h.group(2)}" if h else "-", f"{m.group(1)} of {m.group(2)}" if m else "-"])
write("ccache.tsv", cc)
print("\n".join(sorted(p.name for p in out.glob("*.tsv"))))
