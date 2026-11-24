#!/usr/bin/env python3
# C933 (2026-11-24): compares the hand step of two cpu_tests.sh trees line by line: two runs of one tree, or a base and
# a head. Since C933 the hand step makes its random models with a fixed seed (HAND_SEED), so the same code gives the
# same logs, and any line that differs comes from the code.
#   models        hand/models.md5, file by file (the random models' checksums);
#   hand verdicts the "hand ..." rows of each tree's summary.tsv (verdict and exit code; the detail holds the log's
#                 last line, compared below);
#   logs          every hand/*.log and hand/random/*.log, line by line, after: the log's time stamp
#                 ("0.00.009.685 I ") dropped, the work folder and the tree's tag written as <WORK> and <TAG>, and the
#                 numbers masked in timing lines ("reserve took 3 ms"); thread-safety's lines are compared as a
#                 sorted list, because its contexts run in threads and print in the order they finish;
#   eval sums     the "sum = " lines of eval-callback.log, one by one.
# Writes <out.tsv> (item, kind, a, b, verdict, detail) and prints a summary; exit 1 if anything differs.
# Usage: python3 -I compare_hand.py <tree dir a> <tree dir b> <out.tsv>   (a tree dir is cpu_tests.sh's $WORK/<tag>)
import re
import sys
from pathlib import Path

STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ?")
TIMING = re.compile(r"( ms\b| ms$|ms per token|tokens per second| t/s|time =|seconds)")
NUMBER = re.compile(r"\d+(\.\d+)?")
SUM = re.compile(r"^\s*sum = (\S+)")
ORDERLESS = {"thread-safety"}


def kept(path: Path, tree: Path) -> list:
    if not path.exists():
        return []
    work, tag = str(tree.parent), tree.name
    out = []
    for ln in path.read_text(errors="replace").splitlines():
        ln = STAMP.sub("", ln).rstrip()
        ln = ln.replace(f"{work}/{tag}/", "<WORK>/<TAG>/").replace(work, "<WORK>")
        if TIMING.search(ln):
            ln = NUMBER.sub("N", ln)
        out.append(ln)
    return out


def md5s(tree: Path) -> dict:
    p = tree / "hand" / "models.md5"
    if not p.exists():
        return {}
    return {name.strip(): h for h, _, name in (ln.partition(" ") for ln in p.read_text().splitlines() if ln.strip())}


def hand_rows(tree: Path) -> dict:
    p = tree.parent / "summary.tsv"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        f = ln.split("\t")
        if len(f) >= 4 and f[0] == tree.name and f[1].startswith("hand "):
            rc = re.match(r"\S+ (rc=\d+)", f[3])
            out[f[1]] = f"{f[2]} {rc.group(1) if rc else ''}".strip()
    return out


def main() -> int:
    a, b, tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []

    ma, mb = md5s(a), md5s(b)
    for name in sorted(set(ma) | set(mb)):
        x, y = ma.get(name, "missing"), mb.get(name, "missing")
        rows.append((name, "model md5", x, y, "same" if x == y else "DIFFERS", ""))
    if not ma and not mb:
        rows.append(("models.md5", "model md5", "missing", "missing", "DIFFERS", "no hand/models.md5 in either tree"))

    ha, hb = hand_rows(a), hand_rows(b)
    for k in sorted(set(ha) | set(hb)):
        x, y = ha.get(k, "missing"), hb.get(k, "missing")
        rows.append((k, "verdict", x, y, "same" if x == y else "DIFFERS", ""))

    logs = sorted({p.relative_to(a) for p in (a / "hand").glob("*.log")} | {p.relative_to(a) for p in (a / "hand").glob("random/*.log")}
                  | {p.relative_to(b) for p in (b / "hand").glob("*.log")} | {p.relative_to(b) for p in (b / "hand").glob("random/*.log")})
    for rel in logs:
        if not (a / rel).exists() or not (b / rel).exists():
            rows.append((str(rel.with_suffix("")).removeprefix("hand/"), "log",
                         "present" if (a / rel).exists() else "missing", "present" if (b / rel).exists() else "missing",
                         "DIFFERS", "the log is in one tree only"))
            continue
        la, lb = kept(a / rel, a), kept(b / rel, b)
        last = next((x for x in reversed(lb) if x.strip()), "")
        how = "log"
        if rel.stem in ORDERLESS and rel.parent.name == "hand":
            la, lb, how = sorted(la), sorted(lb), "log, sorted"
        n = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
        first = next((f"line {i + 1}: {x!r} / {y!r}" for i, (x, y) in enumerate(zip(la, lb)) if x != y),
                     f"line {min(len(la), len(lb)) + 1}: one log ends" if len(la) != len(lb) else "")
        item = str(rel.with_suffix("")).removeprefix("hand/")
        rows.append((item, how, f"{len(la)} lines", f"{len(lb)} lines", "same" if n == 0 else "DIFFERS",
                     last[:150] if n == 0 else f"{n} lines differ; first {first[:300]}"))

    sa = [m.group(1) for m in map(SUM.match, kept(a / "hand/eval-callback.log", a)) if m]
    sb = [m.group(1) for m in map(SUM.match, kept(b / "hand/eval-callback.log", b)) if m]
    n = sum(1 for x, y in zip(sa, sb) if x != y) + abs(len(sa) - len(sb))
    first = next((f"sum {i + 1}: {x} / {y}" for i, (x, y) in enumerate(zip(sa, sb)) if x != y), "")
    rows.append(("eval-callback sums", "sums", f"{len(sa)} sums", f"{len(sb)} sums",
                 "same" if n == 0 and sa else "DIFFERS",
                 (f"first {sa[0]}, last {sa[-1]}" if sa else "no sums") if n == 0 else f"{n} differ; first {first}"))

    with tsv.open("w") as f:
        f.write("item\tkind\ta\tb\tverdict\tdetail\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    bad = [r for r in rows if r[4] != "same"]
    print(f"{a} against {b}: {len(rows)} comparisons, {len(rows) - len(bad)} same, {len(bad)} differ")
    for r in bad:
        print("  " + " | ".join(str(x) for x in r))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
