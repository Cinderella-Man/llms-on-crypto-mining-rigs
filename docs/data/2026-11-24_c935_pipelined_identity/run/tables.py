#!/usr/bin/env python3
"""C935: the tables of this folder from run.sh's work folder, copied back without builds, models and logits.

Usage: python3 -I tables.py <W> <out dir>
  pipe.tsv, checks.tsv, ctest.tsv  copied from <W>/checks (run.sh's compare step writes them)
  identity_jobs.tsv                every check_identity.sh job of both trees, one line each, from <W>/cpu/<tree>/pipe/<arch>.log
  hand.tsv                         compare_hand.py's two comparisons (BEFORE against tip, tip against fix), one table
  models.tsv                       the pipe step's random models' md5 in both trees
"""
import shutil
import sys
from pathlib import Path

ARCHS = ("qwen4exp", "qwen35", "qwen35moe", "qwen3next", "flint")
TREES = ("tip", "fix")

w, out = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("pipe.tsv", "checks.tsv", "ctest.tsv"):
    shutil.copy(w / "checks" / name, out / name)

with open(out / "identity_jobs.tsv", "w") as f:
    f.write("tree\tarch\trun\tn_gen\tguesses\thits\tmisses\twasted\ttokens\tlogits\tstate\tprobe\tverdict\n")
    for t in TREES:
        for a in ARCHS:
            lines = (w / "cpu" / t / "pipe" / f"{a}.log").read_text().splitlines()
            hdr = lines[0].split()
            for line in lines[1:]:
                v = line.split()
                if len(v) != len(hdr):
                    continue
                d = dict(zip(hdr, v))
                f.write("\t".join([t, a] + [d[k] for k in ("run", "n_gen", "guess", "hits", "misses", "wasted", "tokens",
                                                           "logits", "state", "probe", "verdict")]) + "\n")

with open(out / "hand.tsv", "w") as f:
    for i, pair in enumerate(("before-tip", "tip-fix")):
        lines = (w / "checks" / f"hand-{pair}.tsv").read_text().splitlines()
        if i == 0:
            f.write("pair\t" + lines[0] + "\n")
        for line in lines[1:]:
            f.write(pair + "\t" + line + "\n")

with open(out / "models.tsv", "w") as f:
    f.write("arch\tfile\t" + "\t".join(f"md5 {t}" for t in TREES) + "\n")
    tabs = []
    for t in TREES:
        tab = {}
        for line in (w / "cpu" / t / "pipe" / "models.md5").read_text().splitlines():
            md5, path = line.split(None, 1)
            tab[path] = md5
        tabs.append(tab)
    for path in tabs[0]:
        f.write("\t".join([path.split("/")[0], path.split("/")[1]] + [tab.get(path, "-") for tab in tabs]) + "\n")
