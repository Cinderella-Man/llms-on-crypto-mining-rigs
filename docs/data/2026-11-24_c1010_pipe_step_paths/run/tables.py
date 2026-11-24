#!/usr/bin/env python3
"""C1010: the tables of this folder from run.sh's work folder, copied back without builds, models and logits.

Usage: python3 -I tables.py <W> <out dir>
  pipe.tsv, pipe_times.tsv, checks.tsv, ctest.tsv  copied from <W>/checks (run.sh's compare step writes them)
  identity_jobs.tsv  every check_identity.sh job of both trees, one line each, from <W>/cpu/<tree>/pipe/<model>-<arm>.log
  race.tsv           every race job's winner and race tokens (both samples' tokens until the stop), from its results.txt
  hand.tsv           compare_hand.py's three comparisons (C935's tip against tip, C935's fix against fix, tip against fix)
  models.tsv         the pipe step's random models' md5 in both trees
"""
import re
import shutil
import sys
from pathlib import Path

TREES = ("tip", "fix")
ARMS = ("identity", "sampled", "race")

w, out = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("pipe.tsv", "pipe_times.tsv", "checks.tsv", "ctest.tsv"):
    shutil.copy(w / "checks" / name, out / name)

# the models in the order the step checked them (its summary lines)
models = []
for line in (w / "checks" / "pipe.tsv").read_text().splitlines()[1:]:
    m = line.split("\t")[1]
    if m not in models:
        models.append(m)

with open(out / "identity_jobs.tsv", "w") as f:
    f.write("tree\tmodel\tarm\trun\tn_gen\tguesses\thits\tmisses\twasted\ttokens\tlogits\tstate\tprobe\twinner\trace\tverdict\n")
    for t in TREES:
        for m in models:
            for a in ARMS:
                log = w / "cpu" / t / "pipe" / f"{m}-{a}.log"
                if not log.exists():
                    continue
                lines = log.read_text().splitlines()
                hdr = lines[0].split()
                for line in lines[1:]:
                    v = line.split()
                    if len(v) != len(hdr):
                        continue
                    d = dict(zip(hdr, v))
                    f.write("\t".join([t, m, a] + [d[k] for k in ("run", "n_gen", "guess", "hits", "misses", "wasted",
                                                                  "tokens", "logits", "state", "probe", "winner", "race",
                                                                  "verdict")]) + "\n")

with open(out / "race.tsv", "w") as f:
    f.write("tree\tmodel\trun\trace\twinner\trace_tokens\tn_gen\n")
    for t in TREES:
        for m in models:
            res = w / "cpu" / t / "pipe" / f"{m}-race" / "results.txt"
            if not res.exists():
                continue
            for line in res.read_text().splitlines():
                kv = dict(re.findall(r"(\w+)=(\S+)", line))
                f.write("\t".join([t, m] + [kv.get(k, "-") for k in ("name", "race", "winner", "race_tokens", "n_gen")]) + "\n")

with open(out / "hand.tsv", "w") as f:
    for i, pair in enumerate(("before-tip", "before-fix", "tip-fix")):
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
