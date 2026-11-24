#!/usr/bin/env python3
"""C948: every job of every check_identity.sh table in run.sh's output folder, as one TSV.

Usage: python3 -I tables.py <run.sh's WORK/out> > checks.tsv
Each <step>-<model>-<build>.log holds check_identity.sh's table; summary.tsv names the checks in the order they ran.
"""
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
cols = ["run", "n_gen", "guess", "hits", "misses", "wasted", "tok/s", "tokens", "logits", "state", "probe",
        "winner", "race", "verdict"]
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
w.writerow(["step", "model", "build", "job", "n_gen", "guesses", "hits", "misses", "wasted", "tokens", "logits",
            "state", "probe", "verdict"])
with open(out / "summary.tsv", newline="") as f:
    checks = [(r["step"], r["model"], r["build"]) for r in csv.DictReader(f, delimiter="\t")]
for step, model, build in checks:
    rows = []
    started = False
    for line in (out / f"{step}-{model}-{build}.log").read_text(errors="replace").splitlines():
        parts = line.split()
        if parts[:1] == ["run"] and parts[-1:] == ["verdict"]:
            started = True
            continue
        if started and len(parts) == len(cols) and parts[-1] in ("same", "DIFF"):
            rows.append(dict(zip(cols, parts)))
    if not rows:
        w.writerow([step, model, build, "(no table)"] + [""] * 10)
    for r in rows:
        w.writerow([step, model, build, r["run"], r["n_gen"], r["guess"], r["hits"], r["misses"], r["wasted"],
                    r["tokens"], r["logits"], r["state"], r["probe"], r["verdict"]])
