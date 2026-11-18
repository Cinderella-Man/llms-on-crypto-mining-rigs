#!/usr/bin/env python3
"""R202b: the three rung profiles of rig 2's core soak ladder.

  python3 -I make_rungs.py USE_PROFILE.json R202_STEPS.csv R202_PROFILE.json OUT_DIR

USE_PROFILE is R201's use profile (r201-use.json): every card's memory in all three rungs, and the core of the cards
that did not climb in R202 (0, 4 and 8: +256, +206, +231 MHz, which held R201b's 30-minute soak). R202_STEPS is the
fair core climb's steps.csv and R202_PROFILE its profile.json: for the cards that climbed (1, 2, 3, 5, 6, 7) the
suggested core is the profile's core_mhz (two climb steps under the card's first failure), and the rungs take the
climb's own core values, read from steps.csv: rung 1 two climb steps under the suggested core, rung 2 one step
under it, rung 3 the suggested core. No rung steps a card down.
"""
import csv
import json
import sys

use_path, steps_path, r202_path, out = sys.argv[1:5]
with open(use_path) as f:
    use = json.load(f)
with open(r202_path) as f:
    r202 = {c["card"]: c for c in json.load(f)["cards"]}
climbed = sorted(g for g, c in r202.items() if c["core"]["took_part"])
ladder = {g: [] for g in climbed}            # card: the climb's (core MHz, driver steps), in the order it took them
with open(steps_path) as f:
    for row in csv.DictReader(f):
        g = int(row["card"])
        if row["phase"] == "core" and g in ladder:
            v = (int(row["core_mhz"]), int(row["core_steps"]))
            if v not in ladder[g]:
                ladder[g].append(v)
rungs = {}                                  # card: [(core, steps) for rungs 1, 2, 3]
for g in climbed:
    vals = ladder[g]
    want = (r202[g]["core_mhz"], r202[g]["core_steps"])
    i = vals.index(want)
    if i < 2 or vals != sorted(vals):
        raise SystemExit(f"card {g}: the climb's core values {vals} do not give two steps under {want}")
    rungs[g] = [vals[i - 2], vals[i - 1], vals[i]]
for n in (1, 2, 3):
    prof = json.loads(json.dumps(use))
    prof["what"] = (f"R202b rung {n} of 3 for rig 2's core soak ladder: every card's memory at r201-use.json's value; "
                    f"cards {', '.join(map(str, climbed))} at {3 - n} climb step(s) under R202's suggested core, "
                    "the other cards at r201-use.json's core")
    prof["from"] = "results/2026-11-18_r202b_core_soak/make_rungs.py: r201-use.json, R202's steps.csv and profile.json"
    prof["climb_boot"] = None
    prof["rungs"] = 3
    for c in prof["cards"]:
        g = c["card"]
        if g in rungs:
            c["core_mhz"], c["core_steps"] = rungs[g][n - 1]
        for k in ("expected_peak_mhz", "r200_soak_note"):
            c.pop(k, None)
    with open(f"{out}/r202b-rung{n}.json", "w") as f:
        json.dump(prof, f, indent=1)
    print(n, [(c["card"], c["memory_mts"], c["core_mhz"], c["core_steps"]) for c in prof["cards"]])
