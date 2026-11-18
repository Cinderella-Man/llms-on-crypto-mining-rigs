#!/usr/bin/env python3
"""R202b: the use profile for rig 2's next boot, from the core ladder's soaked.json.

  python3 -I make_use.py WORK/soak/soaked.json R201_USE.json R202_STEPS.csv r202b-use.json > use-choice.md

The rule (bench/mem_offset/README.md, "Holding a profile"): each card's highest clean rung, but no higher than two
climb steps under its lowest failure anywhere, in a climb or a soak. Here only the core changed between rungs: every
card's memory sat at R201's use profile (R201_USE.json) in all three rungs, so the memory stays at that value, which is
already capped by R200's and R201's memory failures. A climb step is two of the driver's core steps (bins); R202's
steps.csv gives the MHz of every bin count the climb set. Core caps from earlier runs: R200's core climb
(docs/data/2026-11-17_r200_climb_rig2: cards 0, 4 and 8 failed at +307, +282 and +282 MHz, the 2,177 MHz target, with
memory at +1,400 MT/s, read as the core's edge) and R202's core climb (docs/data/2026-11-18_r202_core_climb_rig2:
cards 1, 2, 3, 5, 6, 7 failed at +307, +320, +282, +256, +332, +307 MHz with memory at the use profile). R200's failures
of cards 2 and 3 at +16 and +104 MHz (memory at +1,400) and R200's first soak failure of card 4 at +1,300 MT/s /
+231 MHz are read as memory errors, as R201's use profile and R202 read them, and set no core cap.
"""
import csv
import json
import sys

EARLIER = {0: (307, "R200 climb"), 1: (307, "R202 climb"), 2: (320, "R202 climb"), 3: (282, "R202 climb"),
           4: (282, "R200 climb"), 5: (256, "R202 climb"), 6: (332, "R202 climb"), 7: (307, "R202 climb"),
           8: (282, "R200 climb")}
STEP_BINS = 2

src, use_path, steps_path, out = sys.argv[1:5]
with open(src) as f:
    soaked = json.load(f)
with open(use_path) as f:
    r201 = {c["card"]: c for c in json.load(f)["cards"]}
mhz = {0: 0}                                   # driver core steps (bins) -> the MHz the climb set for them
with open(steps_path) as f:
    for row in csv.DictReader(f):
        mhz[int(row["core_steps"])] = int(row["core_mhz"])
bins = {v: k for k, v in mhz.items()}


def under(core, n):
    """The core value n climb steps under `core` (MHz), 0 at the bottom."""
    b = bins[core] - n * STEP_BINS
    return mhz[b] if b > 0 else 0


use = json.loads(json.dumps(soaked))
use["what"] = ("R202b's use profile for rig 2: each card's memory at R201's use profile, its core at the highest "
               "clean rung of the core soak ladder, at most two climb steps under its lowest core failure anywhere "
               "(R200's and R202's climbs, R202b's ladder)")
use["from"] = "R202b soaked.json, by results/2026-11-18_r202b_core_soak/make_use.py"
print("| Card | Memory (R201's use) | Highest clean rung (core) | First differed in R202b | Core cap and why | "
      "**Use (MT/s, MHz)** |")
print("|---|---|---|---|---|---|")
for c in use["cards"]:
    g = c["card"]
    held_mem, held_core = c["memory_mts"], c["core_mhz"]
    fd = c.get("soak", {}).get("first_differed")
    fail, why = EARLIER[g]
    if fd and fd["core_mhz"] and fd["core_mhz"] <= fail:
        fail, why = fd["core_mhz"], f"R202b rung {fd['rung']}"
    cap = under(fail, 2)
    mem = r201[g]["memory_mts"]
    if held_mem != mem:
        note = f" (soaked.json: +{held_mem:,} MT/s; no clean rung, so the core goes to 0)"
        core = 0
    else:
        note = ""
        core = min(held_core, cap)
    c["memory_mts"], c["core_mhz"], c["core_steps"] = mem, core, bins[core]
    c.pop("soak", None)
    fds = (f"rung {fd['rung']}, round {fd['round']}, at +{fd['core_mhz']}" if fd else "never")
    print(f"| {g} | +{mem:,} | +{held_core}{note} | {fds} | +{cap} (two steps under {why}'s failure at +{fail}) | "
          f"**+{mem:,}, +{core}** |")
with open(out, "w") as f:
    json.dump(use, f, indent=1)
