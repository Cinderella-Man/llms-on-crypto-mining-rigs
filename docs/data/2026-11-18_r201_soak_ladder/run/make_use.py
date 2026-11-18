#!/usr/bin/env python3
"""R201: the use profile for rig 2's next boot, from the ladder's soaked.json.

  python3 -I make_use.py WORK/soak/soaked.json use-profile.json > use-choice.md

The rule (bench/mem_offset/README.md, "Holding a profile"): each card's highest clean rung, but no higher than two
memory steps (200 MT/s) under its lowest failure anywhere, in a climb or a soak. A card that first differed in the
ladder with its core at 0 gets a new cap 200 MT/s under the memory it differed at; a card that first differed with a
core offset set keeps its memory and goes to core 0. Memory caps from R200 (docs/data/2026-11-17_r200_climb_rig2):
card 1 +1,000, cards 2, 4 and 6 +1,100, card 5 +1,200, card 7 +900; cards 0, 3 and 8 had no memory failure of their
own there (their only R200 failures came in the core climb, with memory at +1,400).
"""
import json
import sys

STEP = 200
R200_CAP = {1: (1000, "R200 climb: failed at +1,200"), 2: (1100, "R200 soak try 2: differed at +1,300"),
            4: (1100, "R200 soak try 1: differed at +1,300 (core +231)"), 5: (1200, "R200 climb: failed at +1,400"),
            6: (1100, "R200 climb: failed at +1,300"), 7: (900, "R200 soak tries 3 and 4: differed at +1,200 and +1,100")}

src, out = sys.argv[1], sys.argv[2]
with open(src) as f:
    soaked = json.load(f)
use = json.loads(json.dumps(soaked))
use["what"] = ("R201's use profile for rig 2: each card's highest clean rung of the soak ladder, at most two memory "
               "steps under its lowest failure anywhere (R200's climb and soaks, R201's ladder)")
use["from"] = "R201 soaked.json, by results/2026-11-18_r201_soak_ladder/make_use.py"
print("| Card | Highest clean rung (MT/s, MHz) | First differed in R201 | Cap and why | Use (MT/s, MHz) |")
print("|---|---|---|---|---|")
for c in use["cards"]:
    g = c["card"]
    mem, core = held_mem, held_core = c["memory_mts"], c["core_mhz"]
    steps = c.get("core_steps", 0)
    fd = c.get("soak", {}).get("first_differed")
    cap, why = R200_CAP.get(g, (None, "no failure of its own in R200"))
    if fd and fd["core_mhz"] == 0:          # a memory rung: a new failure, a new cap
        new = fd["memory_mts"] - STEP
        if cap is None or new < cap:
            cap, why = new, f"R201 rung {fd['rung']}: differed at +{fd['memory_mts']:,}"
    if fd and fd["core_mhz"] != 0:          # rung 4: the core's failure; the core goes to 0
        core, steps = 0, 0
    if cap is not None and mem > cap:
        mem = cap
    c["memory_mts"], c["core_mhz"], c["core_steps"] = mem, core, steps
    c.pop("soak", None)
    fds = (f"rung {fd['rung']}, round {fd['round']}, at +{fd['memory_mts']:,} / +{fd['core_mhz']}" if fd else "never")
    caps = f"+{cap:,} ({why})" if cap is not None else f"none ({why})"
    print(f"| {g} | +{held_mem:,}, +{held_core} | {fds} | {caps} | +{mem:,}, +{core} |")
with open(out, "w") as f:
    json.dump(use, f, indent=1)
