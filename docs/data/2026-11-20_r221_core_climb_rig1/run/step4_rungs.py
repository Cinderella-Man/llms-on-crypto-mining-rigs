#!/usr/bin/env python3
"""R221: the three rungs of the first rig's core soak ladder (step 4), from step 2's use profile and step 3's climb.

  python3 -I step4_rungs.py MAKE_RUNGS.py USE.json STEPS.csv PROFILE.json OUT_DIR [--hold CARD ...]

MAKE_RUNGS is R202b's make_rungs.py (docs/data/2026-11-18_r202b_core_soak/run/make_rungs.py), run unchanged: each card
that climbed the core gets three rungs, two climb steps, one climb step and zero steps under its result core (taken
from the climb's own steps), with every card's memory at USE's value; the other cards keep USE's core. It stops on a
climbing card with fewer than two clean core values under its result core. --hold names such cards: a copy of PROFILE
marks them as not having climbed, so they keep USE's core in all three rungs (on the first rig, card 6, whose climb
failed at its first core step, +28 MHz, and so stays at 0) and their memory is soaked like every other card's.

Writes OUT_DIR/step4-rung1.json ... step4-rung3.json: make_rungs.py's files with their `what` and `from` naming these
inputs (make_rungs.py's own text names the second rig's).
"""
import json
import os
import subprocess
import sys
import tempfile

argv = sys.argv[1:]
hold = set()
if "--hold" in argv:
    i = argv.index("--hold")
    hold = {int(c) for c in argv[i + 1:]}
    argv = argv[:i]
make_rungs, use, steps, profile, out = argv
with open(profile) as f:
    prof = json.load(f)
found = set()
for c in prof["cards"]:
    if c["card"] in hold:
        c["core"]["took_part"] = False
        found.add(c["card"])
if found != hold:
    raise SystemExit(f"--hold cards {sorted(hold - found)} are not in {profile}")
climbed = [c["card"] for c in prof["cards"] if c["core"]["took_part"]]
os.makedirs(out, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    held = os.path.join(tmp, "profile.json")
    with open(held, "w") as f:
        json.dump(prof, f, indent=1)
    subprocess.run([sys.executable, "-I", make_rungs, use, steps, held, tmp], check=True)
    for n in (1, 2, 3):
        with open(os.path.join(tmp, f"r202b-rung{n}.json")) as f:
            rung = json.load(f)
        rung["what"] = (f"Step 4 rung {n} of 3 for the first rig's core soak ladder: every card's memory at "
                        f"{os.path.basename(use)}'s value (step 2, R218); cards {', '.join(map(str, climbed))} at "
                        f"{3 - n} climb step(s) under the core climb's result core (step 3, R221), "
                        f"card(s) {', '.join(map(str, sorted(hold))) or 'none'} at the use profile's core")
        rung["from"] = ("docs/data/2026-11-20_r221_core_climb_rig1/run/step4_rungs.py with R202b's make_rungs.py: "
                        "R218's use.json, R221's steps.csv and profile.json"
                        + (f", card(s) {', '.join(map(str, sorted(hold)))} held" if hold else ""))
        with open(os.path.join(out, f"step4-rung{n}.json"), "w") as f:
            json.dump(rung, f, indent=1)
        print(f"step4-rung{n}.json", [(c["card"], c["memory_mts"], c["core_mhz"], c["core_steps"])
                                      for c in rung["cards"]])
