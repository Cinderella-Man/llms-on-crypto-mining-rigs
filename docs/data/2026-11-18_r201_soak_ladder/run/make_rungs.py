#!/usr/bin/env python3
"""R201: the four rung profiles of rig 2's soak ladder, from R200's last profile (profile-try4.json).

  python3 -I make_rungs.py R200_LAST_PROFILE.json OUT_DIR    # R200 try 4's profile; its offsets are replaced

Rung 3 is R200's last values with card 7 at +1,000 MT/s (try 5 as planned on 2026-11-17), core 0; rungs 1 and 2 are
two and one memory step (100 MT/s) under it; rung 4 adds the core offsets that held R200's soaks.
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
TOP = [1300, 1000, 1200, 1300, 1200, 1200, 1100, 1000, 1300]          # cards 0 to 8, MT/s
CORE = {0: (256, 20), 3: (54, 4), 4: (206, 16), 8: (231, 18)}         # card: (MHz, driver steps)
with open(src) as f:
    base = json.load(f)
for n, (down, core) in enumerate(((200, False), (100, False), (0, False), (0, True)), 1):
    prof = json.loads(json.dumps(base))
    prof["what"] = (f"R201 rung {n} of 4 for rig 2's soak ladder: memory {down} MT/s under R200's last values"
                    + (", with the core offsets that held R200's soaks" if core else ", core 0"))
    for c in prof["cards"]:
        g = c["card"]
        c["memory_mts"] = TOP[g] - down
        c["core_mhz"], c["core_steps"] = CORE.get(g, (0, 0)) if core else (0, 0)
        c.pop("expected_peak_mhz", None)
    with open(f"{out}/r201-rung{n}.json", "w") as f:
        json.dump(prof, f, indent=1)
    print(n, [(c["memory_mts"], c["core_mhz"]) for c in prof["cards"]])
