#!/usr/bin/env python3
"""Q1640 (c): a rig's profile with every card's memory offset one or more steps lower, core offsets kept.

  python3 -I mem_down.py USE.json OUT.json [--steps 1]

Copies USE.json (a soak's use profile, here R223's use.json for the first rig) to OUT.json with every card's
memory_mts lowered by --steps x 100 MT/s (the kit's memory step), never below 0; core_mhz and core_steps and every
other field are kept, with `what` and `from` naming the change. Then check it with the kit's own profile check:

  python3 half_profile.py climb.py OUT.json --check     (on the rig, after sourcing the kit's env.sh)

Sets nothing and needs no rig, GPU or root. The same USE.json gives the same OUT.json byte for byte.
"""
import argparse
import hashlib
import json

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("use")
p.add_argument("out")
p.add_argument("--steps", type=int, default=1, help="memory steps of 100 MT/s to take off every card")
a = p.parse_args()
if a.steps < 1:
    p.error("--steps must be 1 or more")

with open(a.use, "rb") as f:
    raw = f.read()
prof = json.loads(raw)
down = 100 * a.steps
for c in prof["cards"]:
    c["memory_mts"] = max(0, c.get("memory_mts", 0) - down)
name = a.use.rsplit("/", 1)[-1]
prof["what"] = (f"Q1640 (c): every card's memory {down} MT/s under {name}'s value (never below 0), core as in "
                f"{name}. Was: {prof.get('what', '')}")
prof["from"] = (f"docs/data/2026-11-20_r226_rig1_card_search/run/mem_down.py --steps {a.steps} from {name} "
                f"(md5 {hashlib.md5(raw).hexdigest()})")
with open(a.out, "w") as f:
    json.dump(prof, f, indent=1)
    f.write("\n")
with open(a.out, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"{a.out} (md5 {md5}): memory MT/s, core MHz on cards "
      + ", ".join(f"{c['card']} +{c['memory_mts']}/+{c['core_mhz']}" for c in prof["cards"]))
