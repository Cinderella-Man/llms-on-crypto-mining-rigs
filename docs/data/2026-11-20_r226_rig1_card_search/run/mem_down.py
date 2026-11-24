#!/usr/bin/env python3
"""Q1640 (c), Q1760: a rig's profile with every card's memory and/or core offset one or more steps lower.

  python3 -I mem_down.py USE.json OUT.json [--steps 1|N|all] [--core-steps N|all] [--climb CLIMB_PY]

Copies USE.json (a soak's use profile, here R223's use.json for the first rig) to OUT.json with every card's
memory_mts lowered by --steps x 100 MT/s (the kit's memory step) and its core lowered by --core-steps of the kit's
core steps (CORE_STEP_BINS driver steps of CORE_BIN_MHZ each: 2 x 12.66 MHz), each never below 0; `all` takes that
offset to 0 on every card, and --steps 0 keeps every memory offset. Every other field is kept, with `what` and `from`
naming the change. The core is moved in whole driver steps from the card's core_steps, and core_mhz is set from them
by the kit's own rule (climb.py's core_offset(), the offset `climb.py` sets for that many steps), so the two agree
and `apply`'s check accepts them. Without --core-steps the core is kept and climb.py is not read.

  mem_down.py USE OUT                               every memory one step lower, cores kept (Q1640's option (c))
  mem_down.py USE OUT --steps 13                    every memory offset 0 on the first rig, cores kept (R235)
  mem_down.py USE OUT --steps all                   every memory offset 0 on any rig, cores kept
  mem_down.py USE OUT --steps 0 --core-steps all    every core offset 0, memory kept (the memory alone, Q1795)
  mem_down.py USE OUT --steps all --core-steps 1    every core one step lower, every memory offset 0
  mem_down.py USE OUT --core-steps 1                memory and core each one step lower (Q1640's option (c'))

CLIMB_PY is the kit's bench/mem_offset/climb.py: by default the one beside this script (the kit's folder on the rig),
else this repository's. CORE_BIN_MHZ and CORE_STEP_BINS are read from the environment as climb.py reads them (source
the kit's env.sh first on the rig). Then check the profile with the kit's own profile check:

  python3 half_profile.py climb.py OUT.json --check     (on the rig, after sourcing the kit's env.sh)

Sets nothing and needs no rig, GPU or root. The same USE.json and options give the same OUT.json byte for byte, and
the calls without --core-steps give the same bytes as this script did before it had the option.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def count(v):
    if v == "all":
        return v
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v!r} is not a whole number or 'all'")
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("use")
p.add_argument("out")
p.add_argument("--steps", type=count, default=1,
               help="memory steps of 100 MT/s to take off every card, 0 to keep the memory, all for every memory at 0")
p.add_argument("--core-steps", type=count, default=0,
               help="the kit's core steps (2 x 12.66 MHz) to take off every card, all for every core at 0 [0: kept]")
p.add_argument("--climb", default="", help="the kit's climb.py [beside this script, else this repository's]")
a = p.parse_args()
if a.steps == 0 and a.core_steps == 0:
    p.error("--steps 0 needs --core-steps: nothing would change")

with open(a.use, "rb") as f:
    raw = f.read()
prof = json.loads(raw)
name = a.use.rsplit("/", 1)[-1]
src = f"from {name} (md5 {hashlib.md5(raw).hexdigest()})"

if a.core_steps == 0 and a.steps != "all":
    # The memory alone, as before --core-steps existed: the same `what` and `from`, so the same bytes.
    down = 100 * a.steps
    for c in prof["cards"]:
        c["memory_mts"] = max(0, c.get("memory_mts", 0) - down)
    prof["what"] = (f"Q1640 (c): every card's memory {down} MT/s under {name}'s value (never below 0), core as in "
                    f"{name}. Was: {prof.get('what', '')}")
    prof["from"] = f"docs/data/2026-11-20_r226_rig1_card_search/run/mem_down.py --steps {a.steps} {src}"
else:
    bin_mhz = step_bins = climb = None
    if a.core_steps:
        path = a.climb or next((q for q in (os.path.join(HERE, "climb.py"),
                                            os.path.join(HERE, "..", "..", "..", "..", "bench", "mem_offset", "climb.py"))
                                if os.path.exists(q)), "")
        if not path or not os.path.exists(path):
            sys.exit("--core-steps needs the kit's climb.py: none beside this script or in this repository, give --climb")
        spec = importlib.util.spec_from_file_location("climb", path)
        climb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(climb)
        kit = climb.parse(["state"])
        bin_mhz, step_bins = kit.core_bin_mhz, kit.core_step_bins
        bad, have = [], {}
        for c in prof["cards"]:
            mhz, steps = int(c.get("core_mhz") or 0), int(c.get("core_steps") or 0)
            gives = climb.core_bins(mhz, bin_mhz)
            if steps and steps != gives:    # apply's own test: core_steps 0 means "as core_mhz gives"
                bad.append(f"card {c['card']}: core +{mhz} MHz is {gives} driver steps, its core_steps says {steps}")
            have[c["card"]] = gives
        if bad:
            sys.exit(f"{a.use}: core_mhz and core_steps disagree, so the steps to take off are unknown: "
                     + "; ".join(bad))
    for c in prof["cards"]:
        if a.steps == "all":
            c["memory_mts"] = 0
        elif a.steps:
            c["memory_mts"] = max(0, c.get("memory_mts", 0) - 100 * a.steps)
        if a.core_steps == "all":
            c["core_mhz"], c["core_steps"] = 0, 0
        elif a.core_steps:
            n = max(0, have[c["card"]] - a.core_steps * step_bins)
            c["core_mhz"], c["core_steps"] = climb.core_offset(n, bin_mhz), n
    mem = ("every memory offset 0" if a.steps == "all" else f"memory as in {name}" if not a.steps else
           f"every card's memory {100 * a.steps} MT/s under {name}'s value (never below 0)")
    core = ("every core offset 0" if a.core_steps == "all" else f"core as in {name}" if not a.core_steps else
            f"every card's core {a.core_steps} x {step_bins} driver steps of {bin_mhz} MHz under {name}'s "
            f"core_steps (never below 0), core_mhz from them by climb.py's core_offset()")
    prof["what"] = f"Q1760: {mem}; {core}. Was: {prof.get('what', '')}"
    prof["from"] = (f"docs/data/2026-11-20_r226_rig1_card_search/run/mem_down.py --steps {a.steps} "
                    f"--core-steps {a.core_steps} {src}")

with open(a.out, "w") as f:
    json.dump(prof, f, indent=1)
    f.write("\n")
with open(a.out, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"{a.out} (md5 {md5}): memory MT/s, core MHz on cards "
      + ", ".join(f"{c['card']} +{c['memory_mts']}/+{c['core_mhz']}" for c in prof["cards"]))
