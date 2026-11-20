#!/usr/bin/env python3
"""R226: a part of a rig's profile, for finding by halves the card that spoiled the served model's whole-memory check.

  python3 -I half_profile.py CLIMB_PY USE.json OUT.json --stock 5 6 7 8 [--tag a]
  python3 -I half_profile.py CLIMB_PY PROFILE.json --check

CLIMB_PY is the kit's bench/mem_offset/climb.py. The first form copies USE.json (a soak's use profile, here R223's
use.json for the first rig) to OUT.json with the --stock cards' memory_mts, core_mhz and core_steps set to 0; every
other card keeps USE's values, and every other field (pci, name, failures ...) is kept, with `what` and `from` naming
the half. Both forms then run the kit's own profile check, the one `climb.py apply` runs before it starts any process
(check_apply: every card present, whole offsets not below 0 and under the caps, core_steps that core_mhz gives, a pci
and a name for each card), with the kit's options from the environment as `apply` reads them (source the kit's env.sh
first on the rig), and print what `apply` would set on each card. Exit 0: the check passed; 1: refused.

Sets nothing and needs no rig, GPU or root.
"""
import argparse
import hashlib
import importlib.util
import json
import sys

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("climb_py")
p.add_argument("profile", help="USE.json to copy from, or with --check the profile to check")
p.add_argument("out", nargs="?", help="the half profile to write")
p.add_argument("--stock", type=int, nargs="+", default=[], help="cards set to 0 memory and 0 core")
p.add_argument("--tag", default="", help="the half's name in `what` (a, b ...)")
p.add_argument("--check", action="store_true", help="only check PROFILE")
a = p.parse_args()
if a.check == bool(a.out) or (a.out and not a.stock):
    p.error("either USE.json OUT.json --stock CARD ..., or PROFILE --check")

spec = importlib.util.spec_from_file_location("climb", a.climb_py)
climb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(climb)

path = a.profile
if a.out:
    with open(a.profile, "rb") as f:
        raw = f.read()
    prof = json.loads(raw)
    have = {c["card"] for c in prof["cards"]}
    if not set(a.stock) <= have:
        sys.exit(f"--stock cards {sorted(set(a.stock) - have)} are not in {a.profile}")
    kept = []
    for c in prof["cards"]:
        if c["card"] in a.stock:
            c["memory_mts"], c["core_mhz"], c["core_steps"] = 0, 0, 0
        elif c.get("memory_mts") or c.get("core_mhz"):
            kept.append(c["card"])
    stock = sorted(a.stock)
    prof["what"] = (f"R226{a.tag}: half of the card search; cards {', '.join(map(str, kept))} at "
                    f"{a.profile.rsplit('/', 1)[-1]}'s values, cards {', '.join(map(str, stock))} at 0 memory and "
                    f"0 core (stock). Was: {prof.get('what', '')}")
    prof["from"] = (f"docs/data/2026-11-20_r226_rig1_card_search/run/half_profile.py --stock "
                    f"{' '.join(map(str, stock))} from {a.profile.rsplit('/', 1)[-1]} (md5 "
                    f"{hashlib.md5(raw).hexdigest()})")
    with open(a.out, "w") as f:
        json.dump(prof, f, indent=1)
        f.write("\n")
    path = a.out

args = climb.parse(["apply", path])
try:
    prof, cards, plan = climb.check_apply(args)
except climb.Refused as e:
    print(f"REFUSED: {e}")
    sys.exit(1)
with open(path, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"{path} (md5 {md5}): the kit's profile check passed (memory cap +{args.mem_cap} MT/s, core cap "
      f"+{args.core_cap} MHz, {args.core_bin_mhz} MHz a driver step); apply would set:")
by = {c["card"]: c for c in prof["cards"]}
for g in cards:
    q = plan[g]
    print(f"  card {g} ({by[g]['pci']}): memory +{q['memory_mts']} MT/s, core +{q['core_mhz']} MHz "
          f"({q['core_steps']} steps)" + ("  stock" if not q["memory_mts"] and not q["core_mhz"] else ""))
