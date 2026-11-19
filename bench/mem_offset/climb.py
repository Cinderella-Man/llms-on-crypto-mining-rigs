#!/usr/bin/env python3
"""climb.py: every card's memory and core clock offsets found in one sitting, and the result applied on a fresh boot.

Part of bench/mem_offset (its README has the rules, the commands and how to read the result). All cards climb at
once: memory first with the core at 0, then the core by target peak clock. At every step every card runs its own
greedy check (temperature 0, every token's top-10 probabilities hashed) on its own llama-server, then a short
mmvq_pascal check, each under a time limit of LIMIT_FACTOR x the same check's stock time. A card that fails is set
back to 0 and leaves the climb while the others go on; its last clean step is recorded. Out comes a profile with each
card's offsets one step under its last clean step.

  climb.py [--dry-run [--scenario FILE]] [options] COMMAND

  state             read-only: load processes, every card's memory, power limit, P-state and both offsets, Xid lines
                    this boot, host memory, the watchdog, and any step down this boot (the ledger)
  start-check       read-only: exit 0 only if no load process runs, every card uses 0 MiB, has its default power limit
                    and both offsets at 0, and no offset stepped down since the last boot
  plan              read-only: the memory steps, and each core step's target peak and every card's core offset
  trip-test         the watchdog's four trips before any offset, each with stand-in processes: temperature, power and
                    a kernel-log line each set every card to 0 and kill every stand-in; a kernel-log line naming one
                    card kills only that card's stand-ins
  watchdog start|stop
                    the watchdog on every card, detached (scripts/nvml_watchdog.py --per-card-xid)
  climb             detached: the climb (needs the watchdog); writes out/climb.log, out/steps.csv, out/climb.json and
                    out/profile.json; ends with every offset of every card at 0
  --from-profile FILE climb
                    detached: the core climb in its own boot, with no memory climb. Every card's offsets go straight to
                    FILE's values (a climb's profile.json or soak.py's soaked.json), one check round there for every
                    card, then the core climb for the cards FILE gives no core value (or --core-cards), each card's
                    memory held at FILE's value. The same files out; out/profile.json keeps each card's memory at
                    FILE's value. `plan` with --from-profile prints what it would set and climb
  profile           write out/profile.json again from out/climb.json
  rungs PROFILE [--failures FILE] [--out DIR] [--prefix rung]
                    the soak ladder's rung profiles (rung1.json ... rung4.json and rungs-choice.md) from a profile
                    and each card's known failures; sets and runs nothing, needs no rig. Memory rungs two steps, one
                    step and no step under each card's top value (the profile's, lowered to two steps under the
                    card's lowest memory failure), core 0; then a rung that adds the core (the profile's, lowered to
                    two climb steps under the card's lowest core failure)
  apply PROFILE     detached: on a freshly started rig (needs the watchdog), every card's offsets from PROFILE, read
                    back, then each card's greedy check against its own stock result; any failure sets every offset of
                    every card to 0. On success the offsets stay; the watchdog must keep running (it holds the cards).
                    Each offset is the profile's, never more: the core is never derived from this boot's stock peak
  reset-all         every offset of every card to 0, read back
  reset-card N      both offsets of card N to 0, read back; the step down goes into the ledger (soak.py uses it for a
                    card that leaves a soak while the others go on)
  stop             a running climb or apply stops at its next check: every offset to 0, the profile written

Needs root on the rig, except --dry-run, which prints every command it would run and sets nothing: no root, no GPU,
no NVML. --scenario FILE gives the simulated cards' failure points (see test/). Exit status: 0 done or passed, 1 a
check failed or a refusal, 2 usage.

Options take their defaults from the environment variable of the same name in capitals (--check-model: CHECK_MODEL).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOADS = {"llama-server", "mmvq_pascal", "cuda_memtest", "llama-bench", "llama-perplexit", "llama-completi",
         "llama-cli", "test-backend-o"}
PCI_RE = re.compile(r"\(PCI:([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})(?:\.[0-7])?\)")
XID_CODE_RE = re.compile(r"Xid \([^)]*\):\s*(\d+)")
TICK = 0.5          # seconds between looks at the running checks
XID_EVERY = 5.0     # seconds between kernel-log scans while checks run
STOCK_LIMIT = 900   # seconds any stock check may take (no reference yet)
STOCK_ROUNDS_MAX = 12   # stock rounds at most, however short they are, while the stock peak window fills
TEST_LINE = "clock-offset watchdog test line C871WDTEST"


class GlobalStop(Exception):
    """Ends the climb or apply: every offset of every card goes to 0."""


class Refused(Exception):
    """A precondition failed before any offset was set."""


# ---------------------------------------------------------------- small helpers (pure, tested in test/test_climb.py)

def pci_key(text):
    """'00000000:05:00.0', '0000:05:00' or an Xid line's '(PCI:0000:05:00)' -> (domain, bus, device) as ints."""
    m = PCI_RE.search(text) or re.search(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})", text)
    return tuple(int(v, 16) for v in m.groups()) if m else None


def xid_card(line, bus_of):
    """The card an Xid line names, from {card: pci key}; None when the line names no known card."""
    m = PCI_RE.search(line)
    if not m:
        return None
    key = tuple(int(v, 16) for v in m.groups())
    for g, k in bus_of.items():
        if k == key:
            return g
    return None


def xid_code(line):
    m = XID_CODE_RE.search(line)
    return m.group(1) if m else "?"


def parse_get(text):
    """nvml_memoc.py get: 'gpu 0 MemClk vf offset: 1200 (allowed ...)' -> 1200 (None if absent)."""
    m = re.search(r"vf offset: (-?\d+)", text or "")
    return int(m.group(1)) if m else None


def core_offset(bins, bin_mhz):
    """The MHz to pass for `bins` whole steps of the core clock. The driver drops the remainder of an offset (R194:
    +249 MHz gave 19 steps of ~12.65 MHz), so n steps plus a quarter step lands on n steps whether it drops or rounds."""
    return 0 if bins <= 0 else int(round(bins * bin_mhz + bin_mhz / 4))


def core_bins(mhz, bin_mhz):
    """The whole core steps an offset of `mhz` gives: the inverse of core_offset() (soak.py's, the same rule)."""
    return 0 if mhz <= 0 else int(round((mhz - bin_mhz / 4) / bin_mhz))


def stock_bin(peak, bin_mhz):
    return int(round(peak / bin_mhz))


def parse_greedy(text, rc, timed_out, limit, ref):
    """greedy_check.py's last line -> (ok, hash, reason)."""
    lines = [ln for ln in (text or "").splitlines() if ln.startswith("greedy ")]
    last = lines[-1] if lines else ""
    m = re.search(r"hash ([0-9a-f]{16})", last)
    h = m.group(1) if m else None
    if timed_out:
        return False, h, f"greedy longer than its limit ({limit:.0f} s)"
    if rc == 0 and h and (ref is None or "SAME as" in last):
        return True, h, ""
    if rc == 1 or "DIFFERS" in last:
        return False, h, "greedy output differs from stock"
    if rc == 3 or "request failed" in last:
        return False, h, "greedy request failed" + (f" ({last.split('request failed: ', 1)[-1]})" if "request failed: " in last else "")
    return False, h, f"greedy exit {rc}"


def parse_mmvq(csv_text, log_text, rc, timed_out, limit):
    """mmvq_pascal's CSV (or, when it never finished, its log) -> (ok, passes, fails, stream GB/s, reason)."""
    passes = fails = 0
    gbs = None
    if csv_text:
        for row in csv.DictReader(csv_text.splitlines()):
            if row.get("check") == "PASS":
                passes += 1
            elif row.get("check") == "FAIL":
                fails += 1
            if row.get("shape") == "STREAM_READ_PLAIN" and row.get("gbs"):
                gbs = float(row["gbs"])
    else:
        for ln in (log_text or "").splitlines():
            if ln.rstrip().endswith(" PASS"):
                passes += 1
            elif ln.rstrip().endswith(" FAIL"):
                fails += 1
    err = next((ln.strip() for ln in (log_text or "").splitlines() if "CUDA error" in ln or "illegal" in ln), "")
    reasons = []
    if fails:
        reasons.append(f"{fails} wrong kernel result{'s' if fails > 1 else ''}")
    if timed_out:
        reasons.append(f"mmvq longer than its limit ({limit:.0f} s), {passes} checks passed before")
    elif rc != 0:
        reasons.append(f"mmvq exit {rc}" + (f" ({err})" if err else ""))
    elif passes == 0:
        reasons.append("mmvq ran no check")
    ok = not reasons
    return ok, passes, fails, gbs, "; ".join(reasons)


def one_under(passed):
    """The value one step under the last clean one: the step before it in the card's own passed list."""
    return passed[-2] if len(passed) >= 2 else passed[0]


def profile_plan(prof, cards, core_cards, mem_cap, core_cap, bin_mhz):
    """--from-profile: what each card gets from a profile (a climb's profile.json or soak.py's soaked.json).

    cards: the run's cards; core_cards: the cards named to climb the core, or None for the default, every card the
    profile gives no core value. A card that climbs the core starts it at 0, whatever core value the profile has.
    Returns ({card: {"memory_mts", "core_mhz", "core_steps", "core_climb", "profile_core_mhz"}}, [why it cannot run]):
    a card missing from the profile, a value that is not a whole number or is below 0, a value above the caps, a card
    the profile cannot be checked against (no pci or name), or a --core-cards card that is not in the run."""
    by = {}
    for c in (prof.get("cards") if isinstance(prof, dict) else None) or []:
        if isinstance(c, dict) and isinstance(c.get("card"), int):
            by[c["card"]] = c
    plan, bad = {}, []
    if not by:
        return plan, ["the profile has no cards"]
    for g in cards:
        c = by.get(g)
        if c is None:
            bad.append(f"card {g} is not in the profile")
            continue
        try:
            mem, core, steps = (int(c.get(k) or 0) for k in ("memory_mts", "core_mhz", "core_steps"))
        except (TypeError, ValueError):
            bad.append(f"card {g}: memory_mts, core_mhz or core_steps is not a whole number")
            continue
        climb = g in core_cards if core_cards is not None else core == 0
        if min(mem, core, steps) < 0:
            bad.append(f"card {g}: +{mem} MT/s, +{core} MHz: an offset below 0")
        if mem > mem_cap:
            bad.append(f"card {g}: memory +{mem} MT/s is above the cap +{mem_cap} (the top of --mem-steps)")
        if not climb and core > core_cap:
            bad.append(f"card {g}: core +{core} MHz is above --core-cap {core_cap}")
        if not c.get("pci") or not c.get("name"):
            bad.append(f"card {g}: the profile has no pci or name to check the card against")
        if not climb and core and not steps:
            steps = int(core / bin_mhz)
        plan[g] = {"memory_mts": mem, "core_mhz": 0 if climb else core, "core_steps": 0 if climb else steps,
                   "core_climb": climb, "profile_core_mhz": core}
    if core_cards is not None:
        extra = [g for g in core_cards if g not in cards]
        if extra:
            bad.append(f"--core-cards {extra} are not among this run's cards {cards}")
    return plan, bad


def whole(v):
    """A profile's offset: a JSON whole number (None or absent is 0); ValueError for anything else."""
    if v is None:
        return 0
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(v)
    return v


def apply_plan(prof, cards, mem_cap, core_cap, bin_mhz):
    """apply: the offsets each card gets, which are exactly the profile's. The core is the profile's core_mhz and
    never more: it is not derived from this boot's stock peak (that reading moves a driver step between boots and
    inside one, while a given offset gives the same peak on every boot, Q1586), and a profile whose core_steps and
    core_mhz disagree is refused rather than guessed at.
    Returns ({card: {"memory_mts", "core_mhz", "core_steps"}}, [why it cannot run]): a card missing, a value that is
    not a whole number or is below 0, a value above the caps, core_steps that core_mhz does not give, or no pci or
    name to check the card against."""
    by = {}
    for c in (prof.get("cards") if isinstance(prof, dict) else None) or []:
        if isinstance(c, dict) and isinstance(c.get("card"), int):
            by[c["card"]] = c
    if not by:
        return {}, ["the profile has no cards"]
    plan, bad = {}, []
    for g in cards:
        c = by.get(g)
        if c is None:
            bad.append(f"card {g} is not in the profile")
            continue
        try:
            mem, core, steps = (whole(c.get(k)) for k in ("memory_mts", "core_mhz", "core_steps"))
        except ValueError:
            bad.append(f"card {g}: memory_mts, core_mhz or core_steps is not a whole number")
            continue
        if min(mem, core, steps) < 0:
            bad.append(f"card {g}: +{mem} MT/s, +{core} MHz, {steps} steps: an offset below 0")
            continue
        if mem > mem_cap:
            bad.append(f"card {g}: memory +{mem} MT/s is above the cap +{mem_cap} (the top of --mem-steps)")
        if core > core_cap:
            bad.append(f"card {g}: core +{core} MHz is above --core-cap {core_cap}")
        gives = core_bins(core, bin_mhz)
        if steps and steps != gives:
            bad.append(f"card {g}: core +{core} MHz is {gives} of the driver's steps, its core_steps says {steps}: "
                       "the profile disagrees with itself (fix core_mhz or core_steps)")
        if not c.get("pci") or not c.get("name"):
            bad.append(f"card {g}: the profile has no pci or name to check the card against")
        plan[g] = {"memory_mts": mem, "core_mhz": core, "core_steps": gives}
    return plan, bad


def failure_entries(items, where_default):
    """Failure entries ({"card", "memory_mts" and/or "core_mhz", "where"}) checked as soak.py --failures checks them;
    ValueError for an entry that is not one."""
    out = []
    for i, e in enumerate(items):
        ok = isinstance(e, dict) and isinstance(e.get("card"), int) and any(k in e for k in ("memory_mts", "core_mhz"))
        ok = ok and all(isinstance(e[k], int) and not isinstance(e[k], bool) and e[k] >= 0
                        for k in ("memory_mts", "core_mhz") if k in e)
        if not ok:
            raise ValueError(f"failure {i + 1} needs a card and a whole memory_mts or core_mhz: {e!r}")
        out.append(dict({k: e[k] for k in ("card", "memory_mts", "core_mhz") if k in e},
                        where=e.get("where") or where_default))
    return out


def read_failures(path):
    """Earlier failures from a JSON file, the format soak.py --failures reads ({"failures": [...]}, a list, or a
    use.json soak.py wrote) -> [{"card", "memory_mts" and/or "core_mhz", "where"}]; ValueError if one is not."""
    with open(path) as f:
        data = json.load(f)
    items = data.get("failures") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: no list of failures")
    try:
        return failure_entries(items, os.path.basename(path))
    except ValueError as e:
        raise ValueError(f"{path}: {e}")


def profile_failures(prof, source):
    """The failures a profile records itself: each card's memory.failed_at_mts and core.failed_at_mhz (a climb's
    profile.json: a failure in the memory climb counts against the memory, one in the core climb against the core, the
    offset that went up into it; a failure in --from-profile's check round with both offsets set counts against both),
    and the "failures" list of a use.json soak.py wrote. ValueError for an entry of that list that is not one."""
    out = []
    for c in prof.get("cards") or []:
        if not isinstance(c, dict) or not isinstance(c.get("card"), int):
            continue
        g = c["card"]
        m = (c.get("memory") or {}).get("failed_at_mts")
        if isinstance(m, int) and not isinstance(m, bool):
            out.append({"card": g, "memory_mts": m, "where": f"{source}: failed at +{m:,} MT/s"})
        k = (c.get("core") or {}).get("failed_at_mhz")
        if isinstance(k, int) and not isinstance(k, bool):
            out.append({"card": g, "core_mhz": k, "where": f"{source}: failed at +{k} MHz"})
    if isinstance(prof.get("failures"), list):
        out += failure_entries(prof["failures"], source)
    return out


def rung_rule(under):
    n = {1: "one step", 2: "two steps", 3: "three steps"}.get(under, f"{under} steps")
    return ("memory_mts is two steps, one step or no step under the card's top value (rungs 1, 2, 3): the profile's "
            f"memory_mts, lowered to {n} under the lowest value at which the card's memory failed, if that is lower; "
            "core 0. The last rung keeps rung 3's memory and adds the core: the profile's core_mhz, lowered to "
            f"{n} of the climb under the lowest value at which the card's core failed, if that is lower. Memory steps "
            "are memory_step_mts, climb steps core_step_mhz; the failures used are under \"failures\"")


def ladder_rungs(prof, failures, mem_step=100, bin_mhz=12.66, step_bins=2, under=2, source="profile.json"):
    """The soak ladder's rung profiles from a profile and each card's known failures (README, "The ladder's rungs").
    Each card's top memory value is its memory_mts, lowered to `under` memory steps under its lowest memory failure if
    that is lower; rungs 1, 2 and 3 are two steps, one step and no step under the top (never below 0), core 0. Rung 4
    keeps rung 3's memory and adds the core: core_mhz, lowered to `under` climb steps (each `step_bins` of the driver's
    steps) under the card's lowest core failure if that is lower. With no core on any card there is no rung 4.
    Each rung is a copy of the profile with each card's three offsets replaced and its expected peak dropped (it
    depends on the boot's stock reading). Returns ([rung profiles], [one Markdown table row per card])."""
    seen, used = set(), []
    for e in failures:   # one entry per card, offset and value: the first one's "where" is kept
        key = (e["card"], "memory_mts" in e, e.get("memory_mts"), "core_mhz" in e, e.get("core_mhz"))
        if key not in seen:
            seen.add(key)
            used.append(e)
    vals, rows = {}, []
    for c in prof["cards"]:
        g = c["card"]
        mem, core = whole(c.get("memory_mts")), whole(c.get("core_mhz"))
        mf = [e for e in used if e["card"] == g and "memory_mts" in e]
        cf = [e for e in used if e["card"] == g and "core_mhz" in e]
        top, mem_cap, core_cap = mem, None, None
        if mf:
            low = min(mf, key=lambda e: e["memory_mts"])
            mem_cap = (max(0, low["memory_mts"] - under * mem_step), low)
            top = min(top, mem_cap[0])
        steps = core_bins(core, bin_mhz)
        if cf:
            low = min(cf, key=lambda e: e["core_mhz"])
            cap_bins = max(0, core_bins(low["core_mhz"], bin_mhz) - under * step_bins)
            core_cap = (core_offset(cap_bins, bin_mhz), low)
            if steps > cap_bins:
                core, steps = core_cap[0], cap_bins
        mems = [max(0, top - (2 - i) * mem_step) for i in range(3)]
        vals[g] = (mems, core, steps if core else 0)

        def cap_text(cap, unit):
            if cap is None:
                return "none"
            v = cap[1]["memory_mts" if unit == "MT/s" else "core_mhz"]
            return f"+{cap[0]:,} {unit} ({under} steps under +{v:,}, {cap[1]['where']})"
        rows.append(f"| {g} | +{mem:,}, +{whole(c.get('core_mhz'))} | {cap_text(mem_cap, 'MT/s')} | "
                    f"{cap_text(core_cap, 'MHz')} | " + " | ".join(f"+{v:,}" for v in mems) + f" | +{core} |")
    n = 4 if any(v[1] for v in vals.values()) else 3
    rungs = []
    for k in range(1, n + 1):
        p = json.loads(json.dumps(prof))
        under_text = {1: "two memory steps under each card's top value", 2: "one memory step under each card's top "
                      "value", 3: "each card's top memory value"}
        p["what"] = (f"P106 clock-offset soak ladder, rung {k} of {n}, from {source} by bench/mem_offset/climb.py "
                     "rungs: " + (under_text[k] + ", core 0" if k <= 3 else "rung 3's memory with the core added"))
        p["rung"] = {"n": k, "of": n, "from": source, "cap_steps": under, "memory_step_mts": mem_step}
        p["each_card"] = rung_rule(under)
        p["failures"] = used
        for c in p["cards"]:
            mems, core, steps = vals[c["card"]]
            c["memory_mts"] = mems[min(k, 3) - 1]
            c["core_mhz"], c["core_steps"] = (core, steps) if k == 4 else (0, 0)
            c.pop("expected_peak_mhz", None)
        rungs.append(p)
    return rungs, rows


RUNG_TABLE = ["| Card | Profile (MT/s, MHz) | Memory cap | Core cap | Rung 1 | Rung 2 | Rung 3 | Core (rung 4) |",
              "|---|---|---|---|---|---|---|---|"]


def card_profile(c, fp, bin_mhz):
    """One card's profile values and records: (memory_mts, [core MHz, core steps, peak], memory {...}, core {...})."""
    mem_passed = c["mem_passed"] or [0]
    core_passed = c["core_passed"] or [[0, 0, c.get("stock_peak")]]
    f = c["failure"]
    phase = f["phase"] if f else None
    if not fp:
        mem_fail, core_fail = phase == "memory", phase == "core"
        return one_under(mem_passed), one_under(core_passed), {
            "last_clean_mts": mem_passed[-1], "failed_at_mts": f["mem"] if mem_fail else None,
            "limit_found": mem_fail}, {
            "took_part": c["core_took_part"], "last_clean_mhz": core_passed[-1][0],
            "last_clean_peak_mhz": core_passed[-1][2], "failed_at_mhz": f["core"] if core_fail else None,
            "failed_at_peak_mhz": f.get("peak") if core_fail else None, "limit_found": core_fail}
    # --from-profile: the memory is the profile's value if the card passed its check round there, else 0; the core is
    # one step under the last clean step for a card that climbed it, the profile's value for a card that kept it
    src = c["from_profile"]
    clean = c.get("profile_check") == "PASS"
    mem = src["memory_mts"] if clean else 0
    if not clean:
        core = [0, 0, c.get("stock_peak")]
    elif src["core_climb"]:
        core = one_under(core_passed)
    else:
        core = [src["core_mhz"], src["core_steps"], c.get("profile_peak")]
    at_profile = phase == "profile"
    memory = {"from_profile_mts": src["memory_mts"], "profile_check": c.get("profile_check"),
              "last_clean_mts": src["memory_mts"] if clean else None,
              "failed_at_mts": f["mem"] if at_profile else None,
              "limit_found": bool(at_profile and not f["core"])}
    if src["core_climb"]:
        core_fail = phase == "core"
        cr = {"took_part": c["core_took_part"], "from_profile_mhz": src["profile_core_mhz"],
              "last_clean_mhz": core_passed[-1][0] if clean else None,
              "last_clean_peak_mhz": core_passed[-1][2] if clean else None,
              "failed_at_mhz": f["core"] if core_fail else None,
              "failed_at_peak_mhz": f.get("peak") if core_fail else None, "limit_found": core_fail}
    else:
        cr = {"took_part": False, "from_profile_mhz": src["profile_core_mhz"],
              "last_clean_mhz": src["core_mhz"] if clean else None,
              "last_clean_peak_mhz": c.get("profile_peak") if clean else None,
              "failed_at_mhz": f["core"] if at_profile else None, "failed_at_peak_mhz": None, "limit_found": False}
    return mem, core, memory, cr


def build_profile(st):
    """The profile (public-ready JSON) from the climb's state."""
    cfg = st["config"]
    bin_mhz = cfg["core_bin_mhz"]
    fp = st.get("from_profile")
    cards = []
    for c in st["cards"]:
        stock = c.get("stock_peak")
        mem, prof_core, memory, core = card_profile(c, fp, bin_mhz)
        cards.append({
            "card": c["card"], "pci": c["pci"], "name": c["name"], "vbios": c["vbios"],
            "stock_peak_mhz": stock,
            "memory_mts": mem,
            "core_mhz": prof_core[0],
            "core_steps": prof_core[1],
            "expected_peak_mhz": round(stock + prof_core[1] * bin_mhz) if stock else None,
            "memory": memory,
            "core": core,
            "failure": c["failure"],
            "needs_reboot_before_use": bool(c["stepped_down_mid_climb"]),
        })
    out = {
        "what": "P106 clock-offset profile from bench/mem_offset/climb.py",
        "date": st["started"][:10],
        "climb_started": st["started"], "climb_ended": st.get("ended"),
        "complete": st.get("complete", False), "stop_reason": st.get("stop_reason"),
        "climb_boot": st["boot"],
        "driver": st.get("driver"),
        "check_model": st.get("check_model"), "check_model_bytes": st.get("check_model_bytes"),
        "greedy_tokens_per_check": 2 * cfg["greedy_n"],
        "memory_steps_mts": [] if fp else cfg["mem_steps"],
        "core_step_mhz": round(cfg["core_step_bins"] * bin_mhz, 2), "core_bin_mhz": bin_mhz,
        "stock_peak_window_s": st.get("stock_peak_window_s"),
        "units": {"memory_mts": "NVML MemClk VF offset in MT/s; +800 is +10 % of the P106-090's 8,008 MT/s",
                  "core_mhz": "NVML GpcClk VF offset in MHz; the driver moves the core in whole steps of core_bin_mhz"},
        "rules": ["Offsets only go up, and only after a fresh boot: apply refuses a rig that has not rebooted since "
                  "the climb or since any step down.",
                  "A reboot after any step down: a card whose memory offset went down ran about 8 % slow until the "
                  "next boot (Q1563).",
                  "The watchdog always on: apply needs it, and it holds the cards so that the offsets stay."],
        "each_card": "memory_mts and core_mhz are one step under the card's last clean step; a card that failed in "
                     "this sitting was set back to 0 then and needs a reboot before use",
        "cards": cards,
    }
    if fp:
        out["from_profile"] = fp
        out["core_climb_cards"] = [c["card"] for c in st["cards"] if c["from_profile"]["core_climb"]]
        out["each_card"] = ("memory_mts is the value from_profile gave the card, kept if the card passed one check "
                            "round there (0 if it failed); core_mhz is one step under the last clean step for a card "
                            "in core_climb_cards (the core climb ran with the memory at that value), and from_profile's "
                            "value, checked in the same round, for the others; a card that failed in this sitting was "
                            "set back to 0 then and needs a reboot before use")
    return out


def ledger_line(boot, card, mem, core, why):
    return f"{boot} {time.strftime('%Y-%m-%d %H:%M:%S')} card {card} mem {mem}->0 core {core}->0 {why}"


def stepped_down_this_boot(ledger_text, boot):
    return [ln for ln in (ledger_text or "").splitlines() if ln.split(" ", 1)[0] == boot]


def boot_tag(boot_id):
    """A boot's tag in the ledger and the profile: a hash of the kernel's random boot id, not the id itself."""
    return hashlib.sha256(boot_id.strip().encode()).hexdigest()[:12]


def render(cmd, env=None):
    pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    body = cmd if isinstance(cmd, str) else shlex.join(str(a) for a in cmd)
    return f"{pre} {body}" if pre else body


# ---------------------------------------------------------------- executors: the real rig, and the simulated one

class RealProc:
    def __init__(self, p, fh):
        self.p, self.fh, self.pid = p, fh, p.pid

    def poll(self):
        rc = self.p.poll()
        if rc is not None and self.fh:
            self.fh.close()
            self.fh = None
        return rc

    def stop(self, grace=5.0):
        """TERM to the process group, KILL after `grace` s; True when it is gone."""
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(self.p.pid, sig)
            except OSError:
                pass
            try:
                self.p.wait(timeout=wait)
                self.poll()
                return True
            except subprocess.TimeoutExpired:
                continue
        return False


class RealExec:
    dry = False

    def __init__(self, log):
        self.log = log
        self.wd = {}   # the watchdog log: path -> [header index, byte position read so far]

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)

    def run(self, cmd, op=None, env=None, timeout=60):
        e = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
        try:
            p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout,
                               shell=isinstance(cmd, str))
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout} s"
        except OSError as ex:
            return 127, str(ex)

    def spawn(self, cmd, op=None, env=None, out=None):
        e = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
        fh = open(out, "w") if out else None
        p = subprocess.Popen(cmd, env=e, stdout=fh or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        return RealProc(p, fh)

    def read(self, path):
        try:
            with open(path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def exists(self, path):
        return os.path.exists(path)

    def append(self, path, line):
        with open(path, "a") as f:
            f.write(line + "\n")

    def touch(self, path):
        open(path, "a").close()

    def peaks(self, wd_csv, cards, t0, t1):
        """Each card's highest SM and memory clock at P0 in the watchdog's samples between t0 and t1."""
        out = {g: [None, None] for g in cards}
        try:
            f = open(wd_csv)
        except OSError:
            return out
        idx, pos = self.wd.get(wd_csv, (None, 0))
        with f:
            f.seek(pos)   # the windows come in time order: only lines written since the last call are read
            while True:
                ln = f.readline()
                if not ln or not ln.endswith("\n"):
                    break
                self.wd[wd_csv] = (idx, f.tell())
                if ln.startswith("#"):
                    continue
                if idx is None:
                    idx = {k: i for i, k in enumerate(ln.rstrip("\n").split(","))}
                    self.wd[wd_csv] = (idx, f.tell())
                    continue
                v = ln.rstrip("\n").split(",")
                try:
                    t = time.mktime(time.strptime(v[0][:19], "%Y-%m-%d %H:%M:%S"))
                except (ValueError, IndexError):
                    continue
                if t < t0 - 1 or t > t1 + 1:
                    continue
                for g in cards:
                    try:
                        if v[idx[f"g{g}_pstate"]] != "P0":
                            continue
                        sm, mem = int(v[idx[f"g{g}_sm_mhz"]]), int(v[idx[f"g{g}_mem_mhz"]])
                    except (KeyError, ValueError, IndexError):
                        continue
                    out[g][0] = max(out[g][0] or 0, sm)
                    out[g][1] = max(out[g][1] or 0, mem)
        return out


class SimProc:
    def __init__(self, x, done_at, rc, on_done=None):
        self.x, self.done_at, self.rc, self.on_done = x, done_at, rc, on_done
        x.next_pid += 1
        self.pid = x.next_pid
        self.final = None

    def poll(self):
        if self.final is None and self.done_at is not None and self.x.t >= self.done_at:
            if self.on_done:
                self.on_done()
            self.final = self.rc
        return self.final

    def stop(self, grace=5.0):
        if self.final is None:
            self.final = -15
        return True


SCENARIO = {  # the simulated rig of --dry-run: R193's nine cards, nothing fails unless `faults` says so
    "cards": 9,
    "pci": ["00000000:03:00.0", "00000000:04:00.0", "00000000:05:00.0", "00000000:06:00.0", "00000000:07:00.0",
            "00000000:0B:00.0", "00000000:0C:00.0", "00000000:0D:00.0", "00000000:0E:00.0"],
    "stock_peaks": [1860, 1860, 1911, 1911, 1885, 1911, 1885, 1911, 1898],
    "driver_bin_mhz": 12.65,
    "server_s": 45, "greedy_s": 20, "mmvq_s": 8,
    "avail_mb": 1800,
    "faults": [],    # {"card": 4, "mem_min": 1200, "peak_min": 0, "check": "mmvq", "kind": ["wrong", "hang"]}
    "cold_peaks": [],  # {"card": 8, "mhz": 13, "secs": 10}: a window that starts within secs of the run's first check
                       # reads mhz higher, as R200's cool cards did in the first seconds of the first stock round
    "points": [],    # replay: {"card", "mem", "core", "greedy": {...}, "mmvq": {...}, "xid": [...], "peak": ...}
}


class SimExec:
    """Prints every command and answers it from a simulated rig, on a simulated clock: nothing is set or run."""
    dry = True

    def __init__(self, log, scenario=None, fixtures=None):
        self.log = log
        self.s = dict(SCENARIO, **(scenario or {}))
        self.fixtures = fixtures or HERE
        self.t = float(int(time.time()))
        self.next_pid = 40000
        n = self.s["cards"]
        self.off = {g: [0, 0] for g in range(n)}
        self.files = {}
        self.ledger = ""
        self.xids = []          # (time, line)
        self.server_up = {}
        self.commands = []      # every command printed, for the tests
        self.t_first = None     # when the run's first check started (cold_peaks)

    def _print(self, text):
        self.commands.append(text)
        self.log("+ " + text)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def read(self, path):
        return self.files.get(path, "")

    def exists(self, path):
        return path in self.files

    def append(self, path, line):
        self._print(f"append to {path}: {line}")
        self.files[path] = self.files.get(path, "") + line + "\n"

    def touch(self, path):
        self.files.setdefault(path, "")

    def peak(self, g):
        n = int(self.off[g][1] / self.s["driver_bin_mhz"]) if self.off[g][1] > 0 else 0
        return int(round(self.s["stock_peaks"][g] + n * self.s["driver_bin_mhz"]))

    def peaks(self, wd_csv, cards, t0, t1):
        out = {}
        for g in cards:
            pt = self.point(g)
            cold = sum(c["mhz"] for c in self.s.get("cold_peaks", []) if c["card"] == g
                       and self.t_first is not None and t0 <= self.t_first + c["secs"])
            out[g] = [(pt["peak"] if pt and pt.get("peak") else self.peak(g)) + cold, int(4004 + self.off[g][0] / 2)]
        return out

    def point(self, g):
        for p in self.s["points"]:
            if p["card"] == g and p["mem"] == self.off[g][0] and p["core"] == self.off[g][1]:
                return p
        return None

    def fault(self, g, check):
        for f in self.s["faults"]:
            if f["card"] != g or f.get("check", check) not in (check, "any"):
                continue
            if self.off[g][0] >= f.get("mem_min", 0) and self.peak(g) >= f.get("peak_min", 0) \
                    and (f.get("mem_min") or f.get("peak_min")):
                return f
        return None

    def run(self, cmd, op=None, env=None, timeout=60):
        self._print(render(cmd, env))
        op = op or ("other",)
        k = op[0]
        if k == "smi":
            rows = [f"{g}, {self.s['pci'][g]}, NVIDIA P106-090, 86.06.5B.00.0C, 0, 75.00, 75.00, P8, 580.178.04"
                    for g in range(self.s["cards"])]
            return 0, "\n".join(rows) + "\n"
        if k == "get":
            g, dom = op[1], op[2]
            v = self.off[g][0 if dom == "mem" else 1]
            name = "MemClk" if dom == "mem" else "GpcClk"
            return 0, f"gpu {g} {name} vf offset: {v} (allowed (-2000, 2000))\ngpu {g} clock offsets P0: {v}\n"
        if k == "set":
            g, dom, v = op[1], op[2], op[3]
            self.off[g][0 if dom == "mem" else 1] = v
            return 0, f"gpu {g}: set {v}\n"
        if k == "reset":
            g, dom = op[1], op[2]
            self.off[g][0 if dom == "mem" else 1] = 0
            return 0, f"gpu {g}: offset 0\n"
        if k == "boot_id":
            return 0, self.s.get("boot_id", "5f0c4b8e-0000-4000-8000-000000000000") + "\n"
        if k == "ledger":
            return (0, self.ledger) if self.ledger else (1, "")
        if k == "ps":
            return 0, f"1 systemd\n2 python3 nvml_watchdog.py --marker {op[1]}\n" if self.s.get("watchdog", True) \
                else "1 systemd\n"
        if k == "meminfo":
            return 0, f"MemTotal: 3900000 kB\nMemAvailable: {self.s['avail_mb'] * 1024} kB\n"
        if k == "journal":
            since = op[1]
            return 0, "".join(line + "\n" for t, line in self.xids if t >= since)
        if k == "health":
            g = op[1]
            up = self.server_up.get(g)
            return (0, "ok") if up is not None and self.t >= up else (7, "")
        if k == "stat":
            return 0, "4081004928\n"
        return 0, ""

    def spawn(self, cmd, op=None, env=None, out=None):
        self._print(render(cmd, env) + (f" > {out}" if out else "") + " &")
        op = op or ("other",)
        k = op[0]
        if k == "server":
            g = op[1]
            self.server_up[g] = self.t + self.s["server_s"]
            return SimProc(self, None, None)
        if k == "greedy":
            if self.t_first is None:
                self.t_first = self.t
            return self._greedy(op[1], op[2], op[3], op[4], out)
        if k == "mmvq":
            return self._mmvq(op[1], op[2], op[3], out)
        return SimProc(self, self.t + 1, 0)

    def _fixture(self, name):
        if not name:
            return ""
        path = os.path.join(self.fixtures, name)
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "rt") as f:
                return f.read()
        with open(path) as f:
            return f.read()

    def _greedy(self, g, tag, ref, card_dir, out):
        pt = self.point(g)
        if pt and "greedy" in pt:
            r = pt["greedy"]
            if r.get("not_recorded"):
                line, rc = f"greedy {tag}: hash 0000000000000000, not recorded in this replay | SAME as {ref}", 0
            else:
                line, rc = r["line"].replace("{tag}", tag), r["rc"]
            s = r.get("s", self.s["greedy_s"])
        else:
            f = self.fault(g, "greedy")
            kinds = f["kind"] if f else []
            h = "d1ffd1ffd1ffd1ff" if "differs" in kinds else "5a1e5a1e5a1e5a1e"
            line = f"greedy {tag}: hash {h}, 512 tokens, prompt tok/s 55.00 60.00, generation tok/s 32.00 32.00"
            rc = 0
            if ref:
                line += f" | DIFFERS from {ref} (5a1e5a1e5a1e5a1e)" if "differs" in kinds else f" | SAME as {ref}"
                rc = 1 if "differs" in kinds else 0
            if "request-failed" in kinds:
                line, rc = f"greedy {tag}: request failed: Remote end closed connection without response", 3
            if "xid" in kinds:
                self.xids.append((self.t + 5, self._xid_line(g)))
            s = None if "hang" in kinds else self.s["greedy_s"]

        def done():
            self.files[out] = line + "\n"
        return SimProc(self, None if s is None else self.t + s, rc, done)

    def _xid_line(self, g, code="31"):
        b = pci_key(self.s["pci"][g])
        return (f"NVRM: Xid (PCI:{b[0]:04x}:{b[1]:02x}:{b[2]:02x}): {code}, pid={self.next_pid}, name=mmvq_pascal, "
                "channel 0x00000018, intr 10000000. MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted")

    def _mmvq(self, g, tag, csv_path, out):
        pt = self.point(g)
        if pt and "mmvq" in pt:
            r = pt["mmvq"]
            for x in pt.get("xid", []):
                self.xids.append((self.t + x["after_s"], x["line"]))
            log_text, csv_text = self._fixture(r.get("log")), self._fixture(r.get("csv"))
            s, rc = r.get("s"), r.get("rc", 0)
            if s is None:   # a hang: what it printed before it stopped is all there is
                self.files[out] = log_text
        else:
            f = self.fault(g, "mmvq")
            kinds = f["kind"] if f else []
            fails = 2 if "wrong" in kinds else 0
            gbs = 179.2 + self.off[g][0] * 0.0212
            head = ("device,code_arch,shape,K,N,experts,used,calls_per_token,type,variant,bytes_per_call,us_median,"
                    "us_min,us_max,gbs,pct_of_ref,ref_gbs,regs,blocks_per_sm,threads_per_block,grid_blocks,"
                    "err_over_tol,check,weight_sets,swept_bytes")
            rows = [f'"NVIDIA P106-090",610,attn_q,2048,4096,0,0,49,q8_0,v{i},8912896,60,59,61,148,85,173.0,64,8,'
                    f'128,4096,{"549.358,FAIL" if i < fails else "0.005,PASS"},8,71303168' for i in range(31)]
            rows.append(f'"NVIDIA P106-090",610,STREAM_READ_PLAIN,0,0,0,0,0,-,-,268435456,,,,{gbs:.2f},118.36,'
                        '173.0,,,,,,,,')
            csv_text = "\n".join([head] + rows) + "\n"
            log_text = "".join(f"attn_q q8_0 v{i} {'549.358 FAIL' if i < fails else '0.005 PASS'}\n" for i in range(31))
            rc, s = 0, self.s["mmvq_s"]
            if "xid" in kinds:
                self.xids.append((self.t + 5, self._xid_line(g)))
                csv_text, rc, s = "", 1, 6
                log_text += ("CUDA error at src/mmvq_pascal/src/main.cu:349: an illegal memory access was encountered\n")
            if "hang" in kinds:
                s = None
                csv_text = ""
                self.files[out] = log_text[: len(log_text) // 4]

        def done():
            self.files[out] = log_text
            if csv_text:
                self.files[csv_path] = csv_text
        return SimProc(self, None if s is None else self.t + s, rc, done)


# ---------------------------------------------------------------- the kit

class Kit:
    def __init__(self, a, x):
        self.a, self.x = a, x
        self.out = os.path.join(a.work, "out")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(os.path.join(a.work, "pids"), exist_ok=True)
        self.logf = os.path.join(self.out, "climb.log")
        self.st = None
        self.servers = {}
        self.bus_of = {}
        self.units = None
        self.any_set = False

    # -- logging and files
    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.x.now()))} {msg}"
        print(line, flush=True)
        with open(self.logf, "a") as f:
            f.write(line + "\n")

    def p(self, *parts):
        return os.path.join(self.out, *parts)

    def card_dir(self, g):
        d = self.p(f"card{g}")
        os.makedirs(d, exist_ok=True)
        return d

    # -- reading the rig
    def smi(self):
        rc, o = self.x.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,name,vbios_version,memory.used,power.limit,"
                            "power.default_limit,pstate,driver_version", "--format=csv,noheader,nounits"], op=("smi",))
        if rc != 0:
            raise Refused(f"nvidia-smi failed: {o.strip()[:200]}")
        cards = {}
        for row in csv.reader(o.strip().splitlines()):
            r = [v.strip() for v in row]
            if len(r) < 9:
                continue
            cards[int(r[0])] = {"pci": r[1], "name": r[2], "vbios": r[3], "used": r[4], "limit": r[5],
                                "default_limit": r[6], "pstate": r[7], "driver": r[8]}
        self.bus_of = {g: pci_key(c["pci"]) for g, c in cards.items()}
        return cards

    def memoc(self, *args):
        return ["python3", self.a.memoc] + [str(v) for v in args]

    def readback(self, g):
        _, o1 = self.x.run(self.memoc("get", g), op=("get", g, "mem"))
        _, o2 = self.x.run(self.memoc("--core", "get", g), op=("get", g, "core"))
        return parse_get(o1), parse_get(o2)

    def boot(self):
        rc, o = self.x.run(["cat", "/proc/sys/kernel/random/boot_id"], op=("boot_id",))
        return boot_tag(o) if rc == 0 and o.strip() else "unknown"

    def ledger_this_boot(self):
        rc, o = self.x.run(["cat", self.a.ledger], op=("ledger",))
        return stepped_down_this_boot(o if rc == 0 else "", self.boot())

    def load_procs(self):
        _, o = self.x.run(["ps", "-eo", "pid=,comm="], op=("ps-comm",))
        return [ln.strip() for ln in o.splitlines() if ln.split() and ln.split()[-1] in LOADS]

    def watchdog_running(self):
        _, o = self.x.run(["ps", "-eo", "pid=,args="], op=("ps", self.marker()))
        return any("nvml_watchdog.py" in ln and f"--marker {self.marker()}" in ln for ln in o.splitlines())

    def marker(self):
        return os.path.join(self.a.work, "TRIPPED")

    def card_marker(self):
        return os.path.join(self.a.work, "CARD_TRIPS")

    def avail_mb(self):
        _, o = self.x.run(["cat", "/proc/meminfo"], op=("meminfo",))
        m = re.search(r"MemAvailable:\s+(\d+) kB", o)
        return int(m.group(1)) // 1024 if m else 0

    def xid_lines(self, since):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
        _, o = self.x.run(["journalctl", "-k", "-q", "--no-pager", "-o", "cat", "--since", stamp],
                          op=("journal", since))
        return [ln for ln in o.splitlines() if "NVRM: Xid" in ln]

    def xids_this_boot(self):
        _, o = self.x.run(["journalctl", "-k", "-b", "-q", "--no-pager", "-o", "cat"], op=("journal", 0))
        return [ln for ln in o.splitlines() if "NVRM: Xid" in ln]

    # -- setting offsets (going up only), resetting
    def load_units(self):
        try:
            with open(self.a.units) as f:
                self.units = json.load(f)
        except (OSError, ValueError) as e:
            raise Refused(f"units file {self.a.units}: {e}")
        if self.units.get("raw_per_mts") not in (1.0, 0.5):
            raise Refused(f"units file {self.a.units}: raw_per_mts {self.units.get('raw_per_mts')}")

    def set_offsets(self, g, mem=None, core=None):
        """Raises one card's offsets (never lowers them) and reads both back from a new process. Returns '' or why
        the card failed."""
        c = self.card(g)
        for dom, v in (("mem", mem), ("core", core)):
            if v is None or v == c[dom]:
                continue
            if v < c[dom]:
                raise GlobalStop(f"card {g}: {dom} {c[dom]} -> {v} would step down (a bug in the climb)")
            cap = self.a.mem_cap if dom == "mem" else self.a.core_cap
            if v > cap:
                raise GlobalStop(f"card {g}: {dom} +{v} is above the cap {cap}")
            if dom == "mem":
                cmd = self.memoc("--state", self.a.units, "--cap", self.a.mem_cap, "set", g, v)
            else:
                cmd = self.memoc("--core", "set", g, v)
            rc, o = self.x.run(cmd, op=("set", g, dom, v), timeout=30)
            c[dom] = v
            self.any_set = True
            if rc != 0:
                return f"the driver refused {dom} +{v} (exit {rc})"
        rb = self.readback(g)
        want = (round(c["mem"] * (self.units or {}).get("raw_per_mts", 1.0)), c["core"])
        if rb != want:
            return f"offsets read back as mem {rb[0]}, core {rb[1]} instead of {want[0]}, {want[1]}"
        return ""

    def reset_card(self, g, why):
        """Both offsets of one card to 0, read back; a step down goes into the ledger. GlobalStop if not 0."""
        c = self.card(g, must=False)
        before = (c["mem"], c["core"]) if c else self.readback(g)
        self.x.run(self.memoc("reset", g), op=("reset", g, "mem"), timeout=30)
        self.x.run(self.memoc("--core", "reset", g), op=("reset", g, "core"), timeout=30)
        rb = self.readback(g)
        if c:
            c["mem"], c["core"], c["bins"] = 0, 0, 0
        if any(before):
            self.x.append(self.a.ledger, ledger_line(self.boot(), g, before[0], before[1], why))
        self.log(f"reset: card {g} offsets now (mem core) {rb[0]} {rb[1]} ({why})")
        if rb != (0, 0):
            raise GlobalStop(f"card {g} does not read back 0 after its reset: {rb}")

    def reset_all(self, why):
        bad = []
        for g in self.a.cards:
            try:
                self.reset_card(g, why)
            except GlobalStop as e:
                bad.append(str(e))
        if bad:
            self.log("RESET FAILED: " + "; ".join(bad) + ". Power the rig off at the console.")
            return False
        return True

    def card(self, g, must=True):
        if self.st is None:
            if must:
                raise RuntimeError("no state")
            return None
        return self.st["cards"][self.a.cards.index(g)] if g in self.a.cards else None

    # -- the per-card check servers
    def port(self, g):
        return self.a.port_base + g

    def write_pids(self, g, extra=()):
        pids = [self.servers[g].pid] if g in self.servers else []
        pids += [p for p in extra if p]
        path = os.path.join(self.a.work, "pids", f"{g}.pids")
        if self.x.dry:
            return
        with open(path, "w") as f:
            f.write("".join(f"{p}\n" for p in pids))

    def start_servers(self, cards):
        """One llama-server per card at offset 0, one after another (each reads the model once through the page cache)."""
        for g in cards:
            self.watch_global()
            cmd = [os.path.join(self.a.bin, "llama-server"), "-m", self.a.check_model] + \
                shlex.split(self.a.server_args) + ["--host", "127.0.0.1", "--port", str(self.port(g))]
            env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": g}
            t0 = self.x.now()
            self.servers[g] = self.x.spawn(cmd, op=("server", g), env=env,
                                           out=os.path.join(self.card_dir(g), "server.log"))
            self.write_pids(g)
            while True:
                self.x.sleep(3)
                if self.health(g):
                    self.log(f"server: card {g} ready in {self.x.now() - t0:.0f} s (port {self.port(g)})")
                    break
                if self.servers[g].poll() is not None:
                    raise Refused(f"the check server on card {g} exited while loading (out/card{g}/server.log)")
                if self.x.now() - t0 > self.a.server_limit:
                    raise Refused(f"the check server on card {g} was not ready after {self.a.server_limit} s")

    def health(self, g):
        rc, _ = self.x.run(["curl", "-sf", "-m", "5", f"http://127.0.0.1:{self.port(g)}/health"], op=("health", g),
                           timeout=10)
        return rc == 0

    def stop_server(self, g):
        s = self.servers.pop(g, None)
        if s and s.poll() is None:
            if not s.stop():
                raise GlobalStop(f"the check server on card {g} does not stop")
        self.write_pids(g)

    def stop_servers(self):
        for g in list(self.servers):
            try:
                self.stop_server(g)
            except GlobalStop as e:
                self.log(f"STOP: {e}")

    # -- checks
    def start_check(self, g, kind, tag, ref):
        d = self.card_dir(g)
        if kind == "greedy":
            cmd = ["python3", self.a.greedy, "--url", f"http://127.0.0.1:{self.port(g)}", "--out", d, "--tag", tag,
                   "--n", str(self.a.greedy_n)] + (["--ref", ref] if ref else [])
            proc = self.x.spawn(cmd, op=("greedy", g, tag, ref, d), out=os.path.join(d, f"greedy-{tag}.out"))
        else:
            csvp = os.path.join(d, f"mmvq-{tag}.csv")
            cmd = [self.a.mmvq] + shlex.split(self.a.mmvq_args) + ["--csv", csvp]
            proc = self.x.spawn(cmd, op=("mmvq", g, tag, csvp), env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                                                     "CUDA_VISIBLE_DEVICES": g},
                                out=os.path.join(d, f"mmvq-{tag}.log"))
        self.write_pids(g, [proc.pid])
        return {"card": g, "kind": kind, "tag": tag, "ref": ref, "proc": proc, "t0": self.x.now()}

    def finish_check(self, ck, rc, timed_out, limit):
        d = self.card_dir(ck["card"])
        el = self.x.now() - ck["t0"]
        if ck["kind"] == "greedy":
            ok, h, why = parse_greedy(self.x.read(os.path.join(d, f"greedy-{ck['tag']}.out")), rc, timed_out, limit,
                                      ck["ref"])
            return {"ok": ok, "hash": h, "why": why, "s": el}
        ok, passes, fails, gbs, why = parse_mmvq(self.x.read(os.path.join(d, f"mmvq-{ck['tag']}.csv")),
                                                 self.x.read(os.path.join(d, f"mmvq-{ck['tag']}.log")), rc, timed_out,
                                                 limit)
        return {"ok": ok, "passes": passes, "fails": fails, "gbs": gbs, "why": why, "s": el}

    def watch_global(self):
        if self.x.exists(self.marker()):
            last = self.x.read(self.marker()).strip().splitlines()
            raise GlobalStop(f"the watchdog tripped ({last[-1] if last else 'see TRIPPED'})")
        if self.x.exists(os.path.join(self.a.work, "STOP")):
            raise GlobalStop("stop requested (STOP file)")

    def run_round(self, cards, tag, ref, limits, on_fail):
        """Every card's checks: the cards in parallel, each card's checks in turn (self.a.order), at most
        mmvq_jobs mmvq processes at once, each check under its limit. on_fail(g, why) is called as soon as a card
        fails; its remaining checks are dropped. Returns {g: {"greedy": {...}, "mmvq": {...}, "ok": bool, "why": str}}."""
        res = {g: {"ok": True, "why": [], "xid": 0} for g in cards}
        queue = {g: [k for k in self.a.order if k != "mmvq" or self.a.mmvq_jobs > 0] for g in cards}
        running = {}
        t_round = self.x.now()
        seen, last_scan = set(), t_round

        def fail(g, why):
            if res[g]["ok"]:
                res[g]["ok"] = False
                queue[g] = []
            res[g]["why"].append(why)
            on_fail(g, "; ".join(res[g]["why"]))   # the card to 0 first (a hung kernel may then finish) ...
            ck = running.pop(g, None)               # ... then its check is stopped (if on_fail raised: by finally)
            if ck and ck["proc"].poll() is None and not ck["proc"].stop():
                raise GlobalStop(f"card {g}: its {ck['kind']} check does not stop")

        try:
            while True:
                for g in cards:
                    if g in running or not queue[g] or not res[g]["ok"]:
                        continue
                    kind = queue[g][0]
                    if kind == "mmvq" and sum(1 for c in running.values() if c["kind"] == "mmvq") >= self.a.mmvq_jobs:
                        continue
                    queue[g].pop(0)
                    running[g] = self.start_check(g, kind, tag, ref)
                if not running:
                    break
                self.x.sleep(TICK)
                self.watch_global()
                for g, ck in list(running.items()):
                    limit = limits.get((g, ck["kind"]), STOCK_LIMIT) if limits is not None else STOCK_LIMIT
                    rc = ck["proc"].poll()
                    timed_out = rc is None and self.x.now() - ck["t0"] > limit
                    if rc is None and not timed_out:
                        continue
                    # a check past its limit is read as it stands; fail() sets the card to 0, then stops the check
                    r = self.finish_check(ck, rc, timed_out, limit)
                    res[g][ck["kind"]] = r
                    if r["ok"]:
                        running.pop(g, None)
                    else:
                        fail(g, r["why"])
                if self.x.now() - last_scan >= XID_EVERY or not running:
                    last_scan = self.x.now()
                    self.scan_xids(t_round, seen, cards, res, fail)
            self.scan_xids(t_round, seen, cards, res, fail)
        finally:
            for g, ck in list(running.items()):
                if ck["proc"].poll() is None:
                    ck["proc"].stop()
        return res

    def scan_xids(self, since, seen, cards, res, fail):
        for line in self.xid_lines(since - 2):
            if line in seen:
                continue
            seen.add(line)
            g = xid_card(line, self.bus_of)
            if g is None or g not in cards:
                if g is not None and self.card(g, must=False) and self.card(g)["status"] != "active":
                    self.log(f"xid: on card {g}, which already left the climb: {line}")
                    continue
                raise GlobalStop(f"an Xid that names no card in this round: {line}")
            res[g]["xid"] += 1
            self.log(f"xid: card {g}: {line}")
            fail(g, f"Xid {xid_code(line)}")
        for line in self.x.read(self.card_marker()).splitlines():
            if line in seen:
                continue
            seen.add(line)
            m = re.search(r"CARD (\d+) TRIP", line)
            if m and int(m.group(1)) in cards and res[int(m.group(1))]["ok"]:
                fail(int(m.group(1)), "the watchdog tripped on this card")


# ---------------------------------------------------------------- the climb

class Climb:
    def __init__(self, kit):
        self.k, self.a, self.x = kit, kit.a, kit.x

    def preflight(self, need_watchdog=True):
        k = self.k
        if not self.x.dry and os.geteuid() != 0:
            raise Refused("needs root (or --dry-run)")
        k.load_units()
        smi = k.smi()
        missing = [g for g in self.a.cards if g not in smi]
        if missing:
            raise Refused(f"cards {missing} are not on this rig")
        bad = start_check_reasons(k, smi)
        if bad:
            raise Refused("start check: " + "; ".join(bad))
        if need_watchdog and not k.watchdog_running():
            raise Refused(f"the watchdog is not running (climb.py watchdog start); marker {k.marker()}")
        for path, what in ((self.a.mmvq, "mmvq_pascal"), (self.a.greedy, "greedy_check.py")):
            if self.a.mmvq_jobs == 0 and what == "mmvq_pascal":
                continue
            if not self.x.dry and not os.path.exists(path):
                raise Refused(f"{what} not found at {path}")
        rc, o = self.x.run(["stat", "-L", "-c", "%s", self.a.check_model], op=("stat",))
        if rc != 0:
            raise Refused(f"check model {self.a.check_model}: {o.strip()}")
        return smi, int(o.split()[0])

    def new_state(self, smi, size, kind):
        cfg = {k: getattr(self.a, k) for k in ("mem_steps", "core_bin_mhz", "core_step_bins", "core_peak_max",
                                                "core_first_peak", "limit_factor", "limit_floor", "mmvq_args",
                                                "mmvq_jobs", "greedy_n", "order", "server_args", "mem_cap",
                                                "core_cap", "stock_peak_secs")}
        st = {"kind": kind, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.x.now())),
              "boot": self.k.boot(), "driver": smi[self.a.cards[0]]["driver"],
              "check_model": os.path.basename(self.a.check_model), "check_model_bytes": size, "config": cfg,
              "cards": []}
        for g in self.a.cards:
            st["cards"].append({"card": g, "pci": smi[g]["pci"], "name": smi[g]["name"], "vbios": smi[g]["vbios"],
                                "status": "active", "mem": 0, "core": 0, "bins": 0, "stock_peak": None,
                                "stock_hash": None, "stock_s": {}, "limits": {}, "mem_passed": [],
                                "core_passed": [], "core_took_part": False, "failure": None,
                                "stepped_down_mid_climb": False, "history": []})
        self.k.st = st
        return st

    def save(self):
        with open(self.k.p("climb.json" if self.k.st["kind"] == "climb" else "apply.json"), "w") as f:
            json.dump(self.k.st, f, indent=1)

    def tested(self, cards):
        return {g: (self.k.card(g)["mem"], self.k.card(g)["core"], self.k.card(g)["bins"]) for g in cards}

    def steps_row(self, phase, step, g, res, peak, tested=None):
        c = self.k.card(g)
        mem, core, bins = tested or (c["mem"], c["core"], c["bins"])
        path = self.k.p("steps.csv" if self.k.st["kind"] == "climb" else "apply_steps.csv")
        new = not os.path.exists(path)
        gr, mv = res.get("greedy", {}), res.get("mmvq", {})
        lim = c["limits"]
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "phase", "step", "card", "mem_mts", "core_mhz", "core_steps", "expected_peak_mhz",
                            "measured_peak_mhz", "greedy", "greedy_s", "greedy_limit_s", "mmvq_pass", "mmvq_fail",
                            "stream_gbs", "mmvq_s", "mmvq_limit_s", "xid", "verdict", "reason"])
            exp = round(c["stock_peak"] + bins * self.a.core_bin_mhz) if c["stock_peak"] else ""
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.x.now())), phase, step, g, mem,
                        core, bins, exp, peak if peak else "",
                        ("same" if gr.get("ok") else "FAIL") if gr else "", f"{gr['s']:.1f}" if gr else "",
                        f"{lim.get('greedy', 0):.0f}" if lim else "", mv.get("passes", ""), mv.get("fails", ""),
                        mv.get("gbs") or "", f"{mv['s']:.1f}" if mv else "", f"{lim.get('mmvq', 0):.0f}" if lim else "",
                        res.get("xid", 0), "PASS" if res["ok"] else "FAIL", "; ".join(res["why"])])

    def fail_card(self, g, phase, why):
        c = self.k.card(g)
        if c["status"] != "active":
            return
        before = {"mem": c["mem"], "core": c["core"], "bins": c["bins"]}
        peak = round(c["stock_peak"] + c["bins"] * self.a.core_bin_mhz) if c["stock_peak"] else None
        c["status"] = "failed"
        c["failure"] = {"phase": phase, "mem": before["mem"], "core": before["core"], "core_steps": before["bins"],
                        "peak": peak if phase == "core" else None, "reason": why,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.x.now()))}
        self.k.log(f"FAIL card {g} at mem +{before['mem']} MT/s, core +{before['core']} MHz: {why}")
        self.k.reset_card(g, f"{phase} failure")
        c["stepped_down_mid_climb"] = any(before[k] for k in ("mem", "core"))
        self.k.stop_server(g)

    def stock(self, cards):
        """Two stock rounds, and more while they have run less than --stock-peak-secs: the reference hash (every later
        round must give the same: the check is deterministic), each card's check times in round 2, from which the
        limits come, and its stock peak clock: the highest reading from the start of round 1 to the end of the last
        round. Round 2 alone read a driver step low (R200: card 8 at 1,898 MHz on all five boots, where both rounds
        read 1,911 and its soak peak at +231 MHz implies 1,911)."""
        def refuse(g, why):
            raise Refused(f"card {g} fails its check at stock: {why}")
        t0 = self.x.now()
        self.k.log(f"stock: round 1 (each card's reference) on cards {cards}")
        r1 = self.k.run_round(cards, "stock", None, None, refuse)
        self.k.log("stock: round 2 (the same result again, and the check times)")
        r2 = self.k.run_round(cards, "stock2", "stock", None, refuse)
        n = 2
        while self.x.now() - t0 < self.a.stock_peak_secs and n < STOCK_ROUNDS_MAX:
            n += 1
            self.k.log(f"stock: round {n} (the same result again; the stock peak window has run "
                       f"{self.x.now() - t0:.0f} s of --stock-peak-secs {self.a.stock_peak_secs:.0f})")
            self.k.run_round(cards, f"stock{n}", "stock", None, refuse)
        t1 = self.x.now()
        self.k.st["stock_peak_window_s"] = round(t1 - t0)
        peaks = self.x.peaks(os.path.join(self.a.work, "wd.csv"), cards, t0, t1)
        self.k.log(f"stock: each card's stock peak is its highest reading in the {t1 - t0:.0f} s of the {n} stock "
                   "rounds")
        hashes = set()
        for i, g in enumerate(cards):
            c = self.k.card(g)
            c["stock_hash"] = r1[g]["greedy"]["hash"]
            hashes.add(c["stock_hash"])
            for kind in self.a.order:
                if kind in r2[g]:
                    c["stock_s"][kind] = round(r2[g][kind]["s"], 1)
                    c["limits"][kind] = max(self.a.limit_factor * r2[g][kind]["s"], self.a.limit_floor)
            if self.a.stock_peaks:
                c["stock_peak"] = self.a.stock_peaks[self.a.cards.index(g)]
                if peaks[g][0] and peaks[g][0] != c["stock_peak"]:
                    self.k.log(f"stock: card {g} peaked at {peaks[g][0]} MHz, the given stock peak is {c['stock_peak']}")
            else:
                c["stock_peak"] = peaks[g][0]
            c["mem_passed"] = [0]
            mv = r2[g].get("mmvq", {})
            self.k.log(f"stock: card {g} hash {c['stock_hash']}, greedy {c['stock_s'].get('greedy')} s, mmvq "
                       f"{c['stock_s'].get('mmvq')} s ({mv.get('gbs')} GB/s), limits "
                       f"{', '.join(f'{k} {v:.0f} s' for k, v in c['limits'].items())}, stock peak {c['stock_peak']} MHz")
            self.steps_row("stock", 0, g, r2[g], peaks[g][0])
        if len(hashes) > 1:
            self.k.log(f"stock: the cards' hashes differ ({sorted(hashes)}): each card is checked against its own")
        missing = [g for g in cards if not self.k.card(g)["stock_peak"]]
        if missing and self.k.st["kind"] == "climb":
            raise Refused(f"no stock peak clock for cards {missing} (no P0 samples in wd.csv): pass --stock-peaks")
        self.save()

    def memory_phase(self):
        for v in self.a.mem_steps:
            act = [g for g in self.a.cards if self.k.card(g)["status"] == "active"]
            if not act:
                self.k.log("memory: every card has failed")
                return
            self.k.log(f"memory: step +{v} MT/s on cards {act} (core 0)")
            self.check_host()
            for g in act:
                why = self.k.set_offsets(g, mem=v)
                if why:
                    self.fail_card(g, "memory", why)
            act = [g for g in act if self.k.card(g)["status"] == "active"]
            tested = self.tested(act)
            t0 = self.x.now()
            res = self.k.run_round(act, f"mem{v}", "stock", self.limits(act), lambda g, why: self.fail_card(g, "memory", why))
            peaks = self.x.peaks(os.path.join(self.a.work, "wd.csv"), act, t0, self.x.now())
            for g in act:
                c = self.k.card(g)
                if res[g]["ok"] and c["status"] == "active":
                    c["mem_passed"].append(v)
                self.steps_row("memory", f"+{v}", g, res[g], peaks[g][0], tested[g])
                c["history"].append(["memory", v, 0, peaks[g][0], "PASS" if res[g]["ok"] else "FAIL"])
            self.k.log(f"memory: +{v} done, PASS {[g for g in act if res[g]['ok']]}, "
                       f"FAIL {[g for g in act if not res[g]['ok']]}")
            self.save()

    def core_targets(self, stocks):
        """[(target bin, target peak MHz)] of the core climb, from the lowest stock peak up."""
        b = self.a.core_bin_mhz
        s_min = min(stock_bin(p, b) for p in stocks)
        p_min = min(stocks)
        t = s_min + self.a.core_step_bins
        if self.a.core_first_peak:
            t = max(t, s_min + math.ceil((self.a.core_first_peak - p_min) / b - 1e-9))
        out = []
        while p_min + (t - s_min) * b <= self.a.core_peak_max + 0.5:
            out.append((t, round(p_min + (t - s_min) * b)))
            t += self.a.core_step_bins
        return out

    def core_phase(self, part=None, memory_note="memory at each card's last clean step"):
        """The core climb for the cards in `part` (default: every card that never failed), each card's memory held
        where it is."""
        if part is None:
            part = [g for g in self.a.cards if self.k.card(g)["status"] == "active"
                    and not self.k.card(g)["stepped_down_mid_climb"]]
            if not part:
                self.k.log("core: no card left (every card failed in the memory climb)")
                return
        if not part:
            self.k.log("core: no card to climb")
            return
        b = self.a.core_bin_mhz
        for g in part:
            c = self.k.card(g)
            c["core_took_part"] = True
            c["core_passed"] = [[0, 0, c["stock_peak"]]]
        sb = {g: stock_bin(self.k.card(g)["stock_peak"], b) for g in part}
        for t, peak in self.core_targets([self.k.card(g)["stock_peak"] for g in part]):
            act = [g for g in part if self.k.card(g)["status"] == "active"]
            if not act:
                self.k.log("core: every card has failed")
                return
            self.check_host()
            plan = []
            for g in act:
                c = self.k.card(g)
                n = max(0, t - sb[g])
                if n > c["bins"]:
                    off = core_offset(n, b)
                    if off > self.a.core_cap:
                        self.k.log(f"core: card {g} would need +{off} MHz, above --core-cap {self.a.core_cap}: it stays")
                        continue
                    why = self.k.set_offsets(g, core=off)
                    c["bins"] = n
                    if why:
                        self.fail_card(g, "core", why)
                        continue
                plan.append(f"{g}:+{c['core']}")
            self.k.log(f"core: step to a {peak} MHz peak (cards {' '.join(plan)} MHz; {memory_note})")
            act = [g for g in act if self.k.card(g)["status"] == "active"]
            tested = self.tested(act)
            t0 = self.x.now()
            res = self.k.run_round(act, f"core{peak}", "stock", self.limits(act), lambda g, why: self.fail_card(g, "core", why))
            peaks = self.x.peaks(os.path.join(self.a.work, "wd.csv"), act, t0, self.x.now())
            for g in act:
                c = self.k.card(g)
                if res[g]["ok"] and c["status"] == "active" and c["core"] > c["core_passed"][-1][0]:
                    c["core_passed"].append([c["core"], c["bins"], peaks[g][0]])
                self.steps_row("core", peak, g, res[g], peaks[g][0], tested[g])
                c["history"].append(["core", tested[g][0], tested[g][1], peaks[g][0], "PASS" if res[g]["ok"] else "FAIL"])
            self.k.log(f"core: {peak} MHz done, PASS {[g for g in act if res[g]['ok']]}, "
                       f"FAIL {[g for g in act if not res[g]['ok']]}")
            self.save()

    def limits(self, cards):
        return {(g, kind): v for g in cards for kind, v in self.k.card(g)["limits"].items()}

    def check_host(self):
        self.k.watch_global()
        if not self.k.watchdog_running():
            raise GlobalStop("the watchdog is no longer running")
        mb = self.k.avail_mb()
        if mb < self.a.min_avail_mb:
            raise GlobalStop(f"host memory available {mb} MiB, below --min-avail-mb {self.a.min_avail_mb}")

    def fit_mmvq_jobs(self):
        if self.a.mmvq_jobs == 0:
            return
        mb = self.k.avail_mb()
        room = (mb - self.a.min_avail_mb) // 170
        if room < 1:
            raise Refused(f"host memory available {mb} MiB with the servers loaded: no room for mmvq_pascal "
                          f"(~170 MiB each) above --min-avail-mb {self.a.min_avail_mb}; use --mmvq-jobs 0 or a "
                          "smaller check model")
        if room < self.a.mmvq_jobs:
            self.k.log(f"host: {mb} MiB available with the servers loaded: mmvq jobs {self.a.mmvq_jobs} -> {room}")
            self.a.mmvq_jobs = int(room)
        else:
            self.k.log(f"host: {mb} MiB available with the servers loaded")

    def run(self):
        smi, size = self.preflight()
        st = self.new_state(smi, size, "climb")
        self.save()
        self.k.log(f"climb: cards {self.a.cards}, memory steps {self.a.mem_steps}, core +{self.a.core_step_bins} "
                   f"steps of {self.a.core_bin_mhz} MHz to a {self.a.core_peak_max} MHz peak, limits "
                   f"{self.a.limit_factor}x stock (at least {self.a.limit_floor} s), check model "
                   f"{st['check_model']}, boot {st['boot']}")
        return self.sitting(st, self.climb_phases)

    def climb_phases(self):
        self.memory_phase()
        self.core_phase()

    def sitting(self, st, phases):
        """The servers, the stock rounds, then phases(); whatever happens, every offset of every card to 0 at the end,
        the servers stopped, the state and the profile written."""
        ok = False
        try:
            try:
                self.k.start_servers(self.a.cards)
                self.fit_mmvq_jobs()
                self.stock(self.a.cards)
            except Refused as e:
                st["stop_reason"] = f"refused before any offset: {e}"
                self.k.log(f"REFUSED: {e}")
                return 1
            phases()
            st["complete"] = True
            ok = True
        except GlobalStop as e:
            st["stop_reason"] = str(e)
            self.k.log(f"STOP: {e}")
        finally:
            for c in st["cards"]:
                if c["status"] == "active" and not ok and (c["mem"] or c["core"]):
                    c["status"] = "stopped"
            done = self.k.reset_all("end of the climb" if ok else f"stop: {st.get('stop_reason')}")
            self.k.stop_servers()
            st["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.x.now()))
            if not done:
                st["stop_reason"] = (st.get("stop_reason") or "") + "; a reset did not read back 0"
            self.save()
            write_profile(self.k)
            self.summary()
        return 0 if ok else 1

    def summary(self):
        fp = self.k.st.get("from_profile")
        self.k.log("summary (profile.json: " + (f"memory at {fp}'s value, core one step under the last clean step for "
                                                "the cards that climbed it, kept for the others):" if fp else
                                                "one step under each card's last clean step):"))
        def plus(v):
            return "none" if v is None else f"+{v}"
        for c in build_profile(self.k.st)["cards"]:
            f = c["failure"]
            self.k.log(f"  card {c['card']}: profile mem +{c['memory_mts']} MT/s, core +{c['core_mhz']} MHz "
                       f"(peak ~{c['expected_peak_mhz']} MHz) | last clean mem {plus(c['memory']['last_clean_mts'])}, "
                       f"core {plus(c['core']['last_clean_mhz'])}" + (" (climbed)" if fp and c["core"]["took_part"] else "")
                       + " | "
                       + (f"failed in the {f['phase']} {'check' if f['phase'] == 'profile' else 'climb'} at mem "
                          f"+{f['mem']}, core +{f['core']}: {f['reason']}" if f else "no failure")
                       + (" | NEEDS A REBOOT BEFORE USE" if c["needs_reboot_before_use"] else ""))
        if self.k.any_set:
            self.k.log("every offset of every card is 0 now: reboot the rig before `climb.py apply` (offsets only go "
                       "up after a fresh boot)")
        if not fp:
            self.k.log(f"the soak ladder's rungs (file work, any machine): climb.py rungs {self.k.p('profile.json')} "
                       "--failures EARLIER.json (the rig's failures before this climb, if any)")
        else:
            self.k.log("no offset was set in this run")


def write_profile(kit):
    if not kit.st or kit.st["kind"] != "climb":
        return
    with open(kit.p("profile.json"), "w") as f:
        json.dump(build_profile(kit.st), f, indent=1)
    kit.log(f"profile: {kit.p('profile.json')}")


def read_profile(path):
    try:
        with open(path) as f:
            prof = json.load(f)
    except (OSError, ValueError) as e:
        raise Refused(f"profile {path}: {e}")
    if not isinstance(prof, dict):
        raise Refused(f"profile {path}: not a profile")
    return prof


def profile_cards(a, prof):
    """The cards of a --from-profile run: --cards (CARDS) if given, else every card in the profile."""
    have = sorted(c["card"] for c in prof.get("cards") or [] if isinstance(c, dict) and isinstance(c.get("card"), int))
    return list(a.cards) if a.cards_given else have


# ---------------------------------------------------------------- the core climb from a profile (--from-profile)

class FromProfile(Climb):
    """No memory climb. After the stock rounds every card's memory goes straight to the profile's value (with its
    core, for a card that keeps the profile's core value), every card runs one check round there, and the cards that
    keep the profile's core are then done (set to 0, read back, their servers stopped). The others climb the core
    from 0 with their memory held at the profile's value."""

    def run(self, path):
        prof = read_profile(path)
        given = list(self.a.cards)
        self.a.cards = profile_cards(self.a, prof)
        if self.a.stock_peaks:   # one per card of --cards: kept with their cards
            if any(g not in given for g in self.a.cards):
                raise Refused(f"--stock-peaks gives cards {given}, the run has cards {self.a.cards}")
            self.a.stock_peaks = [self.a.stock_peaks[given.index(g)] for g in self.a.cards]
        plan, bad = profile_plan(prof, self.a.cards, self.a.core_cards, self.a.mem_cap, self.a.core_cap,
                                 self.a.core_bin_mhz)
        if bad:
            raise Refused(f"profile {path}: " + "; ".join(bad))
        smi, size = self.preflight()
        by_card = {c["card"]: c for c in prof["cards"] if isinstance(c, dict)}
        for g in self.a.cards:
            if pci_key(smi[g]["pci"]) != pci_key(by_card[g]["pci"]) or smi[g]["name"] != by_card[g]["name"]:
                raise Refused(f"card {g} is {smi[g]['name']} at {smi[g]['pci']}, the profile has "
                              f"{by_card[g]['name']} at {by_card[g]['pci']}")
        if prof.get("climb_boot") and self.k.boot() == prof["climb_boot"]:
            raise Refused("the rig has not rebooted since the climb that wrote the profile: reboot first")
        st = self.new_state(smi, size, "climb")
        st["from_profile"] = os.path.basename(path)
        for g in self.a.cards:
            c = self.k.card(g)
            c["from_profile"] = plan[g]
            c["profile_check"] = None
            c["profile_peak"] = None
        self.save()
        climbers = [g for g in self.a.cards if plan[g]["core_climb"]]
        mems = " ".join("%d:+%d" % (g, plan[g]["memory_mts"]) for g in self.a.cards)
        self.k.log(f"climb from profile {path}: cards {self.a.cards}; memory straight to the profile's values "
                   f"(card:MT/s {mems}), one check round, "
                   f"then the core climb for cards {climbers} (+{self.a.core_step_bins} steps of "
                   f"{self.a.core_bin_mhz} MHz to a {self.a.core_peak_max} MHz peak); limits {self.a.limit_factor}x "
                   f"stock (at least {self.a.limit_floor} s), check model {st['check_model']}, boot {st['boot']}")
        for g in self.a.cards:
            p = plan[g]
            if p["core_climb"] and p["profile_core_mhz"]:
                self.k.log(f"climb from profile: card {g}'s core climbs from 0 (the profile's +{p['profile_core_mhz']} "
                           "MHz is not used)")
        return self.sitting(st, self.from_profile_phases)

    def from_profile_phases(self):
        self.profile_round()
        part = [g for g in self.a.cards if self.k.card(g)["status"] == "active"
                and self.k.card(g)["from_profile"]["core_climb"]]
        self.core_phase(part, "memory at the profile's value")

    def profile_round(self):
        """Every card to its profile values, read back, then one check round for every card at once."""
        cards = [g for g in self.a.cards if self.k.card(g)["status"] == "active"]
        self.check_host()
        for g in cards:
            c = self.k.card(g)
            src = c["from_profile"]
            core = None if src["core_climb"] else (src["core_mhz"] or None)
            why = self.k.set_offsets(g, mem=src["memory_mts"] or None, core=core)
            if core:
                c["bins"] = src["core_steps"]
            if why:
                c["profile_check"] = "FAIL"
                self.fail_card(g, "profile", why)
        act = [g for g in cards if self.k.card(g)["status"] == "active"]
        self.k.log("profile: check round at (card:memory MT/s/core MHz) "
                   + " ".join(f"{g}:+{self.k.card(g)['mem']}/+{self.k.card(g)['core']}" for g in act))
        tested = self.tested(act)
        t0 = self.x.now()
        res = self.k.run_round(act, "profile", "stock", self.limits(act),
                               lambda g, why: self.fail_card(g, "profile", why))
        peaks = self.x.peaks(os.path.join(self.a.work, "wd.csv"), act, t0, self.x.now())
        for g in act:
            c = self.k.card(g)
            ok = res[g]["ok"] and c["status"] == "active"
            c["profile_check"] = "PASS" if ok else "FAIL"
            c["profile_peak"] = peaks[g][0]
            if ok:
                c["mem_passed"].append(c["from_profile"]["memory_mts"])
            self.steps_row("profile", "check", g, res[g], peaks[g][0], tested[g])
            c["history"].append(["profile", tested[g][0], tested[g][1], peaks[g][0], "PASS" if ok else "FAIL"])
        passed = [g for g in act if self.k.card(g)["profile_check"] == "PASS"]
        self.k.log(f"profile: check round done, PASS {passed}, FAIL {[g for g in cards if g not in passed]}")
        done = [g for g in passed if not self.k.card(g)["from_profile"]["core_climb"]]
        for g in done:   # nothing left to find for these: to 0 now rather than idle at their offsets until the end
            self.k.card(g)["status"] = "done"
            self.k.reset_card(g, "checked at the profile's values, no core climb")
            self.k.stop_server(g)
        if done:
            self.k.log(f"profile: cards {done} keep the profile's values (checked); set to 0 now, servers stopped")
        self.save()


# ---------------------------------------------------------------- apply

def apply_cards(a, prof):
    """apply's cards: --cards (CARDS) among the profile's if given, else every card in the profile."""
    have = sorted(c["card"] for c in prof.get("cards") or [] if isinstance(c, dict) and isinstance(c.get("card"), int))
    return [g for g in a.cards if g in have] if a.cards_given else have


def check_apply(a):
    """apply: the profile read and checked (read-only). Returns (profile, cards, plan) or raises Refused."""
    prof = read_profile(a.profile)
    cards = apply_cards(a, prof)
    plan, bad = apply_plan(prof, cards, a.mem_cap, a.core_cap, a.core_bin_mhz)
    if bad:
        raise Refused(f"profile {a.profile}: " + "; ".join(bad))
    return prof, cards, plan


class Apply(Climb):
    def run(self, path):
        self.a.profile = path
        try:
            prof, self.a.cards, plan = check_apply(self.a)
        except Refused as e:
            self.k.log(f"REFUSED: {e}")
            return 1
        by_card = {c["card"]: c for c in prof["cards"] if isinstance(c, dict)}
        try:
            smi, size = self.preflight()
            for g in self.a.cards:
                if pci_key(smi[g]["pci"]) != pci_key(by_card[g]["pci"]) or smi[g]["name"] != by_card[g]["name"]:
                    raise Refused(f"card {g} is {smi[g]['name']} at {smi[g]['pci']}, the profile has "
                                  f"{by_card[g]['name']} at {by_card[g]['pci']}")
            if self.k.boot() == prof.get("climb_boot"):
                raise Refused("the rig has not rebooted since the climb (its last step set every card to 0): reboot "
                              "first")
        except Refused as e:
            self.k.log(f"REFUSED: {e}")
            return 1
        st = self.new_state(smi, size, "apply")
        st["profile"] = os.path.basename(path)
        if not prof.get("complete"):
            self.k.log(f"apply: the profile is from a climb that stopped early ({prof.get('stop_reason')}); its "
                       "values are each card's last clean step less one, as far as the climb got")
        self.save()
        self.k.log(f"apply: {path} on cards {self.a.cards}, {self.a.rounds} check round(s), boot {st['boot']}")
        ok = False
        try:
            try:
                self.k.start_servers(self.a.cards)
                self.fit_mmvq_jobs()
                self.stock(self.a.cards)
            except Refused as e:
                self.k.log(f"REFUSED: {e}")
                st["stop_reason"] = str(e)
                return 1
            for g in self.a.cards:
                want = plan[g]   # the profile's offsets as they are: nothing here reads this boot's stock peak
                why = self.k.set_offsets(g, mem=want["memory_mts"] or None, core=want["core_mhz"] or None)
                c = self.k.card(g)
                c["bins"] = want["core_steps"]
                self.k.log(f"apply: card {g} set mem +{c['mem']} MT/s, core +{c['core']} MHz, read back "
                           f"{'as set' if not why else why}" + (
                               f" (the profile's offset, {c['bins']} steps; stock peak {c['stock_peak']} MHz on this "
                               f"boot, {by_card[g].get('stock_peak_mhz')} MHz in the profile)" if c["core"] else ""))
                if c["core"] > want["core_mhz"] or c["mem"] > want["memory_mts"]:
                    raise GlobalStop(f"card {g}: mem +{c['mem']}, core +{c['core']} is above the profile's "
                                     f"+{want['memory_mts']}, +{want['core_mhz']} (a bug in apply)")
                if why:
                    raise GlobalStop(f"card {g}: {why}")
            for r in range(self.a.rounds):
                self.check_host()

                def bad(g, why):
                    raise GlobalStop(f"card {g} failed its check at the profile's offsets: {why}")
                t0 = self.x.now()
                res = self.k.run_round(self.a.cards, f"apply{r + 1}", "stock", self.limits(self.a.cards), bad)
                peaks = self.x.peaks(os.path.join(self.a.work, "wd.csv"), self.a.cards, t0, self.x.now())
                for g in self.a.cards:
                    self.steps_row("apply", r + 1, g, res[g], peaks[g][0])
                self.k.log(f"apply: round {r + 1}: every card PASS")
            ok = True
        except GlobalStop as e:
            st["stop_reason"] = str(e)
            self.k.log(f"FAIL: {e}")
        finally:
            self.k.stop_servers()
            if not ok:
                self.k.reset_all(f"apply failed: {st.get('stop_reason')}")
                self.k.log("every offset of every card is 0 now" + ("; the rig needs a reboot before the next try"
                                                                   if self.k.any_set else " (none was set)"))
            st["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.x.now()))
            st["complete"] = ok
            self.save()
        self.k.log("apply: PASS. The profile's offsets are set and checked; keep the watchdog running (it holds the "
                   "cards, so the offsets stay) and start the model server" if ok else "apply: FAILED")
        return 0 if ok else 1


# ---------------------------------------------------------------- read-only commands, trip test, watchdog

def start_check_reasons(k, smi):
    bad = []
    procs = k.load_procs()
    if procs:
        bad.append(f"load processes run: {', '.join(procs)}")
    for g in k.a.cards:
        c = smi[g]
        if c["used"] not in ("0", "0 MiB"):
            bad.append(f"card {g} uses {c['used']} MiB")
        if c["limit"] != c["default_limit"]:
            bad.append(f"card {g} power limit {c['limit']} W, default {c['default_limit']} W")
        rb = k.readback(g)
        if rb != (0, 0):
            bad.append(f"card {g} offsets (mem core) {rb[0]} {rb[1]}")
    down = k.ledger_this_boot()
    if down:
        bad.append(f"an offset stepped down this boot ({len(down)} ledger lines, the last: {down[-1]}): reboot first")
    return bad


def cmd_state(k):
    smi = k.smi()
    print(f"boot {k.boot()}, driver {smi[min(smi)]['driver']}")
    print("card,pci,memory_used_mib,power_limit_w,default_limit_w,pstate,mem_offset,core_offset")
    for g in k.a.cards:
        c = smi.get(g)
        if not c:
            print(f"{g},missing")
            continue
        rb = k.readback(g)
        print(f"{g},{c['pci']},{c['used']},{c['limit']},{c['default_limit']},{c['pstate']},{rb[0]},{rb[1]}")
    print(f"load processes: {', '.join(k.load_procs()) or 'none'}")
    print(f"watchdog running: {'yes' if k.watchdog_running() else 'no'}")
    print(f"Xid lines this boot: {len(k.xids_this_boot())}")
    down = k.ledger_this_boot()
    print(f"step downs this boot (ledger {k.a.ledger}): {len(down)}" + (f", the last: {down[-1]}" if down else ""))
    print(f"host memory available: {k.avail_mb()} MiB")
    return 0


def cmd_start_check(k):
    k.load_units()
    bad = start_check_reasons(k, k.smi())
    print("start check: PASS" if not bad else "start check: FAIL\n  " + "\n  ".join(bad))
    return 0 if not bad else 1


def check_from_profile(a):
    """--from-profile: the profile read and checked (read-only). Returns (profile, cards, plan) or raises Refused."""
    prof = read_profile(a.from_profile)
    cards = profile_cards(a, prof)
    plan, bad = profile_plan(prof, cards, a.core_cards, a.mem_cap, a.core_cap, a.core_bin_mhz)
    if bad:
        raise Refused(f"profile {a.from_profile}: " + "; ".join(bad))
    return prof, cards, plan


def cmd_plan_from_profile(k):
    a = k.a
    try:
        prof, cards, plan = check_from_profile(a)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1
    climbers = [g for g in cards if plan[g]["core_climb"]]
    print(f"from profile {a.from_profile}: no memory climb; every card straight to these values, one check round, "
          f"then the core climb for cards {climbers}, memory held")
    print("card,memory_mts,core_mhz,core_climb,profile_core_mhz")
    for g in cards:
        p = plan[g]
        print(f"{g},{p['memory_mts']},{p['core_mhz']},{'yes' if p['core_climb'] else 'no'},{p['profile_core_mhz']}")
    if not climbers:
        print("no card climbs the core")
        return 0
    by = {c["card"]: c for c in prof["cards"] if isinstance(c, dict)}
    if a.stock_peaks:
        peaks, src = [a.stock_peaks[a.cards.index(g)] for g in climbers], "--stock-peaks"
    elif all(by[g].get("stock_peak_mhz") for g in climbers):
        peaks, src = [int(by[g]["stock_peak_mhz"]) for g in climbers], "the profile's, from its own boot"
    else:
        peaks, src = [SCENARIO["stock_peaks"][g % 9] for g in climbers], "rig 1's (R193)"
    print(f"core steps: {a.core_step_bins} steps of {a.core_bin_mhz} MHz of peak each, stock peaks "
          f"{' '.join(str(p) for p in peaks)} MHz ({src}; the climb measures them)")
    b = a.core_bin_mhz
    sb = [stock_bin(p, b) for p in peaks]
    print("target_peak_mhz," + ",".join(f"card{g}_core_mhz" for g in climbers))
    for t, peak in Climb(k).core_targets(peaks):
        print(f"{peak}," + ",".join(str(core_offset(max(0, t - s), b)) for s in sb))
    return 0


def cmd_rungs(a):
    """The ladder's rung profiles from a.profile and the known failures: file work only, nothing set or started."""
    try:
        prof = read_profile(a.profile)
        cards = apply_cards(a, prof)
        if a.cards_given:
            prof["cards"] = [c for c in prof["cards"] if isinstance(c, dict) and c.get("card") in cards]
        _, bad = apply_plan(prof, cards, a.mem_cap, a.core_cap, a.core_bin_mhz)
        if bad:
            raise Refused(f"profile {a.profile}: " + "; ".join(bad))
        source = os.path.basename(a.profile)
        fails = []
        if a.failures:
            try:
                fails = read_failures(a.failures)
            except (OSError, ValueError) as e:
                raise Refused(f"failures {a.failures}: {e}")
        try:
            fails += profile_failures(prof, source)
        except ValueError as e:
            raise Refused(f"profile {a.profile}: its failures list: {e}")
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    rungs, rows = ladder_rungs(prof, fails, a.mem_step_mts, a.core_bin_mhz, a.core_step_bins, a.cap_steps, source)
    out = a.out or os.path.dirname(os.path.abspath(a.profile))
    paths = [os.path.join(out, f"{a.prefix}{k}.json") for k in range(1, len(rungs) + 1)]
    choice = os.path.join(out, f"{a.prefix}s-choice.md")
    there = [p for p in paths + [choice] if os.path.exists(p)]
    if there and not a.overwrite:
        print(f"REFUSED: {', '.join(there)} already there (--overwrite replaces them)", file=sys.stderr)
        return 1
    os.makedirs(out, exist_ok=True)
    for p, r in zip(paths, rungs):
        with open(p, "w") as f:
            json.dump(r, f, indent=1)
    with open(choice, "w") as f:
        f.write(f"Rungs from {source}, by `climb.py rungs`" + (f", failures from {os.path.basename(a.failures)}"
                                                               if a.failures else "") + ".\n\n")
        f.write("\n".join(RUNG_TABLE + rows) + "\n")
    print(f"rungs from {a.profile} and {len(rungs[0]['failures'])} known failures"
          + (f" (with {a.failures})" if a.failures else " (the profile's own)") + "; card:MT/s/MHz")
    for p, r in zip(paths, rungs):
        print(f"  {os.path.basename(p)}: " + " ".join(f"{c['card']}:+{c['memory_mts']}/+{c['core_mhz']}"
                                                     for c in r["cards"]))
    print(f"  {os.path.basename(choice)}: each card's caps and why")
    names = [os.path.basename(p) for p in paths]
    print(f"soak_chain.sh: PROFILE={names[0]} LADDER=\"{' '.join(names[1:])}\"")
    return 0


def cmd_plan(k):
    a = k.a
    if a.from_profile:
        return cmd_plan_from_profile(k)
    peaks = a.stock_peaks or SCENARIO["stock_peaks"][: len(a.cards)]
    if len(peaks) != len(a.cards):
        print("--stock-peaks needs one value per card", file=sys.stderr)
        return 2
    print(f"memory steps (MT/s, all cards, core 0): {' '.join(f'+{v}' for v in a.mem_steps)}")
    print(f"core steps: {a.core_step_bins} steps of {a.core_bin_mhz} MHz of peak each, stock peaks "
          f"{' '.join(str(p) for p in peaks)} MHz" + ("" if a.stock_peaks else " (R193's, rig 1; the climb measures them)"))
    cl = Climb(k)
    b = a.core_bin_mhz
    sb = [stock_bin(p, b) for p in peaks]
    print("target_peak_mhz," + ",".join(f"card{g}_core_mhz" for g in a.cards))
    for t, peak in cl.core_targets(peaks):
        print(f"{peak}," + ",".join(str(core_offset(max(0, t - s), b)) for s in sb))
    return 0


def cmd_trip_test(k):
    """The watchdog's trips with stand-ins before any offset; every check must pass."""
    a, x = k.a, k.x
    t = os.path.join(a.work, "test")
    os.makedirs(os.path.join(t, "pids"), exist_ok=True)
    smi = k.smi()
    if not x.dry:
        for name in ("mmvq_pascal", "llama-server"):
            p = os.path.join(t, name)
            if not os.path.islink(p):
                os.symlink("/bin/sleep", p)
    wd = ["python3", a.watchdog, "--gpus", ",".join(str(g) for g in a.cards), "--power-max", "80"]
    fails = []

    def standins(n=2):
        return [x.spawn([os.path.join(t, "mmvq_pascal" if i % 2 == 0 else "llama-server"), "300"], op=("standin",))
                for i in range(n)]

    def check(name, procs, expect_dead, marker, expect_marker):
        if x.dry:
            print(f"== {name}: (dry-run: nothing ran, so nothing is checked)")
            return
        x.sleep(0.5)
        dead = [p.poll() is not None for p in procs]
        ok = all(d == e for d, e in zip(dead, expect_dead)) and x.exists(marker) == expect_marker
        print(f"== {name}: {'PASS' if ok else 'FAIL'} (stand-ins ended: {dead}, wanted {expect_dead}; marker "
              f"{marker} {'present' if x.exists(marker) else 'absent'}, wanted {'present' if expect_marker else 'absent'})")
        for p in procs:
            if p.poll() is None:
                p.stop(1)
        if not ok:
            fails.append(name)

    print("== A: temperature limit 15 C: every card to 0, every stand-in killed")
    ps = standins()
    x.run(wd + ["--log", f"{t}/wdA.csv", "--marker", f"{t}/TRIPPED_A", "--temp-max", "15", "--duration", "4"],
          op=("watchdog-test", "A"), timeout=60)
    check("A", ps, [True, True], f"{t}/TRIPPED_A", True)
    print("== B: a kernel-log line with the test pattern and no card: every card to 0, every stand-in killed")
    ps = standins()
    x.spawn(["sh", "-c", f"sleep 2; echo '{TEST_LINE}' > /dev/kmsg"], op=("kmsg",))
    x.run(wd + ["--log", f"{t}/wdB.csv", "--marker", f"{t}/TRIPPED_B", "--xid-pattern", "C871WDTEST",
                "--per-card-xid", "--card-pids", f"{t}/pids", "--card-marker", f"{t}/CARD_B", "--duration", "6"],
          op=("watchdog-test", "B"), timeout=60)
    check("B", ps, [True, True], f"{t}/TRIPPED_B", True)
    print("== C: power limit 1 W for 5 s: every card to 0, every stand-in killed")
    ps = standins()
    x.run(wd + ["--log", f"{t}/wdC.csv", "--marker", f"{t}/TRIPPED_C", "--power-max", "1", "--power-secs", "5",
                "--duration", "9"], op=("watchdog-test", "C"), timeout=60)
    check("C", ps, [True, True], f"{t}/TRIPPED_C", True)
    g1, g2 = a.cards[0], a.cards[1] if len(a.cards) > 1 else a.cards[0]
    b = pci_key(smi[g1]["pci"])
    print(f"== D: a kernel-log line naming card {g1}: only card {g1}'s stand-ins killed, no global trip")
    p1, p2 = standins(1), standins(1)
    if not x.dry:
        for g, ps_ in ((g1, p1), (g2, p2)):
            with open(f"{t}/pids/{g}.pids", "w") as f:
                f.write("".join(f"{p.pid}\n" for p in ps_))
    x.spawn(["sh", "-c", f"sleep 2; echo '{TEST_LINE} (PCI:{b[0]:04x}:{b[1]:02x}:{b[2]:02x})' > /dev/kmsg"],
            op=("kmsg",))
    x.run(wd + ["--log", f"{t}/wdD.csv", "--marker", f"{t}/TRIPPED_D", "--xid-pattern", "C871WDTEST",
                "--per-card-xid", "--card-pids", f"{t}/pids", "--card-marker", f"{t}/CARD_D", "--duration", "6"],
          op=("watchdog-test", "D"), timeout=60)
    x.sleep(0.5)
    card_line = x.read(f"{t}/CARD_D")
    ok = (x.dry or (p1[0].poll() is not None and p2[0].poll() is None and f"CARD {g1} TRIP" in card_line
                    and not x.exists(f"{t}/TRIPPED_D")))
    print(f"== D: {'PASS' if ok else 'FAIL'} (card {g1} stand-in ended: {p1[0].poll() is not None}; card {g2} stand-in "
          f"still running: {p2[0].poll() is None}; card marker: {card_line.strip()[:160] or 'none'})")
    for p in p1 + p2:
        if p.poll() is None:
            p.stop(1)
    if not ok:
        fails.append("D")
    print("trip test: " + ("PASS" if not fails else f"FAIL ({', '.join(fails)})"))
    return 0 if not fails else 1


def cmd_watchdog(k, what):
    a, x = k.a, k.x
    stopf = os.path.join(a.work, "WD_STOP")
    if what == "start":
        if not x.dry and k.watchdog_running():
            print("a watchdog with this work folder's marker already runs")
            return 1
        if not x.dry and os.path.exists(stopf):
            os.remove(stopf)
        cmd = ["nice", "-n", "-10", "python3", a.watchdog, "--gpus", "all", "--log", os.path.join(a.work, "wd.csv"),
               "--marker", k.marker(), "--temp-max", str(a.temp_max), "--power-max", str(a.power_max),
               "--power-secs", str(a.power_secs), "--stop", stopf, "--per-card-xid", "--card-pids",
               os.path.join(a.work, "pids"), "--card-marker", k.card_marker(), "--ledger", a.ledger]
        x.spawn(cmd, op=("watchdog",), out=os.path.join(a.work, "wd.out"))
        x.sleep(3)
        head = [ln for ln in x.read(os.path.join(a.work, "wd.csv")).splitlines() if ln.startswith("#")]
        print(head[-1] if head else "watchdog started (no start line yet: check wd.out)")
        return 0
    x.touch(stopf)
    for _ in range(20):
        if not k.watchdog_running():
            break
        x.sleep(0.5)
    if k.watchdog_running() and not x.dry:
        print("watchdog still running")
        return 1
    print("watchdog stopped")
    return 0


def detach(a, name):
    argv = sys.argv[1:]
    i = argv.index(name)
    argv[i] = "run-" + name
    console = os.path.join(a.work, f"{name}.console")
    done = os.path.join(a.work, f"{name.upper()}_DONE")
    if os.path.exists(done):
        os.remove(done)
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv, stdout=open(console, "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"{name} started detached (pid {p.pid}); follow it with: tail -f {os.path.join(a.work, 'out', 'climb.log')}; "
          f"marker {done}; stop it cleanly with: climb.py --work {a.work} stop")
    return 0


# ---------------------------------------------------------------- options

def ints(text):
    return [int(v) for v in re.split(r"[ ,]+", str(text).strip()) if v]


def env(name, default):
    return os.environ.get(name, default)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print every command, set and run nothing (simulated rig)")
    p.add_argument("--scenario", help="dry-run: JSON with the simulated cards' failure points")
    p.add_argument("--work", default=env("WORK", ""), help="work folder [/var/tmp/climb, which survives the reboot before apply; dry-run: ./climb-dryrun]")
    p.add_argument("--cards", default=env("CARDS", "0 1 2 3 4 5 6 7 8"))
    p.add_argument("--bin", default=env("BIN", "/opt/llama-builds/bin"), help="llama.cpp bin folder (llama-server)")
    p.add_argument("--check-model", default=env("CHECK_MODEL", ""), help="a GGUF that fits one card, run on each")
    p.add_argument("--units", default=env("UNITS", os.path.join(HERE, "memoc_units.json")))
    p.add_argument("--memoc", default=env("MEMOC", os.path.join(HERE, "nvml_memoc.py")))
    p.add_argument("--greedy", default=env("GREEDY", os.path.join(HERE, "greedy_check.py")))
    p.add_argument("--mmvq", default=env("MMVQ", os.path.join(HERE, "bin", "mmvq_pascal")))
    wd = os.path.join(HERE, "nvml_watchdog.py")
    p.add_argument("--watchdog", default=env("WATCHDOG", wd if os.path.exists(wd) else
                                             os.path.join(HERE, "..", "..", "scripts", "nvml_watchdog.py")))
    p.add_argument("--ledger", default=env("LEDGER", "/var/tmp/mem_offset_stepdowns.log"),
                   help="every step down, tagged with the boot (survives the work folder, not needed across boots)")
    p.add_argument("--mem-steps", default=env("MEM_STEPS", "200 400 600 800 900 1000 1100 1200 1300 1400"))
    p.add_argument("--core-bin-mhz", type=float, default=float(env("CORE_BIN_MHZ", "12.66")))
    p.add_argument("--core-step-bins", type=int, default=int(env("CORE_STEP_BINS", "2")),
                   help="core steps per climb step (2 x 12.66 = 25 MHz of peak)")
    p.add_argument("--core-peak-max", type=float, default=float(env("CORE_PEAK_MAX", "2250")))
    p.add_argument("--core-first-peak", type=float, default=float(env("CORE_FIRST_PEAK", "0")),
                   help="start the core climb at this target peak instead of one step above the lowest stock peak")
    p.add_argument("--core-cap", type=int, default=int(env("CORE_CAP", "400")))
    p.add_argument("--stock-peaks", default=env("STOCK_PEAKS", ""), help="one per card (MHz); default: measured")
    p.add_argument("--stock-peak-secs", type=float, default=float(env("STOCK_PEAK_SECS", "90")),
                   help="the stock peak is each card's highest reading over the stock rounds, from the first on; "
                        "more rounds run until they cover at least this many seconds")
    p.add_argument("--from-profile", default=env("FROM_PROFILE", ""), metavar="FILE",
                   help="climb: no memory climb; every card's memory straight to FILE's value (a climb's profile.json "
                        "or soak.py's soaked.json), one check round, then the core climb for the cards FILE gives no "
                        "core value; plan: what that would set and climb")
    p.add_argument("--core-cards", default=env("CORE_CARDS", ""),
                   help="--from-profile: the cards that climb the core (from 0), instead of those FILE gives no core "
                        "value")
    p.add_argument("--limit-factor", type=float, default=float(env("LIMIT_FACTOR", "3")))
    p.add_argument("--limit-floor", type=float, default=float(env("LIMIT_FLOOR", "30")))
    p.add_argument("--mmvq-args", default=env("MMVQ_ARGS", "--types q8_0 --shapes attn_q"))
    p.add_argument("--mmvq-jobs", type=int, default=int(env("MMVQ_JOBS", "2")), help="0: no mmvq check")
    p.add_argument("--order", default=env("ORDER", "greedy,mmvq"), help="each card's checks, in this order")
    p.add_argument("--greedy-n", type=int, default=int(env("GREEDY_N", "256")), help="tokens per prompt (2 prompts)")
    p.add_argument("--port-base", type=int, default=int(env("PORT_BASE", "8200")), help="card N's server: base + N")
    p.add_argument("--server-args", default=env("SERVER_ARGS", "-ngl 99 -fa on -c 1024 -np 1 -t 1 --no-warmup"))
    p.add_argument("--server-limit", type=float, default=float(env("SERVER_LIMIT", "900")))
    p.add_argument("--min-avail-mb", type=int, default=int(env("MIN_AVAIL_MB", "300")))
    p.add_argument("--temp-max", type=float, default=float(env("TEMP_MAX", "70")))
    p.add_argument("--power-max", type=float, default=float(env("POWER_MAX", "80")))
    p.add_argument("--power-secs", type=float, default=float(env("POWER_SECS", "5")))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("state", "start-check", "plan", "trip-test", "climb", "run-climb", "profile", "reset-all", "stop"):
        sub.add_parser(name)
    sub.add_parser("watchdog").add_argument("what", choices=("start", "stop"))
    sub.add_parser("reset-card").add_argument("card", type=int)
    for name in ("apply", "run-apply"):
        s = sub.add_parser(name)
        s.add_argument("profile")
        s.add_argument("--rounds", type=int, default=1, help="greedy check rounds at the profile's offsets")
    s = sub.add_parser("rungs", help="the soak ladder's rung profiles from a profile and the known failures")
    s.add_argument("profile")
    s.add_argument("--failures", default=env("FAILURES", ""),
                   help="earlier failures, as soak.py --failures reads them (a use.json soak.py wrote works too)")
    s.add_argument("--out", default="", help="folder for the rungs [the profile's]")
    s.add_argument("--prefix", default="rung", help="file names PREFIX1.json ... and PREFIXs-choice.md [rung]")
    s.add_argument("--overwrite", action="store_true", help="replace rung files already there")
    s.add_argument("--mem-step-mts", type=int, default=int(env("MEM_STEP_MTS", "100")),
                   help="one memory step of the rungs and of the cap (as soak.py's)")
    s.add_argument("--cap-steps", type=int, default=2,
                   help="the top rung stays this many steps under each failure [2, the use rule's; 1 gives R201's "
                        "hand-made rungs]")
    a = p.parse_args(argv)
    a.cards_given = "CARDS" in os.environ or "--cards" in (argv if argv is not None else sys.argv)
    a.cards = ints(a.cards)
    a.mem_steps = ints(a.mem_steps)
    a.mem_cap = max(a.mem_steps)
    a.stock_peaks = ints(a.stock_peaks) if a.stock_peaks else None
    a.core_cards = ints(a.core_cards) if a.core_cards.strip() else None
    a.from_profile = os.path.abspath(a.from_profile) if a.from_profile else ""
    if a.core_cards is not None and not a.from_profile:
        p.error("--core-cards goes with --from-profile")
    a.order = [v for v in a.order.split(",") if v]
    a.work = os.path.abspath(a.work or ("climb-dryrun" if a.dry_run else "/var/tmp/climb"))
    if a.mem_steps != sorted(set(a.mem_steps)) or a.mem_steps[0] <= 0:
        p.error("--mem-steps must rise")
    if a.stock_peaks and len(a.stock_peaks) != len(a.cards):
        p.error("--stock-peaks needs one value per card")
    if set(a.order) - {"greedy", "mmvq"} or "greedy" not in a.order:
        p.error("--order: greedy, and optionally mmvq")
    return a


def main(argv=None, x=None):
    a = parse(argv)
    if a.cmd == "rungs":
        return cmd_rungs(a)
    os.makedirs(a.work, exist_ok=True)
    if x is None:
        if a.dry_run:
            sc = None
            if a.scenario:
                with open(a.scenario) as f:
                    sc = json.load(f)
            x = SimExec(lambda s: print(s, flush=True), sc, os.path.dirname(os.path.abspath(a.scenario))
                        if a.scenario else HERE)
        else:
            x = RealExec(print)
    k = Kit(a, x)
    if a.cmd in ("climb", "run-climb", "apply", "run-apply") and not a.check_model:
        print("--check-model (CHECK_MODEL) is required: a GGUF that fits one card", file=sys.stderr)
        return 2
    if a.cmd == "state":
        return cmd_state(k)
    if a.cmd == "start-check":
        return cmd_start_check(k)
    if a.cmd == "plan":
        return cmd_plan(k)
    if a.cmd == "trip-test":
        return cmd_trip_test(k)
    if a.cmd == "watchdog":
        return cmd_watchdog(k, a.what)
    if a.cmd == "reset-all":
        k.smi()
        return 0 if k.reset_all("reset-all") else 1
    if a.cmd == "reset-card":
        if a.card not in k.smi():
            print(f"card {a.card} is not on this rig", file=sys.stderr)
            return 2
        try:
            k.reset_card(a.card, "reset-card")
        except GlobalStop as e:
            k.log(f"RESET FAILED: {e}. Power the rig off at the console.")
            return 1
        return 0
    if a.cmd == "stop":
        x.touch(os.path.join(a.work, "STOP"))
        print(f"asked to stop ({os.path.join(a.work, 'STOP')}): the climb stops at its next look, every offset to 0")
        return 0
    if a.cmd == "profile":
        with open(k.p("climb.json")) as f:
            k.st = json.load(f)
        write_profile(k)
        return 0
    if a.cmd == "climb" and a.from_profile:
        try:
            check_from_profile(a)   # read-only, so that a bad profile is refused here, not only in the detached log
        except Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
    if a.cmd == "apply":
        try:
            check_apply(a)   # read-only: a bad profile is refused here, before any process, and APPLY_DONE says so
        except Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            if not a.dry_run:
                with open(os.path.join(a.work, "APPLY_DONE"), "w") as f:   # soak_chain.sh waits for it
                    f.write("exit 1\n")
            return 1
    if a.cmd in ("climb", "apply") and not a.dry_run:
        return detach(a, a.cmd)
    if not x.dry and os.path.exists(os.path.join(a.work, "STOP")):
        os.remove(os.path.join(a.work, "STOP"))

    def on_signal(signum, frame):
        raise GlobalStop(f"signal {signum}")
    if not x.dry:
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
    name = a.cmd.replace("run-", "")
    rc = 1
    try:
        if name == "climb":
            try:
                rc = FromProfile(k).run(a.from_profile) if a.from_profile else Climb(k).run()
            except Refused as e:
                k.log(f"REFUSED: {e}")
                rc = 1
        else:
            rc = Apply(k).run(a.profile)
    finally:
        if not x.dry:
            with open(os.path.join(a.work, f"{name.upper()}_DONE"), "w") as f:
                f.write(f"exit {rc}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
