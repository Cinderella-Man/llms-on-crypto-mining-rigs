#!/usr/bin/env python3
"""R223's tables from the chain's work folder and the Ornith folder, copied off the rig.

  python3 -I tables.py WORK_DIR ORNITH_DIR TABLES_DIR

WORK_DIR is soak_chain.sh's work folder (out/apply_steps.csv, soak/soak.csv, soak/rungs.csv, soak/soak.log,
soak/soaked.json, soak/use.json, wd.csv), ORNITH_DIR model_check.sh's OUT (model.log, with the stock rounds and, when
the chain ran it, the rounds with the offsets set). R218's tables.py with Ornith's rounds of both tags and the phase
with the offsets on Ornith in the power table, which has no temperature columns. Writes:
  cards.csv        one row per card: each rung's offsets and differing checks (empty for the rungs after a card
                   left), the rounds it ran, whether and where it left, the value soaked.json holds, the first
                   difference, the use value
  differs.csv     every differing check: time, round, card, rung, offsets, the round within the rung, the rounds the
                   card had run at that value before it (only the header when no card differed)
  rungs.csv        soak.py's rungs.csv as written on the rig (one row per rung and card)
  soak.csv         soak.py's soak.csv (one row per round and card: hash, seconds, verdict)
  apply.csv        climb.py apply's stock round and its check round at rung 1 (peaks, kernel checks, streaming reads)
  power_clocks.csv one row per phase (apply, the soak's server start, each rung, each raise between rungs, the time
                   up to Ornith's load with the offsets, Ornith with the offsets) and card from the watchdog's
                   half-second samples: peak and mean power, samples and the longest run above 80 W, samples in a
                   thermal or hardware slowdown, peak core and memory clocks
  ornith.csv       one row per round of Ornith (at stock, and with the offsets if it ran): hash, whether it equals
                   stock round 1, prompt and generation tok/s of the two prompts
  ornith_load.csv  one row per Ornith load: seconds to ready, MiB per card, host memory available, rounds, exit
"""
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time

HW_SLOW, SW_THERMAL, HW_THERMAL, HW_BRAKE = 0x8, 0x20, 0x40, 0x80
ROUND = re.compile(r"^(\S+ \S+) (stock|profile) round (\d+): greedy \S+: hash (\w+), (\d+) tokens, prompt tok/s "
                   r"([\d.]+) ([\d.]+), generation tok/s ([\d.]+) ([\d.]+)(.*)\(exit (\d+)\)$")
READY = re.compile(r"^(\S+ \S+) (stock|profile): ready in (\d+) s; ([\d ]+) MiB used per card; (\d+) MiB host")
DONE = re.compile(r"^(\S+ \S+) (stock|profile): done, (\d+) rounds in (\d+) s, exit (\d+)$")


def stamp(text):
    return time.mktime(time.strptime(text[:19], "%Y-%m-%d %H:%M:%S"))


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def main():
    work, ornith, tables = sys.argv[1:4]
    os.makedirs(tables, exist_ok=True)
    sd = os.path.join(work, "soak")
    for name in ("rungs.csv", "soak.csv"):
        shutil.copyfile(os.path.join(sd, name), os.path.join(tables, name))
    shutil.copyfile(os.path.join(work, "out", "apply_steps.csv"), os.path.join(tables, "apply.csv"))

    # the soak's rows, for the rounds each card ran and each differing check
    with open(os.path.join(sd, "soak.csv")) as f:
        soak = list(csv.DictReader(f))
    with open(os.path.join(sd, "rungs.csv")) as f:
        rungs = list(csv.DictReader(f))
    first_round = {}                               # rung -> its first round number
    for r in soak:
        rg, rd = int(r["rung"]), int(r["round"])
        first_round[rg] = min(first_round.get(rg, rd), rd)
    ran = {}                                       # card -> rounds checked
    for r in soak:
        ran[int(r["card"])] = ran.get(int(r["card"]), 0) + 1

    with open(os.path.join(tables, "differs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "round", "card", "rung", "mem_mts", "core_mhz", "round_in_rung",
                    "rounds_at_this_value_before"])
        for r in soak:
            if r["verdict"] == "PASS":
                continue
            card, rd, rg = int(r["card"]), int(r["round"]), int(r["rung"])
            before = sum(1 for x in soak if int(x["card"]) == card and int(x["round"]) < rd
                         and x["mem_mts"] == r["mem_mts"] and x["core_mhz"] == r["core_mhz"])
            w.writerow([r["time"], rd, card, rg, r["mem_mts"], r["core_mhz"], rd - first_round[rg] + 1, before])

    # cards.csv
    with open(os.path.join(sd, "soaked.json")) as f:
        soaked = json.load(f)
    with open(os.path.join(sd, "use.json")) as f:
        use = {c["card"]: c for c in json.load(f)["cards"]}
    left = {int(r["card"]): (r["rung"], r["left"]) for r in rungs if r["left"]}
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        n = soaked["rungs"]
        w.writerow(["card", "stock_peak_mhz"]
                   + [x for r in range(1, n + 1) for x in (f"rung{r}_mem_mts", f"rung{r}_core_mhz", f"rung{r}_differs")]
                   + ["rounds_checked", "left_in_rung", "left_why", "soaked_mem_mts", "soaked_core_mhz",
                      "first_differed_rung", "first_differed_round", "use_mem_mts", "use_core_mhz"])
        for c in soaked["cards"]:
            rs = c["soak"]["rungs"]
            fd = c["soak"]["first_differed"] or {}
            u = use[c["card"]]
            lr, lw = left.get(c["card"], ("", ""))
            cells = [x for r in rs for x in (r["memory_mts"], r["core_mhz"], r["differing_checks"])]
            cells += [""] * (3 * n - len(cells))     # a card that left has no rows for the rungs after it
            w.writerow([c["card"], c.get("stock_peak_mhz", "")] + cells
                       + [ran.get(c["card"], 0), lr, lw, c["memory_mts"], c["core_mhz"], fd.get("rung", ""),
                          fd.get("round", ""), u["memory_mts"], u["core_mhz"]])

    # Ornith's rounds and loads
    rounds, ready, done = [], {}, {}
    with open(os.path.join(ornith, "model.log")) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            m = ROUND.match(ln)
            if m:
                rounds.append(m.groups())
                continue
            m = READY.match(ln)
            if m:
                ready[m[2]] = (m[1], m[3], m[4].split(), m[5])
                continue
            m = DONE.match(ln)
            if m:
                done[m[2]] = (m[1], m[3], m[4], m[5])
    with open(os.path.join(tables, "ornith.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "tag", "round", "hash", "tokens", "vs_stock1", "prompt1_prompt_tok_s",
                    "prompt2_prompt_tok_s", "prompt1_gen_tok_s", "prompt2_gen_tok_s", "exit"])
        for t, tag, r, h, n, p1, p2, g1, g2, rest, ex in rounds:
            vs = "reference" if (tag == "stock" and r == "1") else ("same" if "SAME as stock1" in rest else "DIFFERS")
            w.writerow([t, tag, r, h, n, vs, p1, p2, g1, g2, ex])
    with open(os.path.join(tables, "ornith_load.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "ready_at", "load_s"] + [f"card{g}_mib" for g in range(9)]
                   + ["host_mib_available", "done_at", "rounds", "round_seconds", "exit"])
        for tag in ("stock", "profile"):
            if tag in ready:
                t, s, mib, host = ready[tag]
                d = done.get(tag, ("", "", "", ""))
                w.writerow([tag, t, s] + (mib + [""] * 9)[:9] + [host, d[0], d[1], d[2], d[3]])

    # the phases, from soak.log and model.log
    marks = []
    with open(os.path.join(sd, "soak.log")) as f:
        for ln in f:
            m = re.match(r"(\S+ \S+) (soak: cards|rung (\d+): [\d.]+ min on|rung (\d+) done)", ln)
            if m:
                marks.append((stamp(m.group(1)), f"rung {m.group(3)}" if m.group(3) else
                              ("servers" if m.group(2) == "soak: cards" else f"raise after rung {m.group(4)}")))
    with open(os.path.join(work, "wd.csv")) as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    head = {k: i for i, k in enumerate(lines[0].split(","))}
    rows = []
    for ln in lines[1:]:
        v = ln.split(",")
        try:
            rows.append((stamp(v[0]), v))
        except ValueError:
            pass
    phases = [("apply", rows[0][0], marks[0][0])]
    phases += [(marks[i][1], marks[i][0], marks[i + 1][0]) for i in range(len(marks) - 1)]
    if "profile" in ready:                         # the last rung's end to the model's ready, then the model's rounds
        r0 = stamp(ready["profile"][0])
        phases.append(("Ornith's load with the offsets", marks[-1][0], r0))
        phases.append(("Ornith with the offsets", r0, stamp(done["profile"][0]) if "profile" in done else rows[-1][0] + 1))
    cards = sorted(int(k[1:k.index("_")]) for k in head if re.fullmatch(r"g\d+_c", k))
    with open(os.path.join(tables, "power_clocks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "card", "samples", "peak_w", "mean_w", "samples_above_80w",
                    "longest_run_above_80w", "samples_slowdown", "peak_sm_mhz", "peak_mem_mhz"])
        for name, t0, t1 in phases:
            sel = [v for t, v in rows if t0 <= t < t1]
            for g in cards:
                def col(k):
                    return [x for x in (num(v[head[f"g{g}_{k}"]]) for v in sel) if x is not None]
                watts, sms, mems = col("w"), col("sm_mhz"), col("mem_mhz")
                over = run = longest = 0
                for x in watts:
                    run = run + 1 if x > 80 else 0
                    over += x > 80
                    longest = max(longest, run)
                slow = 0
                for v in sel:
                    try:
                        slow += bool(int(v[head[f"g{g}_reasons"]], 0) & (HW_SLOW | SW_THERMAL | HW_THERMAL | HW_BRAKE))
                    except (KeyError, ValueError, IndexError):
                        pass
                w.writerow([name, g, len(sel), max(watts, default=""),
                            f"{statistics.mean(watts):.2f}" if watts else "", over, longest, slow,
                            int(max(sms)) if sms else "", int(max(mems)) if mems else ""])


if __name__ == "__main__":
    main()
