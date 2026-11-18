#!/usr/bin/env python3
"""R201's tables from the chain's work folder and the Ornith folder, copied off the rig.

  python3 -I tables.py WORK_DIR ORNITH_DIR USE_PROFILE TABLES_DIR

WORK_DIR is soak_chain.sh's work folder (out/apply_steps.csv, soak/soak.csv, soak/rungs.csv, soak/soak.log,
soak/soaked.json, wd.csv), ORNITH_DIR model_check.sh's OUT (model.log), USE_PROFILE the profile made from soaked.json
by make_use.py. Writes:
  cards.csv        one row per card: each rung's offsets and differing checks, the value soaked.json holds, the use value
  rungs.csv        soak.py's rungs.csv as written on the rig (one row per rung and card)
  soak.csv         soak.py's soak.csv (one row per round and card: hash, seconds, verdict)
  apply.csv        climb.py apply's stock round and its check round at rung 1 (peaks, kernel checks, streaming reads)
  heat_power.csv   one row per phase (apply, the soak's server start, each rung, each raise between rungs) and card
                   from the watchdog's half-second samples: peak and mean temperature, peak and mean power, samples
                   and the longest run above 80 W, samples in a thermal or hardware slowdown, peak core and memory
                   clocks
  ornith_stock.csv one row per round of Ornith at stock: hash, prompt and generation tok/s of the two prompts
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


def stamp(text):
    return time.mktime(time.strptime(text[:19], "%Y-%m-%d %H:%M:%S"))


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def main():
    work, ornith, use_path, tables = sys.argv[1:5]
    os.makedirs(tables, exist_ok=True)
    sd = os.path.join(work, "soak")
    for name in ("rungs.csv", "soak.csv"):
        shutil.copyfile(os.path.join(sd, name), os.path.join(tables, name))
    shutil.copyfile(os.path.join(work, "out", "apply_steps.csv"), os.path.join(tables, "apply.csv"))

    # cards.csv
    with open(os.path.join(sd, "soaked.json")) as f:
        soaked = json.load(f)
    with open(use_path) as f:
        use = {c["card"]: c for c in json.load(f)["cards"]}
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        n = soaked["rungs"]
        w.writerow(["card", "stock_peak_mhz"]
                   + [x for r in range(1, n + 1) for x in (f"rung{r}_mem_mts", f"rung{r}_core_mhz", f"rung{r}_differs")]
                   + ["soaked_mem_mts", "soaked_core_mhz", "first_differed_rung", "first_differed_round",
                      "use_mem_mts", "use_core_mhz"])
        for c in soaked["cards"]:
            rs = c["soak"]["rungs"]
            fd = c["soak"]["first_differed"] or {}
            u = use[c["card"]]
            w.writerow([c["card"], c.get("stock_peak_mhz", "")]
                       + [x for r in rs for x in (r["memory_mts"], r["core_mhz"], r["differing_checks"])]
                       + [c["memory_mts"], c["core_mhz"], fd.get("rung", ""), fd.get("round", ""),
                          u["memory_mts"], u["core_mhz"]])

    # the phases, from soak.log
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
    cards = sorted(int(k[1:k.index("_")]) for k in head if re.fullmatch(r"g\d+_c", k))
    with open(os.path.join(tables, "heat_power.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_above_80w",
                    "longest_run_above_80w", "samples_slowdown", "peak_sm_mhz", "peak_mem_mhz"])
        for name, t0, t1 in phases:
            sel = [v for t, v in rows if t0 <= t < t1]
            for g in cards:
                def col(k):
                    return [x for x in (num(v[head[f"g{g}_{k}"]]) for v in sel) if x is not None]
                temps, watts, sms, mems = col("c"), col("w"), col("sm_mhz"), col("mem_mhz")
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
                w.writerow([name, g, len(sel), max(temps, default=""),
                            f"{statistics.mean(temps):.1f}" if temps else "", max(watts, default=""),
                            f"{statistics.mean(watts):.2f}" if watts else "", over, longest, slow,
                            int(max(sms)) if sms else "", int(max(mems)) if mems else ""])

    # ornith_stock.csv
    pat = re.compile(r"(\S+ \S+) stock round (\d+): greedy \S+ hash (\w+), (\d+) tokens, prompt tok/s ([\d.]+) ([\d.]+), "
                     r"generation tok/s ([\d.]+) ([\d.]+)( \| SAME as stock1)?")
    with open(os.path.join(ornith, "model.log")) as f, \
            open(os.path.join(tables, "ornith_stock.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["time", "round", "hash", "tokens", "prompt1_prompt_tok_s", "prompt2_prompt_tok_s",
                    "prompt1_gen_tok_s", "prompt2_gen_tok_s", "same_as_round1"])
        for ln in f:
            m = pat.match(ln)
            if m:
                g = m.groups()
                w.writerow([g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                            "reference" if g[1] == "1" else ("yes" if g[8] else "no")])


if __name__ == "__main__":
    main()
