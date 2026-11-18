#!/usr/bin/env python3
"""R201's second boot: the use profile's tables from the chain's work folder and the Ornith folder, copied off the rig.

  python3 -I tables_use.py WORK_DIR ORNITH_DIR TABLES_DIR

WORK_DIR is soak_chain.sh's work folder (out/apply_steps.csv, soak/soak.csv, soak/rungs.csv, soak/soak.log, wd.csv or
wd.csv.gz), ORNITH_DIR model_check.sh's OUT (model.log, with the stock and the profile rounds). Writes:
  cards.csv       one row per card: the offsets held, the soak's rounds, passes and differing checks, and the peaks
                  (temperature, power, core and memory clock) in the soak and in Ornith's run with the offsets
  rungs.csv       soak.py's rungs.csv as written on the rig (one rung: one row per card)
  soak.csv        soak.py's soak.csv (one row per round and card: hash, seconds, verdict)
  apply.csv       climb.py apply's stock round and its check round at the profile (peaks, kernel checks, streams)
  heat_power.csv  one row per phase (apply, the soak's server start, the soak, Ornith's load and rounds with the
                  offsets) and card from the watchdog's half-second samples
  ornith.csv      one row per round of Ornith, stock and with the offsets: hash, prompt and generation tok/s of the two
                  prompts, the same as stock's round 1 or not
and prints the medians of Ornith's generation tok/s, stock and with the offsets, and the change in percent.
"""
import csv
import gzip
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


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def main():
    work, ornith, tables = sys.argv[1:4]
    os.makedirs(tables, exist_ok=True)
    sd = os.path.join(work, "soak")
    for name in ("rungs.csv", "soak.csv"):
        shutil.copyfile(os.path.join(sd, name), os.path.join(tables, name))
    shutil.copyfile(os.path.join(work, "out", "apply_steps.csv"), os.path.join(tables, "apply.csv"))

    # Ornith's rounds, stock and with the offsets
    pat = re.compile(r"(\S+ \S+) (stock|profile) round (\d+): greedy \S+ hash (\w+), (\d+) tokens, "
                     r"prompt tok/s ([\d.]+) ([\d.]+), generation tok/s ([\d.]+) ([\d.]+)"
                     r"(?: \| (SAME|DIFFERS) (?:as|from) stock1)?")
    marks = {}
    rounds = []
    with open(os.path.join(ornith, "model.log")) as f:
        for ln in f:
            m = pat.match(ln)
            if m:
                rounds.append(m.groups())
                continue
            m = re.match(r"(\S+ \S+) (stock|profile): (loading|ready|done)", ln)
            if m:
                marks[(m.group(2), m.group(3))] = stamp(m.group(1))
    with open(os.path.join(tables, "ornith.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["time", "run", "round", "hash", "tokens", "prompt1_prompt_tok_s", "prompt2_prompt_tok_s",
                    "prompt1_gen_tok_s", "prompt2_gen_tok_s", "same_as_stock1"])
        for g in rounds:
            same = "reference" if g[1] == "stock" and g[2] == "1" else ("yes" if g[9] == "SAME" else "no")
            w.writerow(list(g[:9]) + [same])
    base = None
    for run in ("stock", "profile"):
        rs = [g for g in rounds if g[1] == run]
        if not rs:
            print(f"{run}: no rounds")
            continue
        # generation: the median of every round; prompt processing: of rounds 2 on (round 1 warms the server up)
        med = [statistics.median(float(g[7]) for g in rs), statistics.median(float(g[8]) for g in rs),
               statistics.median(float(g[5]) for g in rs[1:]), statistics.median(float(g[6]) for g in rs[1:])]
        same = sum(g[9] == "SAME" for g in rs)
        hashes = sorted({g[3] for g in rs})
        print(f"{run}: {len(rs)} rounds, {same} the same as stock1, hashes {' '.join(hashes)}; "
              f"generation tok/s median prompt 1 {med[0]:.2f}, prompt 2 {med[1]:.2f}; "
              f"prompt tok/s median of rounds 2 on, prompt 1 {med[2]:.2f}, prompt 2 {med[3]:.2f}")
        if run == "stock":
            base = med
        elif base:
            ch = [f"{100 * (a / b - 1):+.2f} %" for a, b in zip(med, base)]
            print(f"change: generation prompt 1 {ch[0]}, prompt 2 {ch[1]}; prompt processing prompt 1 {ch[2]}, "
                  f"prompt 2 {ch[3]}")

    # the phases, from soak.log and model.log
    soak_marks = []
    with open(os.path.join(sd, "soak.log")) as f:
        for ln in f:
            m = re.match(r"(\S+ \S+) (soak: cards|rung 1: [\d.]+ min on|soak done)", ln)
            if m:
                soak_marks.append((stamp(m.group(1)), m.group(2)))
    wd = os.path.join(work, "wd.csv")
    if not os.path.exists(wd):
        wd += ".gz"
    with opener(wd) as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    head = {k: i for i, k in enumerate(lines[0].split(","))}
    rows = []
    for ln in lines[1:]:
        v = ln.split(",")
        try:
            rows.append((stamp(v[0]), v))
        except ValueError:
            pass
    servers = next(s for s, n in soak_marks if n.startswith("soak: cards"))
    rung1 = next(s for s, n in soak_marks if n.startswith("rung 1"))
    soak_end = next((s for s, n in soak_marks if n.startswith("soak done")), rows[-1][0])
    phases = [("apply", rows[0][0], servers), ("servers", servers, rung1), ("soak", rung1, soak_end)]
    if ("profile", "loading") in marks:
        load0 = marks[("profile", "loading")]
        ready = marks.get(("profile", "ready"), rows[-1][0])
        done = marks.get(("profile", "done"), rows[-1][0])
        phases += [("ornith load", load0, ready), ("ornith rounds", ready, done + 1)]
    cards = sorted(int(k[1:k.index("_")]) for k in head if re.fullmatch(r"g\d+_c", k))
    peaks = {}
    with open(os.path.join(tables, "heat_power.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_above_80w",
                    "longest_run_above_80w", "samples_slowdown", "peak_sm_mhz", "peak_mem_mhz"])
        for name, t0, t1 in phases:
            sel = [v for ts, v in rows if t0 <= ts < t1]
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
                row = [name, g, len(sel), max(temps, default=""),
                       f"{statistics.mean(temps):.1f}" if temps else "", max(watts, default=""),
                       f"{statistics.mean(watts):.2f}" if watts else "", over, longest, slow,
                       int(max(sms)) if sms else "", int(max(mems)) if mems else ""]
                w.writerow(row)
                peaks[(name, g)] = row

    # cards.csv
    with open(os.path.join(sd, "rungs.csv")) as f:
        rung = {int(r["card"]): r for r in csv.DictReader(f)}
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "mem_mts", "core_mhz", "soak_rounds", "soak_passes", "soak_differs",
                    "first_differing_round", "soak_peak_c", "soak_peak_w", "soak_peak_sm_mhz", "soak_peak_mem_mhz",
                    "ornith_peak_c", "ornith_peak_w", "ornith_peak_sm_mhz", "ornith_peak_mem_mhz"])
        for g in cards:
            r = rung[g]
            s = peaks.get(("soak", g), [""] * 12)
            o = peaks.get(("ornith rounds", g), [""] * 12)
            w.writerow([g, r["mem_mts"], r["core_mhz"], r["rounds"], r["passes"], r["differs"],
                        r["first_differing_round"], s[3], s[5], s[10], s[11], o[3], o[5], o[10], o[11]])


if __name__ == "__main__":
    main()
