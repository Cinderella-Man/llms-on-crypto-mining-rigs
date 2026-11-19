#!/usr/bin/env python3
"""Builds R217's tables from climb.py's work folder, copied off the rig (R200's script; only the docstring differs).

  tables.py WORK_DIR TABLES_DIR

WORK_DIR is the climb's work folder (R217: /var/tmp/r217 on the rig): out/climb.log, out/steps.csv, out/climb.json,
out/profile.json, and after a reboot out/apply.json and out/apply_steps.csv; wd.csv is the watchdog's half-second log
(the climb's watchdog and, appended after the reboot, apply's).

Writes, in TABLES_DIR:
  cards.csv        one row per card: stock peak core clock, the memory climb (last clean step, the step that failed
                   and why), the core climb (last clean offset and peak, the offset and peak that failed and why), the
                   profile (one step under each last clean step) and apply's verdict
  steps.csv        every check of every card at every step, as climb.py wrote it (apply's rows after the climb's)
  stream_gbs.csv   each card's streaming read (mmvq_pascal's STREAM_READ_PLAIN, GB/s) at stock and at each memory step
  heat_power.csv   one row per phase and card from the watchdog's samples: peak and mean temperature, peak and mean
                   power, samples and the longest run above 80 W, samples in a thermal or hardware slowdown, peak core
                   and memory clocks
  watchdog.txt     the watchdog's start, trip and stop lines
"""
import csv
import json
import os
import re
import statistics
import sys
import time

HW_SLOW, SW_THERMAL, HW_THERMAL, HW_BRAKE = 0x8, 0x20, 0x40, 0x80


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stamp(text):
    return time.mktime(time.strptime(text[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S"))


def phases(log_path, climb, apply_):
    """[(name, t0, t1)] from climb.log's markers: the servers loading, the stock rounds, the memory and core climbs,
    and apply."""
    marks = []
    pat = [("servers", r"climb: cards"), ("stock", r"stock: round 1"), ("memory", r"memory: step \+"),
           ("core", r"core: step to"), ("end", r"summary \(profile"), ("apply", r"apply: ")]
    if os.path.exists(log_path):
        with open(log_path) as f:
            for ln in f:
                for name, p in pat:
                    if re.search(p, ln) and not any(m[0] == name for m in marks):
                        marks.append((name, stamp(ln)))
    out = []
    order = [m for m in marks if m[0] not in ("apply",)]
    for i, (name, t0) in enumerate(order):
        if name == "end":
            break
        t1 = order[i + 1][1] if i + 1 < len(order) else (stamp(climb["ended"]) if climb and climb.get("ended") else t0)
        out.append((name, t0, t1))
    if climb and climb.get("started") and climb.get("ended"):
        out.append(("climb (all)", stamp(climb["started"]), stamp(climb["ended"])))
    if apply_ and apply_.get("started"):
        out.append(("apply", stamp(apply_["started"]), stamp(apply_.get("ended") or apply_["started"]) + 1))
    return out


def wd_rows(path):
    head, rows, events = None, [], []
    if not os.path.exists(path):
        return head, rows, events
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                events.append(ln[2:])
                continue
            v = ln.split(",")
            if head is None:
                head = {k: i for i, k in enumerate(v)}
                continue
            try:
                t = stamp(v[0])
            except ValueError:
                continue
            rows.append((t, v))
    return head, rows, events


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    work, tables = sys.argv[1], sys.argv[2]
    os.makedirs(tables, exist_ok=True)
    out = os.path.join(work, "out")
    climb, apply_ = read_json(os.path.join(out, "climb.json")), read_json(os.path.join(out, "apply.json"))
    prof = read_json(os.path.join(out, "profile.json"))
    steps = read_csv(os.path.join(out, "steps.csv"))
    asteps = read_csv(os.path.join(out, "apply_steps.csv"))

    # steps.csv: the climb's rows, then apply's
    if steps or asteps:
        cols = list((steps or asteps)[0].keys())
        with open(os.path.join(tables, "steps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run"] + cols)
            for run, rows in (("climb", steps), ("apply", asteps)):
                for r in rows:
                    w.writerow([run] + [r.get(c, "") for c in cols])

    # stream_gbs.csv: rows = stock and each memory step, columns = cards
    cards = [c["card"] for c in (prof or {}).get("cards", [])] or sorted({int(r["card"]) for r in steps})
    grid = {}
    for r in steps:
        if r["phase"] in ("stock", "memory") and r.get("stream_gbs"):
            grid.setdefault(r["step"] if r["phase"] == "memory" else "stock", {})[int(r["card"])] = r["stream_gbs"]
    with open(os.path.join(tables, "stream_gbs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["memory_step_mts"] + [f"card{g}" for g in cards])
        for step in sorted(grid, key=lambda s: -1 if s == "stock" else int(s.lstrip("+"))):
            w.writerow([step] + [grid[step].get(g, "") for g in cards])

    # cards.csv
    applied = {}
    for r in asteps:
        if r["phase"] == "apply":
            applied.setdefault(int(r["card"]), []).append(r["verdict"])
    stop = (apply_ or {}).get("stop_reason") or ""
    acards = {c["card"]: ("FAIL: " + stop if re.match(rf"card {c['card']}\b", stop) else
                          "reset to 0 (another card failed)" if stop else "")
              for c in (apply_ or {}).get("cards", [])}
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "pci", "stock_peak_mhz", "mem_last_clean_mts", "mem_failed_at_mts", "core_took_part",
                    "core_last_clean_mhz", "core_last_clean_peak_mhz", "core_failed_at_mhz", "core_failed_at_peak_mhz",
                    "failure_reason", "profile_mem_mts", "profile_core_mhz", "profile_expected_peak_mhz",
                    "apply_verdict"])
        for c in (prof or {}).get("cards", []):
            g = c["card"]
            fail = c.get("failure") or {}
            av = applied.get(g)
            verdict = acards.get(g) or (("PASS" if all(v == "PASS" for v in av) else "FAIL") if av else "")
            w.writerow([g, c["pci"].split(":", 1)[-1], c["stock_peak_mhz"], c["memory"]["last_clean_mts"],
                        c["memory"]["failed_at_mts"] or "", c["core"]["took_part"], c["core"]["last_clean_mhz"],
                        c["core"]["last_clean_peak_mhz"] or "", c["core"]["failed_at_mhz"] or "",
                        c["core"]["failed_at_peak_mhz"] or "", fail.get("reason", ""), c["memory_mts"], c["core_mhz"],
                        c["expected_peak_mhz"] or "", verdict])

    # heat_power.csv and watchdog.txt
    head, rows, events = wd_rows(os.path.join(work, "wd.csv"))
    with open(os.path.join(tables, "watchdog.txt"), "w") as f:
        f.write("\n".join(events) + "\n")
    if head:
        with open(os.path.join(tables, "heat_power.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_above_80w",
                        "longest_run_above_80w", "samples_slowdown", "peak_sm_mhz", "peak_mem_mhz"])
            for name, t0, t1 in phases(os.path.join(out, "climb.log"), climb, apply_):
                sel = [v for t, v in rows if t0 - 1 <= t <= t1 + 1]
                for g in cards:
                    def col(k):
                        return [num(v[head[f"g{g}_{k}"]]) for v in sel if f"g{g}_{k}" in head]
                    temps = [x for x in col("c") if x is not None]
                    watts = [x for x in col("w") if x is not None]
                    sms = [x for x in col("sm_mhz") if x is not None]
                    mems = [x for x in col("mem_mhz") if x is not None]
                    over = run = longest = 0
                    for x in watts:
                        run = run + 1 if x > 80 else 0
                        over += x > 80
                        longest = max(longest, run)
                    slow = 0
                    for v in sel:
                        try:
                            r = int(v[head[f"g{g}_reasons"]], 0)
                        except (KeyError, ValueError, IndexError):
                            continue
                        slow += bool(r & (HW_SLOW | SW_THERMAL | HW_THERMAL | HW_BRAKE))
                    w.writerow([name, g, len(sel), max(temps, default=""),
                                f"{statistics.mean(temps):.1f}" if temps else "", max(watts, default=""),
                                f"{statistics.mean(watts):.2f}" if watts else "", over, longest, slow,
                                int(max(sms)) if sms else "", int(max(mems)) if mems else ""])


if __name__ == "__main__":
    main()
