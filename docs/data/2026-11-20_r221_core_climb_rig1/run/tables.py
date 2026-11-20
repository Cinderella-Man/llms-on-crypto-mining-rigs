#!/usr/bin/env python3
"""Builds R221's tables from the work folder of `climb.py --from-profile`, copied off the rig (R202's script,
with the core rungs also for a card that climbed to the top without a failure or failed the check round).

  tables.py WORK_DIR TABLES_DIR

WORK_DIR is the climb's work folder (on the rig /var/tmp/r221): out/climb.log, out/steps.csv, out/profile.json and
wd.csv, the watchdog's half-second log.

Writes, in TABLES_DIR:
  cards.csv        one row per card: stock peak core clock, the memory the profile gave and its check round, whether
                   the card climbed the core, the last clean core step and the step that failed (offset, target peak,
                   the peak the card reached, why, the check's time and its limit, Xid lines), the result profile's
                   core, and streaming reads at stock and at the profile's memory
  steps.csv        every check of every card, as climb.py wrote it
  heat_power.csv   one row per phase and card from the watchdog's samples: peak and mean temperature, peak and mean
                   power, samples and the longest run above 80 W, samples in a thermal or hardware slowdown, peak core
                   and memory clocks
  core_rungs.csv   per card: the first core failure, the suggested core two climb steps under it (its target and
                   measured peak), and where that value comes from
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


def phases(log_path, prof):
    """[(name, t0, t1)] from climb.log's markers: the servers loading, the stock rounds, the profile's check round,
    the core climb, and the whole sitting."""
    pat = [("servers", r"climb from profile"), ("stock", r"stock: round 1"), ("_set", r"stock: card \d+ hash"),
           ("profile check", r"profile: check round at"), ("core climb", r"core: step to"),
           ("end", r"summary \(profile")]
    marks = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for ln in f:
                for name, p in pat:
                    if re.search(p, ln) and not any(m[0] == name for m in marks):
                        marks.append((name, stamp(ln)))
    out = []
    for i, (name, t0) in enumerate(marks):
        if name == "end":
            break
        t1 = marks[i + 1][1] if i + 1 < len(marks) else t0
        if not name.startswith("_"):     # _set: the seconds the profile's offsets were being set, between two phases
            out.append((name, t0, t1))
    if prof and prof.get("climb_started") and prof.get("climb_ended"):
        out.append(("sitting (all)", stamp(prof["climb_started"]), stamp(prof["climb_ended"])))
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
    prof = read_json(os.path.join(out, "profile.json")) or {}
    steps = read_csv(os.path.join(out, "steps.csv"))

    # steps.csv, as written
    if steps:
        cols = list(steps[0].keys())
        with open(os.path.join(tables, "steps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in steps:
                w.writerow([r.get(c, "") for c in cols])

    by = {}
    for r in steps:
        by.setdefault(int(r["card"]), []).append(r)
    cards = [c["card"] for c in prof.get("cards", [])]

    # cards.csv
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "pci", "stock_peak_mhz", "profile_mem_mts", "profile_core_in_mhz", "profile_check",
                    "core_climbed", "core_last_clean_mhz", "core_last_clean_target_peak_mhz",
                    "core_last_clean_measured_peak_mhz", "core_failed_at_mhz", "core_failed_at_target_peak_mhz",
                    "core_failed_at_measured_peak_mhz", "failure_reason", "failure_greedy_s", "failure_greedy_limit_s",
                    "failure_xid", "result_core_mhz", "result_expected_peak_mhz", "stream_gbs_stock",
                    "stream_gbs_profile"])
        for c in prof.get("cards", []):
            g = c["card"]
            rows = by.get(g, [])
            stock = next((r for r in rows if r["phase"] == "stock"), {})
            chk = next((r for r in rows if r["phase"] == "profile"), {})
            core = [r for r in rows if r["phase"] == "core"]
            clean = [r for r in core if r["verdict"] == "PASS"]
            failed = next((r for r in core if r["verdict"] == "FAIL"), None)
            last = clean[-1] if clean else None
            took = c["core"]["took_part"]
            w.writerow([g, c["pci"].split(":", 1)[-1], c["stock_peak_mhz"], c["memory"]["from_profile_mts"],
                        c["core"]["from_profile_mhz"], c["memory"]["profile_check"], took,
                        c["core"]["last_clean_mhz"] if took else "",
                        last["step"] if last else "", last["measured_peak_mhz"] if last else "",
                        failed["core_mhz"] if failed else "", failed["step"] if failed else "",
                        failed["measured_peak_mhz"] if failed else "", failed["reason"] if failed else "",
                        failed["greedy_s"] if failed else "", failed["greedy_limit_s"] if failed else "",
                        failed["xid"] if failed else "", c["core_mhz"], c["expected_peak_mhz"] or "",
                        stock.get("stream_gbs", ""), chk.get("stream_gbs", "")])

    # core_rungs.csv: per card the core two climb steps under its first failure in this sitting, which is the
    # profile's core (one step under the last clean step); the cards that kept the profile's core keep that value
    # (checked in this sitting's check round; their first core failure was in the first sitting, R200)
    with open(os.path.join(tables, "core_rungs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "memory_mts", "first_core_failure_mhz", "first_core_failure_target_peak_mhz",
                    "suggested_core_mhz", "suggested_target_peak_mhz", "suggested_measured_peak_mhz", "source"])
        for c in prof.get("cards", []):
            g = c["card"]
            rows = by.get(g, [])
            core = [r for r in rows if r["phase"] == "core"]
            failed = next((r for r in core if r["verdict"] == "FAIL"), None)
            if failed:
                at = next((r for r in core if int(r["core_mhz"]) == c["core_mhz"] and r["verdict"] == "PASS"), {})
                w.writerow([g, c["memory_mts"], failed["core_mhz"], failed["step"], c["core_mhz"], at.get("step", ""),
                            at.get("measured_peak_mhz", ""),
                            "two climb steps under this sitting's first failure (memory at the profile's value)"])
            elif c["core"]["took_part"] and core:
                at = next((r for r in core if int(r["core_mhz"]) == c["core_mhz"] and r["verdict"] == "PASS"), {})
                w.writerow([g, c["memory_mts"], "", "", c["core_mhz"], at.get("step", ""),
                            at.get("measured_peak_mhz", ""), "one climb step under the top of the climb, no failure"])
            elif c["memory_mts"] == 0 and c["memory"]["profile_check"] != "PASS":
                w.writerow([g, 0, "", "", 0, "", "", "failed the check round at the profile's memory: 0 and 0"])
            else:
                chk = next((r for r in rows if r["phase"] == "profile"), {})
                w.writerow([g, c["memory_mts"], "", "", c["core_mhz"], chk.get("expected_peak_mhz", ""),
                            chk.get("measured_peak_mhz", ""), "the profile's core, checked in this sitting's check round"])

    # heat_power.csv and watchdog.txt
    head, rows, events = wd_rows(os.path.join(work, "wd.csv"))
    with open(os.path.join(tables, "watchdog.txt"), "w") as f:
        f.write("\n".join(events) + "\n")
    if head:
        with open(os.path.join(tables, "heat_power.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_above_80w",
                        "longest_run_above_80w", "samples_slowdown", "peak_sm_mhz", "peak_mem_mhz"])
            for name, t0, t1 in phases(os.path.join(out, "climb.log"), prof):
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
