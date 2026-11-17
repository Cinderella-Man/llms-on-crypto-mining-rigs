#!/usr/bin/env python3
"""R200's apply-and-soak tables, one row per run and card.

  soak_tables.py OUT_CSV NAME=WORK_DIR [NAME=WORK_DIR ...]

A NAME ending in "stock" is a stock control on the same boot as the WORK_DIR's apply: its offsets are reported as 0
and apply's columns left empty.

Each WORK_DIR is a work folder copied off the rig after `climb.py apply` and soak.py ran in it (out/apply.json,
out/apply_steps.csv, soak/soak.csv, wd.csv). For every card: the offsets apply set and its stock peak core clock on that
boot, the expected and measured peak and the streaming read in apply's check round, then the soak: rounds run, passes,
the round that failed, the slowest check, and from the watchdog's half-second samples during the soak the card's
highest temperature, highest power reading, readings above 80 W, and highest core and memory clocks.
"""
import csv
import json
import os
import sys
import time


def stamp(text):
    return time.mktime(time.strptime(text[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S"))


def main():
    out = sys.argv[1]
    rows = []
    for arg in sys.argv[2:]:
        name, work = arg.split("=", 1)
        stock = name.endswith("stock")   # a stock control: every offset 0, no apply of its own
        ap = json.load(open(os.path.join(work, "out", "apply.json")))
        steps = list(csv.DictReader(open(os.path.join(work, "out", "apply_steps.csv"))))
        sp = os.path.join(work, "soak", "soak.csv")
        soak = list(csv.DictReader(open(sp))) if os.path.exists(sp) else []
        t0 = stamp(soak[0]["time"]) - 40 if soak else None   # the first round starts about one check before its row
        t1 = stamp(soak[-1]["time"]) if soak else None
        head, samples = None, []
        with open(os.path.join(work, "wd.csv")) as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                v = ln.rstrip("\n").split(",")
                if head is None or v[0] == "time":
                    head = {k: i for i, k in enumerate(v)}
                    continue
                try:
                    t = stamp(v[0])
                except ValueError:
                    continue
                if t0 and t0 <= t <= t1 + 1:
                    samples.append(v)
        for c in ap["cards"]:
            g = c["card"]
            st = [r for r in steps if int(r["card"]) == g and r["phase"] == "stock"]
            chk = [] if stock else [r for r in steps if int(r["card"]) == g and r["phase"] == "apply"]
            mine = [r for r in soak if int(r["card"]) == g]
            fails = [r for r in mine if r["verdict"] == "FAIL"]

            def col(k, cast=float):
                key = f"g{g}_{k}"
                vals = []
                for v in samples:
                    try:
                        vals.append(cast(v[head[key]]))
                    except (KeyError, ValueError, IndexError):
                        pass
                return vals
            w = col("w")
            rows.append({
                "run": name, "card": g, "boot_stock_peak_mhz": st[0]["measured_peak_mhz"] if st else "",
                "mem_mts": 0 if stock else c["mem"], "core_mhz": 0 if stock else c["core"],
                "core_steps": 0 if stock else c["bins"],
                "apply_expected_peak_mhz": chk[0]["expected_peak_mhz"] if chk else "",
                "apply_measured_peak_mhz": chk[0]["measured_peak_mhz"] if chk else "",
                "apply_greedy": chk[0]["greedy"] if chk else "", "apply_stream_gbs": chk[0]["stream_gbs"] if chk else "",
                "soak_rounds": len(mine), "soak_passes": len(mine) - len(fails),
                "soak_failed_round": fails[0]["round"] if fails else "",
                "soak_slowest_check_s": max((float(r["seconds"]) for r in mine), default=""),
                "soak_peak_c": max(col("c"), default=""), "soak_peak_w": max(w, default=""),
                "soak_readings_above_80w": sum(x > 80 for x in w),
                "soak_peak_sm_mhz": max(col("sm_mhz", int), default=""),
                "soak_peak_mem_mhz": max(col("mem_mhz", int), default=""),
            })
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
