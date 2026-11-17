#!/usr/bin/env python3
"""Curated tables of the Q133/Q135 sitting from offset_climb.sh's output and the watchdog log.

  tables.py OUT WD_CSV DEST

OUT is offset_climb.sh's output folder (steps.csv, climb.log, mmvq_TAG.csv), WD_CSV the watchdog's log. Writes to DEST:
  steps.csv    one row per step and soak round: tag, clock, offset, bench checks, streaming GB/s, greedy, Xid, and the
               card's peaks over the step from the watchdog (clocks, power, temperature), plus how many 0.5 s samples
               were above the power cap in force and the longest run of them
  kernels.csv  per step: the fastest variant's time for one Flint token of matrix-vector products, per weight type
               (the bench's TOKEN_TOTAL rows), in microseconds
  watchdog.txt the watchdog's start, stop and trip lines and its sample count
"""
import csv
import os
import re
import sys

out, wd_path, dest = sys.argv[1:4]
os.makedirs(dest, exist_ok=True)

# tags of the plain steps, in order, from the log ("step TAG: mem offset set to X")
step_tags = [m.group(1) for m in re.finditer(r"step (\S+): (?:mem|core) offset set to", open(f"{out}/climb.log").read())]

# watchdog samples of the card under test (card 0): time, sm, mem, W, limit, C
samples = []
for row in csv.reader(open(wd_path)):
    if not row or row[0].startswith("#") or row[0] == "time":
        continue
    try:
        samples.append((row[0], int(row[5] or 0), int(row[6] or 0), float(row[7] or 0), float(row[8] or 0),
                        int(row[9] or 0)))
    except (ValueError, IndexError):
        continue


def over_cap(t0, t1):
    n, run, longest, cap = 0, 0, 0, None
    for t, _, _, w, lim, _ in samples:
        if t0 <= t < t1:
            cap = lim
            if w > lim:
                n += 1
                run += 1
                longest = max(longest, run)
            else:
                run = 0
    return cap, n, longest


rows = list(csv.DictReader(open(f"{out}/steps.csv")))
si = 0
out_rows = []
for k, r in enumerate(rows):
    t0 = r["time"]
    t1 = rows[k + 1]["time"] if k + 1 < len(rows) else "9999"
    if r["readback"].startswith("soak round"):
        i = r["readback"].split()[-1]
        tag = f"soak-{r['dom']}{r['offset']}-{i}"
        readback = ""
    else:
        tag = step_tags[si] if si < len(step_tags) else ""
        si += 1
        readback = r["readback"].split(" ")[0]
    cap, n_over, longest = over_cap(t0, t1)
    out_rows.append({
        "time": t0, "tag": tag, "clock": r["dom"], "offset": r["offset"], "readback": readback,
        "bench_exit": r["mmvq_exit"], "checks_pass": r["checks_pass"], "checks_fail": r["checks_fail"],
        "stream_gbs": r["stream_gbs"], "greedy": r["greedy"], "xid": r["xid"],
        "sm_peak_mhz": r["g_sm_max_mhz"], "mem_peak_mhz": r["g_mem_max_mhz"], "w_peak": r["g_w_max"],
        "c_peak": r["g_c_max"], "hottest_card_c": r["all_c_max"], "power_cap_w": f"{cap:.0f}" if cap else "",
        "samples": r["samples"], "samples_over_cap": n_over, "longest_run_over_cap": longest,
        "verdict": r["verdict"]})
with open(f"{dest}/steps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)

types = ["q8_0", "q5_0", "q4_1", "q5_K", "iq4_xs"]
with open(f"{dest}/kernels.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["tag", "clock", "offset"] + [f"{t}_token_us" for t in types] + [f"{t}_best_variant" for t in types])
    for r in out_rows:
        p = f"{out}/mmvq_{r['tag']}.csv"
        if not os.path.exists(p):
            continue
        best = {}
        for x in csv.DictReader(open(p)):
            if x["shape"] != "TOKEN_TOTAL" or not x["us_median"]:
                continue
            us = float(x["us_median"])
            if x["type"] not in best or us < best[x["type"]][0]:
                best[x["type"]] = (us, x["variant"])
        w.writerow([r["tag"], r["clock"], r["offset"]] + [f"{best[t][0]:.1f}" if t in best else "" for t in types]
                   + [best[t][1] if t in best else "" for t in types])

with open(f"{dest}/watchdog.txt", "w") as f:
    for line in open(wd_path):
        if line.startswith("#"):
            f.write(line[2:])
    f.write(f"{len(samples)} samples of all nine cards, every 0.5 s\n")
print(f"{len(out_rows)} rows; tables in {dest}")
