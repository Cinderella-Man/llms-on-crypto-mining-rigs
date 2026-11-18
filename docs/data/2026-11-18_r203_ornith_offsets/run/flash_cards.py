#!/usr/bin/env python3
"""R203 (Q1592): the offset kit's apply, card by card, and the cards' heat, power and memory clock during each run.

    python3 -I flash_cards.py apply APPLY_STEPS_CSV
    python3 -I flash_cards.py cards ARM=CARDS_CSV=FROM=TO ...

apply: the kit's out/apply_steps.csv (climb.py apply). Per card: the profile's memory and core offsets, the streaming
read at stock and at the offsets with the change, the peak core clock at stock and at the offsets, and whether the
greedy check equalled the card's own stock hash.
cards: the card log scripts/card_log.py writes every 0.5 s; FROM and TO ("YYYY-MM-DD HH:MM:SS") are the run's tasks.
Per card: the highest temperature and power, the samples above 80 W (the watchdog trips after 5 s above it), and
the median memory clock while busy (utilization above 0 %). Prints Markdown tables. Standard library only.
"""
import csv
import statistics
import sys


def apply_table(path):
    rows = list(csv.DictReader(open(path)))
    stock = {r["card"]: r for r in rows if r["phase"] == "stock"}
    print("| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz "
          "at stock | with the offsets | greedy check | verdict |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in (r for r in rows if r["phase"] == "apply"):
        s = stock[r["card"]]
        a, b = float(s["stream_gbs"]), float(r["stream_gbs"])
        print(f"| {r['card']} | +{int(r['mem_mts']):,} | +{r['core_mhz']} | {a:.2f} | {b:.2f} | "
              f"{(b / a - 1) * 100:+.2f} % | {int(s['measured_peak_mhz']):,} | {int(r['measured_peak_mhz']):,} | "
              f"{r['greedy']} as stock | {r['verdict']} |")


def cards_table(specs):
    print("| run | card | highest °C | highest W | samples above 80 W | busy memory clock median, MHz |")
    print("|---|---:|---:|---:|---:|---:|")
    for spec in specs:
        arm, path, t0, t1 = spec.split("=", 3)
        rows = [r for r in csv.DictReader(ln for ln in open(path) if not ln.startswith("#"))
                if t0 <= r["time"][:19] <= t1]
        n = sum(1 for k in rows[0] if k.endswith("_sm_mhz")) if rows else 0
        for g in range(n):
            def col(name):
                return [float(r[f"g{g}_{name}"]) for r in rows if r.get(f"g{g}_{name}") not in ("", None)]
            w, c = col("w"), col("c")
            busy = [float(r[f"g{g}_mem_mhz"]) for r in rows
                    if r.get(f"g{g}_util") not in ("", None) and float(r[f"g{g}_util"]) > 0]
            print(f"| {arm} | {g} | {max(c):.0f} | {max(w):.2f} | {sum(1 for x in w if x > 80)} | "
                  f"{statistics.median(busy):.0f} |")


if __name__ == "__main__":
    if sys.argv[1] == "apply":
        apply_table(sys.argv[2])
    else:
        cards_table(sys.argv[2:])
