#!/usr/bin/env python3
"""Builds the R193 tables from r193.sh's output folder and the watchdog's log.

  tables.py OUT_DIR WD_CSV TABLES_DIR

Writes, in TABLES_DIR:
  cards.csv        one row per card: the stock streaming read, step 1 (offsets read back, the mmvq checks, the
                   streaming read at the offsets, the greedy check), the soak's mmvq rounds, cuda_memtest
  phases.csv       one row per phase and card from the watchdog's half-second samples: peak and mean temperature, peak
                   and mean power, samples and the longest run above 80 W, samples at the power cap or in a thermal or
                   hardware slowdown, peak core and memory clocks
  soak_10min.csv   the soak in 10-minute windows: each card's peak temperature and power, and the hottest card
  greedy.csv       every greedy check: tag, hash, same as the reference
  watchdog.txt     the watchdog's start, trip and stop lines
  exact_check.txt  the exact check's statistics blocks: the stock rerun and the run at the offsets
"""
import csv
import glob
import os
import re
import statistics
import sys

CAP_BIT, HW_SLOW, SW_THERMAL, HW_THERMAL, HW_BRAKE = 0x4, 0x8, 0x20, 0x40, 0x80


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def wd_rows(path):
    """The watchdog's samples: (time, [per card dict]); with --gpus the offsets of every card too."""
    rows, comments = [], []
    with open(path, newline="") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                comments.append(line[2:].rstrip())
                continue
            parts = line.rstrip("\n").split(",")
            if parts[0] == "time":
                header = parts
                n = sum(1 for h in header if h.endswith("_pstate"))
                offcol = {h: k for k, h in enumerate(header) if h.endswith("_off")}
                continue
            if header is None or len(parts) != len(header):
                continue
            cards = []
            for i in range(n):
                b = 4 + i * 8
                cards.append({"pstate": parts[b], "sm": num(parts[b + 1]), "mem": num(parts[b + 2]),
                              "w": num(parts[b + 3]), "c": num(parts[b + 5]),
                              "reasons": int(parts[b + 7], 16) if parts[b + 7] else 0,
                              "off": (parts[offcol[f"g{i}_mem_off"]], parts[offcol[f"g{i}_gpc_off"]])
                              if f"g{i}_mem_off" in offcol else None})
            rows.append((parts[0], cards))
    return rows, comments


def log_time(log, pattern, last=False):
    hits = [line[:19] for line in log if re.search(pattern, line)]
    return (hits[-1] if last else hits[0]) if hits else None


def phase_stats(samples, i):
    cs = [s[i]["c"] for _, s in samples if s[i]["c"] is not None]
    ws = [s[i]["w"] for _, s in samples if s[i]["w"] is not None]
    run = longest = 0
    for w in ws:
        run = run + 1 if w > 80 else 0
        longest = max(longest, run)
    rs = [s[i]["reasons"] for _, s in samples]
    return {
        "samples": len(samples),
        "peak_c": int(max(cs)) if cs else "",
        "mean_c": f"{statistics.mean(cs):.1f}" if cs else "",
        "peak_w": f"{max(ws):.2f}" if ws else "",
        "mean_w": f"{statistics.mean(ws):.2f}" if ws else "",
        "samples_over_80w": sum(1 for w in ws if w > 80),
        "longest_run_over_80w": longest,
        "samples_power_cap": sum(1 for r in rs if r & CAP_BIT),
        "samples_thermal_or_hw_slowdown": sum(1 for r in rs if r & (HW_SLOW | SW_THERMAL | HW_THERMAL | HW_BRAKE)),
        "peak_sm_mhz": int(max(s[i]["sm"] for _, s in samples if s[i]["sm"] is not None)) if samples else "",
        "peak_mem_mhz": int(max(s[i]["mem"] for _, s in samples if s[i]["mem"] is not None)) if samples else "",
        "offsets_seen": " ".join(sorted({f"{s[i]['off'][0]}/{s[i]['off'][1]}" for _, s in samples if s[i]["off"]})),
    }


def main():
    out, wd, tables = sys.argv[1:4]
    os.makedirs(tables, exist_ok=True)
    log = open(os.path.join(out, "r193.log")).read().splitlines() if os.path.exists(os.path.join(out, "r193.log")) else []
    samples, comments = wd_rows(wd)
    ncards = len(samples[0][1]) if samples else 9

    # phases from the driver's log
    soak = read_csv(os.path.join(out, "soak_cards.csv"))
    cards = read_csv(os.path.join(out, "cards.csv"))
    phases = {
        "stock references (exact check, server, stock greedy, streaming)": (
            log_time(log, r"pre: the exact check"), log_time(log, r"pre: done")),
        "step 1, card by card": (cards[0]["time"] if cards else None, log_time(log, r"cards: every card PASS|FAIL: card")),
        "soak, all nine cards": (min(r["start"] for r in soak) if soak else None, max(r["end"] for r in soak) if soak else None),
        "cuda_memtest, all nine cards": (log_time(log, r"post: cuda_memtest on every card"), log_time(log, r"post: cuda_memtest: ")),
        "exact check at the offsets": (log_time(log, r"post: cuda_memtest: "), log_time(log, r"post: exact check at")),
        "greedy check after the soak": (log_time(log, r"post: exact check at"), log_time(log, r"post: PASS|post: .*greedy")),
    }
    fields = ["phase", "start", "end", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_over_80w",
              "longest_run_over_80w", "samples_power_cap", "samples_thermal_or_hw_slowdown", "peak_sm_mhz", "peak_mem_mhz",
              "offsets_seen"]
    with open(os.path.join(tables, "phases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, (t0, t1) in phases.items():
            if not t0 or not t1:
                continue
            sel = [s for s in samples if t0 <= s[0][:19] <= t1]
            if not sel:
                continue
            for i in range(ncards):
                w.writerow({"phase": name, "start": t0, "end": t1, "card": i, **phase_stats(sel, i)})

    # the soak in 10-minute windows
    if soak:
        t0, t1 = min(r["start"] for r in soak), max(r["end"] for r in soak)
        sel = [s for s in samples if t0 <= s[0][:19] <= t1]
        with open(os.path.join(tables, "soak_10min.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["window", "from", "to"] + [f"card{i}_peak_c" for i in range(ncards)]
                        + [f"card{i}_peak_w" for i in range(ncards)] + ["hottest_card", "hottest_c"])
            if sel:
                import datetime as dt
                start = dt.datetime.strptime(sel[0][0][:19], "%Y-%m-%d %H:%M:%S")
                k = 0
                while True:
                    a = start + dt.timedelta(minutes=10 * k)
                    b = a + dt.timedelta(minutes=10)
                    win = [s for s in sel if a.strftime("%F %T") <= s[0][:19] < b.strftime("%F %T")]
                    if not win:
                        break
                    st = [phase_stats(win, i) for i in range(ncards)]
                    hot = max(range(ncards), key=lambda i: st[i]["peak_c"])
                    wr.writerow([k + 1, win[0][0][:19], win[-1][0][:19]] + [x["peak_c"] for x in st]
                                + [x["peak_w"] for x in st] + [hot, st[hot]["peak_c"]])
                    k += 1

    # one row per card
    stream = {r["card"]: r for r in read_csv(os.path.join(out, "stream.csv")) if r["tag"] == "stock"}
    step1 = {r["card"]: r for r in cards}
    mt = {r["card"]: r for r in read_csv(os.path.join(out, "memtest.csv"))}
    rows = []
    for i in range(ncards):
        c = str(i)
        s0 = stream.get(c, {}).get("stream_gbs", "")
        s1 = step1.get(c, {})
        rounds = [r for r in soak if r["card"] == c]
        gbs = [float(r["stream_gbs"]) for r in rounds if r["stream_gbs"]]
        m = mt.get(c, {})
        gain = ""
        if s0 and s1.get("stream_gbs"):
            gain = f"{(float(s1['stream_gbs']) / float(s0) - 1) * 100:+.1f}"
        rows.append({
            "card": i,
            "stock_stream_gbs": s0,
            "readback_mem_mts": s1.get("readback_mem", ""),
            "readback_core_mhz": s1.get("readback_core", ""),
            "step1_checks_pass": s1.get("checks_pass", ""),
            "step1_checks_fail": s1.get("checks_fail", ""),
            "step1_stream_gbs": s1.get("stream_gbs", ""),
            "stream_gain_pct": gain,
            "step1_greedy": s1.get("greedy", ""),
            "step1_xid": s1.get("xid", ""),
            "step1_verdict": s1.get("verdict", ""),
            "soak_rounds": len(rounds),
            "soak_rounds_pass": sum(1 for r in rounds if r["verdict"] == "PASS"),
            "soak_checks_fail": sum(int(r["checks_fail"] or 0) for r in rounds),
            "soak_stream_gbs_min": f"{min(gbs):.2f}" if gbs else "",
            "soak_stream_gbs_max": f"{max(gbs):.2f}" if gbs else "",
            "memtest_exit": m.get("exit_0_5_7_8_10", ""),
            "memtest_error_blocks": m.get("blocks_with_errors", ""),
            "memtest6_exit": m.get("exit_6", ""),
            "memtest6_error_blocks": m.get("blocks_with_errors_6", ""),
        })
    with open(os.path.join(tables, "cards.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # every greedy check
    ref = open(os.path.join(out, "greedy-ref.sha")).read().strip() if os.path.exists(os.path.join(out, "greedy-ref.sha")) else ""
    with open(os.path.join(tables, "greedy.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag", "hash", "same_as_reference"])
        def order(p):
            t = os.path.basename(p)[7:-4]
            m = re.match(r"soak-(\d+)$", t)
            return (1, int(m.group(1)), t) if m else ((0, 0, t) if t.startswith(("stock", "card")) else (2, 0, t))
        for p in sorted(glob.glob(os.path.join(out, "greedy-*.sha")), key=order):
            tag = os.path.basename(p)[7:-4]
            if tag == "ref":
                continue
            h = open(p).read().strip()
            wr.writerow([tag, h, "yes" if h == ref else "NO"])
        wr.writerow(["ref (the one-card sitting's stock hash)", ref, ""])

    with open(os.path.join(tables, "watchdog.txt"), "w") as f:
        f.write("\n".join(comments) + "\n")
        f.write(f"samples: {len(samples)}, from {samples[0][0] if samples else ''} to {samples[-1][0] if samples else ''}\n")

    with open(os.path.join(tables, "exact_check.txt"), "w") as f:
        for p in sorted(glob.glob(os.path.join(out, "kld-*.stats"))):
            f.write(f"== {os.path.basename(p)}\n{open(p).read()}\n")
        if os.path.exists(os.path.join(out, "exact_check.log")):
            f.write("== exact_check.log\n" + re.sub(r"/\S*/", "", open(os.path.join(out, "exact_check.log")).read()))


if __name__ == "__main__":
    main()
