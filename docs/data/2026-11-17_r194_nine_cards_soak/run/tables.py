#!/usr/bin/env python3
"""Builds the R194 tables from r194.sh's output folder and the watchdog's log.

  tables.py OUT_DIR WD_CSV TABLES_DIR [R193_CARDS_CSV]

R193_CARDS_CSV (docs/data/2026-11-17_r193_nine_cards_soak/tables/cards.csv) gives each card's stock streaming read,
which R194 did not measure again.

Writes, in TABLES_DIR:
  cards.csv        one row per card: the offsets set and read back, the expected and measured peak core clock, step 1
                   (the mmvq checks, the streaming read against R193's stock read, the greedy check), the soak (rounds,
                   checks, streaming read, peak temperature, power and clock), cuda_memtest
  own_check.csv    each card during its own step-1 check, from the watchdog's half-second samples: mean and peak power,
                   readings above 80 W and the longest run of them, readings at the power cap, peak and median core
                   clock, memory clock, peak and mean temperature
  phases.csv       one row per phase and card from the watchdog's samples: peak and mean temperature, peak and mean
                   power, samples and the longest run above 80 W, samples at the power cap or in a thermal or hardware
                   slowdown, peak core and memory clocks, the offsets seen
  soak_10min.csv   the soak in 10-minute windows: each card's peak temperature and power, and the hottest card
  greedy.csv       every greedy check: tag, hash, same as the reference
  watchdog.txt     the watchdog's start, trip and stop lines
  exact_check.txt  the exact check: R193's stock block (the reference), the block at the offsets, the check's log
"""
import csv
import datetime as dt
import glob
import os
import re
import statistics
import sys

CAP_BIT, HW_SLOW, SW_THERMAL, HW_THERMAL, HW_BRAKE = 0x4, 0x8, 0x20, 0x40, 0x80
STEP_MHZ = 12.65  # the core clock moves in whole steps of about this much (Q133: 291 MHz in 23 steps); the rest is dropped
STOCK_PEAK = {0: 1860, 1: 1860, 2: 1911, 3: 1911, 4: 1885, 5: 1911, 6: 1885, 7: 1911, 8: 1898}  # R193, MHz


def read_csv(path):
    if not path or not os.path.exists(path):
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


def longest_over(ws, limit=80):
    run = longest = 0
    for w in ws:
        run = run + 1 if w > limit else 0
        longest = max(longest, run)
    return longest


def phase_stats(samples, i):
    cs = [s[i]["c"] for _, s in samples if s[i]["c"] is not None]
    ws = [s[i]["w"] for _, s in samples if s[i]["w"] is not None]
    rs = [s[i]["reasons"] for _, s in samples]
    sms = [s[i]["sm"] for _, s in samples if s[i]["sm"] is not None]
    mems = [s[i]["mem"] for _, s in samples if s[i]["mem"] is not None]
    return {
        "samples": len(samples),
        "peak_c": int(max(cs)) if cs else "",
        "mean_c": f"{statistics.mean(cs):.1f}" if cs else "",
        "peak_w": f"{max(ws):.2f}" if ws else "",
        "mean_w": f"{statistics.mean(ws):.2f}" if ws else "",
        "samples_over_80w": sum(1 for w in ws if w > 80),
        "longest_run_over_80w": longest_over(ws),
        "samples_power_cap": sum(1 for r in rs if r & CAP_BIT),
        "samples_thermal_or_hw_slowdown": sum(1 for r in rs if r & (HW_SLOW | SW_THERMAL | HW_THERMAL | HW_BRAKE)),
        "peak_sm_mhz": int(max(sms)) if sms else "",
        "peak_mem_mhz": int(max(mems)) if mems else "",
        "offsets_seen": " ".join(sorted({f"{s[i]['off'][0]}/{s[i]['off'][1]}" for _, s in samples if s[i]["off"]})),
    }


def window(samples, t0, t1):
    return [s for s in samples if t0 <= s[0][:19] < t1]


def main():
    out, wd, tables = sys.argv[1:4]
    r193_cards = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(tables, exist_ok=True)
    logp = os.path.join(out, "r194.log")
    log = open(logp).read().splitlines() if os.path.exists(logp) else []
    samples, comments = wd_rows(wd)
    ncards = len(samples[0][1]) if samples else 9

    soak = read_csv(os.path.join(out, "soak_cards.csv"))
    cards = read_csv(os.path.join(out, "cards.csv"))
    # a card that was set but never judged (its benchmark hung and the run was stopped by hand): its row from the
    # driver's log and the benchmark's own log, which prints each check as it runs (the CSV comes only at the end)
    xid_boot = ""
    st = os.path.join(os.path.dirname(os.path.abspath(out)), "end_state.txt")
    if os.path.exists(st):
        lines = open(st).read().splitlines()
        xid_boot = next((lines[k + 1] for k, x in enumerate(lines) if x.startswith("## Xid")), "")
    judged = {r["card"] for r in cards}
    for line in log:
        m = re.match(r"(\S+ \S+) card (\d+): set \+(\d+) MT/s and \+(\d+) MHz, read back .*: (\d+) (\d+)$", line)
        if m and m.group(2) not in judged:
            lp = os.path.join(out, f"mmvq_card{m.group(2)}.log")
            txt = open(lp).read().splitlines() if os.path.exists(lp) else []
            stream = next((re.search(r"([\d.]+) GB/s plain loads", x).group(1) for x in txt if "plain loads" in x), "")
            cards.append({"time": m.group(1), "card": m.group(2), "set_mem": m.group(3), "set_core": m.group(4),
                          "readback_mem": m.group(5), "readback_core": m.group(6), "mmvq_exit": "hung, stopped by hand",
                          "checks_pass": str(sum(1 for x in txt if x.endswith(" PASS"))),
                          "checks_fail": str(sum(1 for x in txt if x.endswith(" FAIL"))), "stream_gbs": stream,
                          "greedy": "not run", "xid": xid_boot, "mmvq_peak_rss_kb": "", "verdict": "FAIL"})
    step1_end = log_time(log, r"cards: every card PASS|FAIL: card|reset: card")

    # every failed kernel check, from the benchmark logs
    with open(os.path.join(tables, "failed_checks.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["card", "shape", "type", "variant", "gbs", "err_over_tol", "checks_run_before"])
        for r in cards:
            lp = os.path.join(out, f"mmvq_card{r['card']}.log")
            txt = [x for x in open(lp).read().splitlines() if x.endswith((" PASS", " FAIL"))] if os.path.exists(lp) else []
            for k, x in enumerate(txt):
                if x.endswith(" FAIL"):
                    p = x.split()
                    wr.writerow([r["card"], p[0], p[1], p[2], p[6], p[-2], k])
    phases = {
        "server load at offset 0": (log_time(log, r"server: pid"), log_time(log, r"serve: the server holds")),
        "step 1, card by card": (cards[0]["time"] if cards else None, step1_end),
        "soak, all nine cards": (min(r["start"] for r in soak) if soak else None, max(r["end"] for r in soak) if soak else None),
        "cuda_memtest, all nine cards": (log_time(log, r"post: cuda_memtest on every card"), log_time(log, r"post: cuda_memtest: ")),
        "exact check at the offsets": (log_time(log, r"post: the exact check at the offsets: a base"),
                                       log_time(log, r"post: exact check at the offsets: ")),
    }
    fields = ["phase", "start", "end", "card", "samples", "peak_c", "mean_c", "peak_w", "mean_w", "samples_over_80w",
              "longest_run_over_80w", "samples_power_cap", "samples_thermal_or_hw_slowdown", "peak_sm_mhz", "peak_mem_mhz",
              "offsets_seen"]
    soak_stats = {}
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
                st = phase_stats(sel, i)
                if name.startswith("soak"):
                    soak_stats[i] = st
                w.writerow({"phase": name, "start": t0, "end": t1, "card": i, **st})

    # each card during its own step-1 check: from its set to the next card's set (the last one to the end of step 1)
    own = {}
    with open(os.path.join(tables, "own_check.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["card", "from", "to", "samples", "mean_w", "peak_w", "samples_over_80w", "longest_run_over_80w",
                     "samples_power_cap", "peak_sm_mhz", "median_sm_mhz", "peak_mem_mhz", "peak_c", "mean_c"])
        for k, r in enumerate(cards):
            i = int(r["card"])
            t0 = r["time"]
            t1 = cards[k + 1]["time"] if k + 1 < len(cards) else (step1_end or "9999")
            sel = window(samples, t0, t1)
            if not sel:
                continue
            ws = [s[i]["w"] for _, s in sel if s[i]["w"] is not None]
            sms = [s[i]["sm"] for _, s in sel if s[i]["sm"] is not None]
            cs = [s[i]["c"] for _, s in sel if s[i]["c"] is not None]
            own[i] = {"peak_sm": int(max(sms)), "mean_w": f"{statistics.mean(ws):.2f}", "peak_w": f"{max(ws):.2f}",
                      "peak_c": int(max(cs))}
            wr.writerow([i, t0, t1[:19], len(sel), f"{statistics.mean(ws):.2f}", f"{max(ws):.2f}",
                         sum(1 for x in ws if x > 80), longest_over(ws),
                         sum(1 for _, s in sel if s[i]["reasons"] & CAP_BIT), int(max(sms)), int(statistics.median(sms)),
                         int(max(s[i]["mem"] for _, s in sel)), int(max(cs)), f"{statistics.mean(cs):.1f}"])

    # the soak in 10-minute windows
    if soak:
        t0, t1 = min(r["start"] for r in soak), max(r["end"] for r in soak)
        sel = [s for s in samples if t0 <= s[0][:19] <= t1]
        with open(os.path.join(tables, "soak_10min.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["window", "from", "to"] + [f"card{i}_peak_c" for i in range(ncards)]
                        + [f"card{i}_peak_w" for i in range(ncards)] + ["hottest_card", "hottest_c"])
            if sel:
                start = dt.datetime.strptime(sel[0][0][:19], "%Y-%m-%d %H:%M:%S")
                k = 0
                while True:
                    a = start + dt.timedelta(minutes=10 * k)
                    b = a + dt.timedelta(minutes=10)
                    win = window(sel, a.strftime("%F %T"), b.strftime("%F %T"))
                    if not win:
                        break
                    st = [phase_stats(win, i) for i in range(ncards)]
                    hot = max(range(ncards), key=lambda i: st[i]["peak_c"])
                    wr.writerow([k + 1, win[0][0][:19], win[-1][0][:19]] + [x["peak_c"] for x in st]
                                + [x["peak_w"] for x in st] + [hot, st[hot]["peak_c"]])
                    k += 1

    # one row per card
    stock_stream = {r["card"]: r["stock_stream_gbs"] for r in read_csv(r193_cards)}
    step1 = {r["card"]: r for r in cards}
    mt = {r["card"]: r for r in read_csv(os.path.join(out, "memtest.csv"))}
    rows = []
    for i in range(ncards):
        c = str(i)
        s0 = stock_stream.get(c, "")
        s1 = step1.get(c, {})
        rounds = [r for r in soak if r["card"] == c]
        gbs = [float(r["stream_gbs"]) for r in rounds if r["stream_gbs"]]
        m = mt.get(c, {})
        ss = soak_stats.get(i, {})
        gain = ""
        if s0 and s1.get("stream_gbs"):
            gain = f"{(float(s1['stream_gbs']) / float(s0) - 1) * 100:+.1f}"
        core = int(s1["set_core"]) if s1.get("set_core") else None
        rows.append({
            "card": i,
            "stock_peak_mhz_r193": STOCK_PEAK.get(i, ""),
            "set_mem_mts": s1.get("set_mem", ""),
            "set_core_mhz": s1.get("set_core", ""),
            "core_steps": int(core / STEP_MHZ) if core is not None else "",
            "expected_peak_mhz": round(STOCK_PEAK[i] + int(core / STEP_MHZ) * STEP_MHZ) if core is not None else "",
            "readback_mem_mts": s1.get("readback_mem", ""),
            "readback_core_mhz": s1.get("readback_core", ""),
            "step1_peak_sm_mhz": own.get(i, {}).get("peak_sm", ""),
            "step1_mean_w": own.get(i, {}).get("mean_w", ""),
            "step1_peak_c": own.get(i, {}).get("peak_c", ""),
            "step1_checks_pass": s1.get("checks_pass", ""),
            "step1_checks_fail": s1.get("checks_fail", ""),
            "stock_stream_gbs_r193": s0,
            "step1_stream_gbs": s1.get("stream_gbs", ""),
            "stream_gain_pct": gain,
            "step1_greedy": s1.get("greedy", ""),
            "step1_xid": s1.get("xid", ""),
            "step1_verdict": s1.get("verdict", ""),
            "soak_rounds": len(rounds),
            "soak_rounds_pass": sum(1 for r in rounds if r["verdict"] == "PASS"),
            "soak_checks_pass": sum(int(r["checks_pass"] or 0) for r in rounds),
            "soak_checks_fail": sum(int(r["checks_fail"] or 0) for r in rounds),
            "soak_stream_gbs_min": f"{min(gbs):.2f}" if gbs else "",
            "soak_stream_gbs_max": f"{max(gbs):.2f}" if gbs else "",
            "soak_peak_c": ss.get("peak_c", ""),
            "soak_mean_w": ss.get("mean_w", ""),
            "soak_peak_w": ss.get("peak_w", ""),
            "soak_longest_run_over_80w": ss.get("longest_run_over_80w", ""),
            "soak_peak_sm_mhz": ss.get("peak_sm_mhz", ""),
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
    refp = os.path.join(out, "greedy-ref.sha")
    ref = open(refp).read().strip() if os.path.exists(refp) else ""
    with open(os.path.join(tables, "greedy.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tag", "hash", "same_as_reference"])

        def order(p):
            t = os.path.basename(p)[7:-4]
            m = re.match(r"soak-(\d+)$", t)
            return (1, int(m.group(1)), t) if m else ((0, int(t[4:]), t) if re.match(r"card\d+$", t) else (2, 0, t))
        for p in sorted(glob.glob(os.path.join(out, "greedy-*.sha")), key=order):
            tag = os.path.basename(p)[7:-4]
            if tag == "ref":
                continue
            h = open(p).read().strip()
            wr.writerow([tag, h, "yes" if h == ref else "NO"])
        wr.writerow(["ref (the stock hash of the Q133/Q135 sitting and R193)", ref, ""])

    with open(os.path.join(tables, "watchdog.txt"), "w") as f:
        f.write("\n".join(comments) + "\n")
        f.write(f"samples: {len(samples)}, from {samples[0][0] if samples else ''} to {samples[-1][0] if samples else ''}\n")

    with open(os.path.join(tables, "exact_check.txt"), "w") as f:
        for name, title in (("kld-stock-r193.stats", "R193's stock block (the reference)"),
                            ("kld-offsets.stats", "at the offsets: the rerun against a base file written at the offsets")):
            p = os.path.join(out, name)
            if os.path.exists(p):
                f.write(f"== {name}: {title}\n{open(p).read()}\n")
        p = os.path.join(out, "exact", "exact_check.log")
        if os.path.exists(p):
            f.write("== exact_check.log\n" + re.sub(r"/\S*/", "", open(p).read()))


if __name__ == "__main__":
    main()
