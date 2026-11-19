#!/usr/bin/env python3
"""Do a card's wrong answers under a clock offset come when the card is warmer? (FN119, Q1615)

Two steps:

  build  reads, for each soak, the kit's soak.csv (each card-round's end time and its seconds since the round began)
         and the watchdog's half-second samples (wd.csv.gz: per card pstate, core and memory clock, watts, power
         limit, temperature, memory used, throttle reasons, and the offsets it read), and writes one row per
         card-round with the card's heat over that round:
           OUT/heat.csv.gz   run, card, round, the window, samples, temperature (mean, peak, first sample), watts
                             (mean, peak), core and memory clock (mean), the offsets the watchdog read, and the mean
                             temperature of all nine cards over the same window
  table  joins heat.csv.gz with FN118's rounds.csv.gz (offsets, wrong answers, minutes since each change) and writes:
           OUT/errors.csv    each wrong answer: its heat, and its place by heat and by time among the rounds the
                             card ran at that value
           OUT/tests.csv     the average place of the wrong answers by each measure, and its chance under even
                             spreading (drawing each wrong answer's round at random from its stratum)
           OUT/power.csv     how often the test would see heat if each degree multiplied the chance of a wrong
                             answer by 1.25, 1.5, 2, 3 or 5
           OUT/warmup.csv    temperature by minutes since each soak's first round: the warm-up
           OUT/confound.csv  how closely temperature follows time within the strata that erred
           OUT/cards.csv     each card's settled temperature and watts per soak, and its wrong answers

  python3 -I errors_temperature.py build --out tables --run try1=SOAK_CSV,WD_CSV_GZ --run r201=... ...
  python3 -I errors_temperature.py table FN118_TABLES/rounds.csv.gz tables/heat.csv.gz tables

A round: all cards' checks start together; the kit then waits on each card in turn and writes its row with the
seconds since the round began. A card-round's window is [row time + 0.5 - seconds, row time + 0.5] (the row time is
written to the whole second).

A stratum is one card at one memory and core value, pooled over every soak, as in FN118. A soak that stops at the
first wrong answer (R200's tries) ends at that answer, so the answer is its soak's last round by design; its place by
time or by heat says nothing, and the tests are given with and without those soaks' answers.
"""
import argparse
import bisect
import csv
import gzip
import io
import math
import os
import random
import statistics
import sys
from datetime import datetime

TFMT = "%Y-%m-%d %H:%M:%S"
FIELDS = ("pstate", "sm_mhz", "mem_mhz", "w", "limit_w", "c", "used_mib", "reasons")
WARMUP_BINS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 15), (15, 30), (30, 60), (60, 90), (90, None)]
TIME_MATCH_MIN = 5.0


def epoch(s):
    if "." in s:
        return datetime.strptime(s, TFMT + ".%f").timestamp()
    return datetime.strptime(s, TFMT).timestamp()


def read_csv(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_wd(path):
    """The watchdog's samples: sorted times and, per sample, {card: {field: value}} plus the offsets it read."""
    times, rows = [], []
    with gzip.open(path, "rt", newline="") as f:
        head = None
        for ln in f:
            if ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split(",")
            if head is None:
                head = parts
                cards = sorted({int(h[1:].split("_")[0]) for h in head if h[:1] == "g" and h[1:2].isdigit()})
                col = {h: i for i, h in enumerate(head)}
                continue
            if len(parts) != len(head):
                continue
            s = {}
            for g in cards:
                s[g] = {k: parts[col[f"g{g}_{k}"]] for k in FIELDS}
                s[g]["mem_off"] = parts[col[f"g{g}_mem_off"]]
                s[g]["gpc_off"] = parts[col[f"g{g}_gpc_off"]]
            times.append(epoch(parts[0]))
            rows.append(s)
    order = sorted(range(len(times)), key=times.__getitem__)
    return [times[i] for i in order], [rows[i] for i in order]


def build(a):
    out = []
    for spec in a.run:
        name, paths = spec.split("=", 1)
        soak, wd = paths.split(",", 1)
        times, samples = read_wd(wd)
        rows = read_csv(soak)
        short = 0
        for r in rows:
            g, end = int(r["card"]), epoch(r["time"]) + 0.5
            start = end - float(r["seconds"])
            i, j = bisect.bisect_left(times, start), bisect.bisect_right(times, end)
            win = [s for s in samples[i:j] if g in s]
            temps = [x for x in (num(s[g]["c"]) for s in win) if x is not None]
            watts = [x for x in (num(s[g]["w"]) for s in win) if x is not None]
            sm = [x for x in (num(s[g]["sm_mhz"]) for s in win) if x is not None]
            mem = [x for x in (num(s[g]["mem_mhz"]) for s in win) if x is not None]
            rig = [statistics.fmean(v for v in (num(s[h]["c"]) for h in s) if v is not None) for s in win]
            offs = sorted({(s[g]["mem_off"], s[g]["gpc_off"]) for s in win})
            if len(temps) < 4:
                short += 1
            out.append({
                "run": name, "card": g, "round": int(r["round"]),
                "start": datetime.fromtimestamp(start).strftime(TFMT + ".%f")[:-5],
                "end": datetime.fromtimestamp(end).strftime(TFMT + ".%f")[:-5],
                "samples": len(temps),
                "temp_mean": f"{statistics.fmean(temps):.2f}" if temps else "",
                "temp_max": f"{max(temps):.0f}" if temps else "",
                "temp_first": f"{temps[0]:.0f}" if temps else "",
                "watts_mean": f"{statistics.fmean(watts):.2f}" if watts else "",
                "watts_max": f"{max(watts):.2f}" if watts else "",
                "sm_mhz_mean": f"{statistics.fmean(sm):.0f}" if sm else "",
                "mem_mhz_mean": f"{statistics.fmean(mem):.0f}" if mem else "",
                "wd_mem_off": " ".join(o[0] for o in offs), "wd_gpc_off": " ".join(o[1] for o in offs),
                "rig_temp_mean": f"{statistics.fmean(rig):.2f}" if rig else ""})
        print(f"{name}: {len(rows)} card-rounds, {len(times)} watchdog samples, {short} card-rounds with fewer "
              f"than 4 samples")
    os.makedirs(a.out, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out)
    with gzip.GzipFile(os.path.join(a.out, "heat.csv.gz"), "wb", mtime=0) as f:
        f.write(buf.getvalue().encode())
    print(f"{len(out)} card-rounds -> {a.out}/heat.csv.gz")


def midrank(values, x):
    lt = sum(1 for v in values if v < x)
    eq = sum(1 for v in values if v == x)
    return (lt + (eq - 1) / 2) / (len(values) - 1) if len(values) > 1 else 0.5


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


MEASURES = [("temp_mean", "temperature, mean over the round"), ("temp_max", "temperature, peak over the round"),
            ("watts_mean", "watts, mean over the round"), ("sm_mhz_mean", "core clock, mean over the round"),
            ("rig_temp_mean", "all nine cards' temperature, mean over the round"),
            ("own_minutes", "minutes since the card's own change (FN118)"),
            ("rig_minutes", "minutes since any card's change (FN118)")]


def table(a):
    heat = {(r["run"], int(r["card"]), int(r["round"])): r for r in read_csv(a.heat)}
    rows = read_csv(a.rounds)
    for r in rows:
        for k in ("card", "round", "rung", "mem_mts", "core_mhz", "differs", "own_rounds", "rig_rounds"):
            r[k] = int(r[k])
        h = heat.pop((r["run"], r["card"], r["round"]))
        for k, _ in MEASURES:
            r[k] = float(h[k]) if k in h else float(r[k])
        r["temp_first"] = float(h["temp_first"])
        r["wd_offsets_agree"] = (h["wd_mem_off"], h["wd_gpc_off"]) == (str(r["mem_mts"]), str(r["core_mhz"]))
        r["t_start"] = epoch(h["start"])
    if heat:
        raise SystemExit(f"{len(heat)} heat rows have no round in {a.rounds}")
    runs = list(dict.fromkeys(r["run"] for r in rows))
    os.makedirs(a.out, exist_ok=True)
    disagree = [r for r in rows if not r["wd_offsets_agree"]]

    strata, same = {}, {}
    for r in rows:
        strata.setdefault((r["card"], r["mem_mts"], r["core_mhz"]), []).append(r)
        same.setdefault((r["run"], r["card"], r["mem_mts"], r["core_mhz"]), []).append(r)
    last = {}
    for r in rows:
        k = (r["run"], r["card"])
        last[k] = max(last.get(k, 0), r["round"])
    errs = [r for r in rows if r["differs"]]
    # a soak that stopped at the first wrong answer: the answer is its card's last round there, and no card ran on
    stopped_runs = {r["run"] for r in errs if all(x["round"] <= r["round"] for x in rows if x["run"] == r["run"])}
    for r in errs:
        r["stopped"] = r["run"] in stopped_runs

    # 1. each wrong answer
    erows = []
    for r in errs:
        s = strata[(r["card"], r["mem_mts"], r["core_mhz"])]
        ss = same[(r["run"], r["card"], r["mem_mts"], r["core_mhz"])]
        tv = sorted(x["temp_mean"] for x in s)
        e = {"run": r["run"], "card": r["card"], "round": r["round"], "time": r["time"], "mem_mts": r["mem_mts"],
             "core_mhz": r["core_mhz"], "soak_stopped_at_it": "yes" if r["stopped"] else "no",
             "temp_mean": f"{r['temp_mean']:.2f}", "temp_max": f"{r['temp_max']:.0f}",
             "temp_first": f"{r['temp_first']:.0f}", "watts_mean": f"{r['watts_mean']:.2f}",
             "sm_mhz_mean": f"{r['sm_mhz_mean']:.0f}", "rig_temp_mean": f"{r['rig_temp_mean']:.2f}",
             "own_minutes": f"{r['own_minutes']:.2f}", "stratum_rounds": len(s),
             "stratum_runs": " ".join(dict.fromkeys(x["run"] for x in s)),
             "stratum_temp_lowest": f"{tv[0]:.2f}", "stratum_temp_median": f"{statistics.median(tv):.2f}",
             "stratum_temp_highest": f"{tv[-1]:.2f}",
             "stratum_rounds_cooler": sum(1 for v in tv if v < r["temp_mean"])}
        for k, _ in MEASURES:
            e[f"place_{k}"] = f"{midrank([x[k] for x in s], r[k]):.3f}"
        e["same_soak_rounds"] = len(ss)
        e["same_soak_place_temp_mean"] = f"{midrank([x['temp_mean'] for x in ss], r['temp_mean']):.3f}"
        e["same_soak_place_own_minutes"] = f"{midrank([x['own_minutes'] for x in ss], r['own_minutes']):.3f}"
        erows.append(e)
    write(a.out, "errors.csv", erows)

    # 2. average place of the wrong answers, against drawing each one's round at random from its stratum
    rng = random.Random(a.seed)
    tests = []
    sets = [("all 8", lambda r: True),
            ("the 4 whose soak went on (R201)", lambda r: not r["stopped"])]
    for set_name, keep in sets:
        sel = [r for r in errs if keep(r)]
        for scope, groups, key in (("the card at that value, every soak", strata,
                                    lambda r: (r["card"], r["mem_mts"], r["core_mhz"])),
                                   ("the card at that value, the same soak", same,
                                    lambda r: (r["run"], r["card"], r["mem_mts"], r["core_mhz"]))):
            by = {}
            for r in sel:
                by.setdefault(key(r), []).append(r)
            for m, desc in MEASURES:
                vals = {k: [x[m] for x in groups[k]] for k in by}
                ranks = {k: [midrank(vals[k], v) for v in vals[k]] for k in by}
                u_obs = sum(midrank(vals[k], x[m]) for k, es in by.items() for x in es)
                kk = len(sel)
                draws = []
                for _ in range(a.draws):
                    draws.append(sum(sum(rng.sample(ranks[k], len(es))) for k, es in by.items()) / kk)
                draws.sort()
                obs = u_obs / kk
                tests.append({"errors": set_name, "among": scope, "measure": desc, "n_errors": kk,
                              "mean_place": f"{obs:.3f}", "even": "0.500",
                              "chance_2.5pct": f"{draws[int(0.025 * a.draws)]:.3f}",
                              "chance_97.5pct": f"{draws[int(0.975 * a.draws) - 1]:.3f}",
                              "p_as_low_or_lower": f"{sum(1 for d in draws if d <= obs + 1e-12) / a.draws:.3f}",
                              "p_as_high_or_higher": f"{sum(1 for d in draws if d >= obs - 1e-12) / a.draws:.3f}"})
    write(a.out, "tests.csv", tests)

    # 3. how much heat would have shown: each wrong answer's round drawn with weight exp(b * temperature) within its
    #    stratum (each degree multiplies the chance by exp(b)); how often the 4 informative answers' average place
    #    by mean temperature reaches the top 5 % of the even spread
    sel = [r for r in errs if not r["stopped"]]
    by = {}
    for r in sel:
        by.setdefault((r["card"], r["mem_mts"], r["core_mhz"]), []).append(r)
    temps = {k: [x["temp_mean"] for x in strata[k]] for k in by}
    ranks = {k: [midrank(temps[k], v) for v in temps[k]] for k in by}
    null = sorted(sum(sum(rng.sample(ranks[k], len(es))) for k, es in by.items()) / len(sel)
                  for _ in range(a.draws))
    bar = null[int(0.95 * a.draws) - 1]
    power = []
    for mult in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        b = math.log(mult)
        cum = {}
        for k in by:
            top = max(temps[k])
            w = [math.exp(b * (t - top)) for t in temps[k]]
            c, acc = [], 0.0
            for x in w:
                acc += x
                c.append(acc)
            cum[k] = c
        hits = 0
        n_sim = max(1, a.draws // 5)
        for _ in range(n_sim):
            tot = 0.0
            for k, es in by.items():
                picked = set()
                while len(picked) < len(es):
                    picked.add(bisect.bisect_left(cum[k], rng.random() * cum[k][-1]))
                tot += sum(ranks[k][i] for i in picked)
            hits += tot / len(sel) > bar + 1e-12
        power.append({"chance_multiplier_per_degree": f"{mult:g}", "errors": len(sel),
                      "bar_mean_place": f"{bar:.3f}", "share_of_draws_over_the_bar": f"{hits / n_sim:.3f}"})
    write(a.out, "power.csv", power)

    # 4. the warm-up: temperature by minutes since each soak's first round began
    first = {run: min(r["t_start"] for r in rows if r["run"] == run) for run in runs}
    settled = {}
    for r in rows:
        if (r["t_start"] - first[r["run"]]) / 60 >= 15:
            settled.setdefault((r["run"], r["card"]), []).append(r["temp_mean"])
    settled = {k: statistics.median(v) for k, v in settled.items()}
    warm = []
    for run in runs + ["all"]:
        for lo, hi in WARMUP_BINS:
            b = [r for r in rows if (run == "all" or r["run"] == run)
                 and lo <= (r["t_start"] - first[r["run"]]) / 60 and (hi is None or (r["t_start"] - first[r["run"]]) / 60 < hi)]
            if not b:
                continue
            below = [settled[(r["run"], r["card"])] - r["temp_mean"] for r in b if (r["run"], r["card"]) in settled]
            warm.append({"run": run, "minutes_since_first_round": f"{lo}+" if hi is None else f"{lo}-{hi}",
                         "card_rounds": len(b), "temp_mean": f"{statistics.fmean(r['temp_mean'] for r in b):.2f}",
                         "below_settled": f"{statistics.fmean(below):.2f}" if below else "",
                         "wrong_answers": sum(r["differs"] for r in b)})
    write(a.out, "warmup.csv", warm)

    # 5. late and warm: how closely temperature follows minutes since the card's own change, within strata
    conf = []
    for r in errs:
        for scope, s in (("every soak", strata[(r["card"], r["mem_mts"], r["core_mhz"])]),
                         ("the same soak", same[(r["run"], r["card"], r["mem_mts"], r["core_mhz"])])):
            rho = spearman([x["own_minutes"] for x in s], [x["temp_mean"] for x in s])
            conf.append({"run": r["run"], "card": r["card"], "round": r["round"], "among": scope, "rounds": len(s),
                         "spearman_temp_vs_minutes": "" if rho is None else f"{rho:.3f}",
                         "temp_sd": f"{statistics.pstdev(x['temp_mean'] for x in s):.2f}"})
    big = [s for s in strata.values() if len(s) >= 20]
    rhos = [(len(s), spearman([x["own_minutes"] for x in s], [x["temp_mean"] for x in s])) for s in big]
    rhos = [(n, v) for n, v in rhos if v is not None]
    conf.append({"run": "every stratum of 20 or more rounds", "card": "", "round": "", "among": "every soak",
                 "rounds": sum(n for n, _ in rhos),
                 "spearman_temp_vs_minutes": f"{sum(n * v for n, v in rhos) / sum(n for n, _ in rhos):.3f}",
                 "temp_sd": ""})
    write(a.out, "confound.csv", conf)

    # 6. each card's settled heat (rounds from 15 minutes after the soak's first round) and its wrong answers
    cards = []
    for g in sorted({r["card"] for r in rows}):
        c = {"card": g}
        for run in runs:
            v = [r for r in rows if r["run"] == run and r["card"] == g and (r["t_start"] - first[run]) / 60 >= 15]
            c[f"{run}_temp"] = f"{statistics.median(r['temp_mean'] for r in v):.1f}" if v else ""
            c[f"{run}_watts"] = f"{statistics.median(r['watts_mean'] for r in v):.1f}" if v else ""
        c["wrong_answers"] = sum(r["differs"] for r in rows if r["card"] == g)
        cards.append(c)
    write(a.out, "cards.csv", cards)

    print(f"{len(rows)} card-rounds, {len(errs)} wrong answers, soaks that stopped at one: {sorted(stopped_runs)}")
    print(f"watchdog offsets agree with FN118's for {len(rows) - len(disagree)}; not for: "
          + "; ".join(f"{r['run']} card {r['card']} round {r['round']}" for r in disagree))
    for e in erows:
        print("  " + "; ".join(f"{k}={v}" for k, v in e.items()))
    print("\nTests:")
    for x in tests:
        print("  " + "; ".join(f"{k}={v}" for k, v in x.items()))
    print("\nPower:")
    for x in power:
        print("  " + "; ".join(f"{k}={v}" for k, v in x.items()))
    print("\nWarm-up:")
    for x in warm:
        print("  " + "; ".join(f"{k}={v}" for k, v in x.items()))
    print("\nConfound:")
    for x in conf:
        print("  " + "; ".join(f"{k}={v}" for k, v in x.items()))


def write(out, name, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--run", action="append", required=True, metavar="NAME=SOAK_CSV,WD_CSV_GZ")
    tb = sub.add_parser("table")
    tb.add_argument("rounds", help="FN118's tables/rounds.csv.gz")
    tb.add_argument("heat", help="heat.csv.gz from build")
    tb.add_argument("out")
    tb.add_argument("--draws", type=int, default=100000)
    tb.add_argument("--seed", type=int, default=119)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    else:
        table(a)


if __name__ == "__main__":
    sys.exit(main())
