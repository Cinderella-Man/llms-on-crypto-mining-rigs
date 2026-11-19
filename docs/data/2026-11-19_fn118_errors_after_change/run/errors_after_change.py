#!/usr/bin/env python3
"""Do a card's wrong answers under a clock offset come soon after an offset change? (FN118, Q1596)

Two steps:

  build  reads the soak kit's work folders (bench/mem_offset: WORK/out/climb.log, WORK/soak/soak.log,
         WORK/soak/soak.csv) and writes, for every card and round of every soak, its offsets, whether its check
         differed from stock, and the rounds and minutes since that card's own last offset change and since the last
         change of any card on the rig:
           OUT/changes.csv     every offset change: run, card, time, memory, core, what set it
           OUT/rounds.csv.gz   every card's every soak round with those columns
  table  reads rounds.csv.gz and writes the tables and the tests:
           OUT/buckets.csv     card-rounds and differing checks by rounds and by minutes since a change, per run
           OUT/errors.csv      each differing check: its stratum (card at a value), how many rounds that stratum ran
                               and how many of them were as early as the error
           OUT/tests.csv       the chance of the observed clustering if errors were spread evenly (see below)

  python3 -I errors_after_change.py build --out tables \
      --r200-offsets R200_TABLES/soak.csv --r200 try1=W1 --r200 try2=W2 ... --kit r201=W5 --kit r201b=W6 ...
  python3 -I errors_after_change.py table tables/rounds.csv.gz tables

R200's work folders hold soak.csv without offset columns (the soak kit before the ladder); their offsets come from
R200's published tables/soak.csv by try and card. Every R200 try starts at `apply` on a fresh boot.

The test. A stratum is one card at one memory and core offset, pooled over every run and boot in which it ran. If a
card's chance of a wrong answer at a value were the same in every round, the rounds holding its k wrong answers would
be k rounds drawn at random from the n rounds it ran at that value. Two statistics, each for both clocks (the card's
own change, any card's change):
  early count   wrong answers within the first EARLY rounds (or minutes) after the change; its exact distribution
                is the sum of one hypergeometric per stratum. A stratum whose every round was early (a soak that
                stopped at the error) tells nothing and is listed but cannot move the result.
  mean rank     each wrong answer's place among its stratum's rounds ordered by time since the change (0 = the
                earliest, 1 = the latest, ties shared), averaged; its distribution by drawing the k rounds at random
                DRAWS times (seed SEED). No bucket to choose.
  rate ratio    how many times likelier a wrong answer is in an early round than in a later one, by maximum
                likelihood over the strata, with the interval where the likelihood stays within 1.92 of its top
                (95 %).
Soaks that stopped at the first wrong answer end every one of their strata at that answer; with errors this rare
(fewer than 1 in 25 rounds) drawing the rounds evenly is a close stand-in for a constant chance a round.
"""
import argparse
import csv
import gzip
import io
import math
import os
import random
import re
import sys
from datetime import datetime

TFMT = "%Y-%m-%d %H:%M:%S"
APPLY_RE = re.compile(r"^(\S+ \S+) apply: card (\d+) set mem \+(-?\d+) MT/s, core \+(-?\d+) MHz")
RAISE_RE = re.compile(r"^(\S+ \S+) rung (\d+): raised")
ROUND_BUCKETS = [(1, 5), (6, 20), (21, 60), (61, None)]
MINUTE_BUCKETS = [(0, 2), (2, 5), (5, 15), (15, None)]


def t(s):
    return datetime.strptime(s, TFMT)


def read_csv(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def applies(work):
    """The last `apply` in WORK/out/climb.log: {card: (time, mem, core)}."""
    out = {}
    with open(os.path.join(work, "out", "climb.log"), errors="replace") as f:
        for ln in f:
            m = APPLY_RE.match(ln)
            if m:
                out[int(m.group(2))] = (m.group(1), int(m.group(3)), int(m.group(4)))
    if not out:
        raise SystemExit(f"{work}: no 'apply: card N set' line in out/climb.log")
    return out


def raises(work):
    """{rung: time} from WORK/soak/soak.log's 'rung K: raised' lines (a ladder; none for a plain soak)."""
    out = {}
    path = os.path.join(work, "soak", "soak.log")
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            for ln in f:
                m = RAISE_RE.match(ln)
                if m:
                    out[int(m.group(2))] = m.group(1)
    return out


def build(a):
    offs = {}
    if a.r200:
        for r in read_csv(a.r200_offsets):
            offs[(r["run"], int(r["card"]))] = (int(r["mem_mts"]), int(r["core_mhz"]))
    runs = [(spec, True) for spec in a.r200] + [(spec, False) for spec in a.kit]
    changes, rounds = [], []
    for spec, is_r200 in runs:
        name, work = spec.split("=", 1)
        rows = read_csv(os.path.join(work, "soak", "soak.csv"))
        for r in rows:
            r["card"], r["round"] = int(r["card"]), int(r["round"])
            if is_r200:
                r["rung"] = 1
                r["mem_mts"], r["core_mhz"] = offs[(name, r["card"])]
            else:
                r["rung"], r["mem_mts"], r["core_mhz"] = int(r["rung"]), int(r["mem_mts"]), int(r["core_mhz"])
        cards = sorted({r["card"] for r in rows})
        ch = []   # (time, card, mem, core, event)
        for g, (tm, mem, core) in sorted(applies(work).items()):
            if g in cards:
                ch.append((tm, g, mem, core, "apply"))
        first = {}
        for r in rows:
            first.setdefault((r["rung"], r["card"]), (r["mem_mts"], r["core_mhz"]))
        for g in cards:
            v = first.get((1, g))
            a_v = [c for c in ch if c[1] == g]
            if v and a_v and (a_v[0][2], a_v[0][3]) != v:
                raise SystemExit(f"{name} card {g}: apply set {a_v[0][2:4]}, rung 1 ran at {v}")
        for k, tm in sorted(raises(work).items()):
            for g in cards:
                prev, cur = first.get((k - 1, g)), first.get((k, g))
                if prev and cur and cur != prev:
                    ch.append((tm, g, cur[0], cur[1], f"rung {k} raise"))
        ch.sort()
        for tm, g, mem, core, ev in ch:
            changes.append({"run": name, "card": g, "time": tm, "mem_mts": mem, "core_mhz": core, "event": ev})
        rig_times = sorted({c[0] for c in ch})
        own_times = {g: sorted(c[0] for c in ch if c[1] == g) for g in cards}
        count_own, count_rig, last_own, last_rig = {}, {}, {}, {}
        for r in sorted(rows, key=lambda r: (r["round"], r["card"])):
            g, tm = r["card"], r["time"]
            own = max(x for x in own_times[g] if x <= tm)
            rig = max(x for x in rig_times if x <= tm)
            count_own[g] = count_own[g] + 1 if last_own.get(g) == own else 1
            count_rig[g] = count_rig[g] + 1 if last_rig.get(g) == rig else 1
            last_own[g], last_rig[g] = own, rig
            rounds.append({
                "run": name, "card": g, "round": r["round"], "time": tm, "rung": r["rung"],
                "mem_mts": r["mem_mts"], "core_mhz": r["core_mhz"], "differs": 0 if r["verdict"] == "PASS" else 1,
                "own_change": own, "own_rounds": count_own[g],
                "own_minutes": f"{(t(tm) - t(own)).total_seconds() / 60:.2f}",
                "rig_change": rig, "rig_rounds": count_rig[g],
                "rig_minutes": f"{(t(tm) - t(rig)).total_seconds() / 60:.2f}"})
        bad = [r for r in rows if r["verdict"] != "PASS" and r.get("reason") != "differs from stock"]
        if bad:
            raise SystemExit(f"{name}: a check failed for another reason than a difference: {bad[0]}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "changes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(changes[0]))
        w.writeheader()
        w.writerows(changes)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rounds[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rounds)
    with gzip.GzipFile(os.path.join(a.out, "rounds.csv.gz"), "wb", mtime=0) as f:
        f.write(buf.getvalue().encode())
    print(f"{len(changes)} offset changes, {len(rounds)} card-rounds, "
          f"{sum(r['differs'] for r in rounds)} differing checks -> {a.out}")


def label(lo, hi, unit):
    if unit == "rounds":
        return f"{lo}+" if hi is None else f"{lo}-{hi}"
    return f"{lo}+ min" if hi is None else f"{lo}-{hi} min"


def in_bucket(x, lo, hi, unit):
    if unit == "rounds":
        return x >= lo and (hi is None or x <= hi)
    return x >= lo and (hi is None or x < hi)


def hyper_pmf(n, e, k):
    """P(j of k draws without replacement from n land among the e early ones), j = 0..k."""
    tot = math.comb(n, k)
    return [math.comb(e, j) * math.comb(n - e, k - j) / tot for j in range(k + 1)]


def convolve(pmfs):
    out = [1.0]
    for p in pmfs:
        nxt = [0.0] * (len(out) + len(p) - 1)
        for i, x in enumerate(out):
            for j, y in enumerate(p):
                nxt[i + j] += x * y
        out = nxt
    return out


def midrank(values, x):
    lt = sum(1 for v in values if v < x)
    eq = sum(1 for v in values if v == x)
    return (lt + (eq - 1) / 2) / (len(values) - 1) if len(values) > 1 else 0.5


def rate_ratio(errs):
    """Errors as (f, early): f = the stratum's share of early rounds. If an early round is th times as likely to hold
    a wrong answer as a later one, an error is early with chance th f / (th f + 1 - f). Returns the most likely th
    and the exact 95 % interval: every th under which the observed early count is not in either 2.5 % tail."""
    info = [(f, e) for f, e in errs if 0 < f < 1]
    if not info:
        return None, None, None
    y = sum(1 for _, e in info if e)

    def dist(th):
        return convolve([[1 - th * f / (th * f + 1 - f), th * f / (th * f + 1 - f)] for f, _ in info])

    def ll(th):
        return sum(math.log(th * f / (th * f + 1 - f)) if e else math.log((1 - f) / (th * f + 1 - f))
                   for f, e in info)
    grid = [10 ** (i / 200) for i in range(-800, 801)]   # 1e-4 .. 1e4
    vals = [ll(th) for th in grid]
    best = grid[vals.index(max(vals))]
    inside = [th for th in grid if sum(dist(th)[:y + 1]) > 0.025 and sum(dist(th)[y:]) > 0.025]
    lo = 0.0 if y == 0 else inside[0]
    hi = float("inf") if y == len(info) else inside[-1]
    return (0.0 if y == 0 else best), lo, hi


def table(a):
    rows = read_csv(a.rounds)
    for r in rows:
        for k in ("card", "round", "rung", "mem_mts", "core_mhz", "differs", "own_rounds", "rig_rounds"):
            r[k] = int(r[k])
        r["own_minutes"], r["rig_minutes"] = float(r["own_minutes"]), float(r["rig_minutes"])
    runs = list(dict.fromkeys(r["run"] for r in rows))
    os.makedirs(a.out, exist_ok=True)

    # 1. buckets: per run and in total, by rounds and by minutes, since the card's own change and any card's change
    out = []
    for clock in ("own", "rig"):
        for unit, buckets in (("rounds", ROUND_BUCKETS), ("minutes", MINUTE_BUCKETS)):
            col = f"{clock}_{unit}"
            for run in runs + ["all"]:
                sel = [r for r in rows if run == "all" or r["run"] == run]
                for lo, hi in buckets:
                    b = [r for r in sel if in_bucket(r[col], lo, hi, unit)]
                    out.append({"since": "the card's own change" if clock == "own" else "any card's change",
                                "unit": unit, "run": run, "bucket": label(lo, hi, unit), "card_rounds": len(b),
                                "differing": sum(r["differs"] for r in b)})
    with open(os.path.join(a.out, "buckets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # 2. strata: one card at one value, pooled over runs
    strata = {}
    for r in rows:
        strata.setdefault((r["card"], r["mem_mts"], r["core_mhz"]), []).append(r)
    errs = [r for r in rows if r["differs"]]
    erows = []
    for r in errs:
        s = strata[(r["card"], r["mem_mts"], r["core_mhz"])]
        e = {"run": r["run"], "card": r["card"], "round": r["round"], "time": r["time"],
             "mem_mts": r["mem_mts"], "core_mhz": r["core_mhz"],
             "own_rounds": r["own_rounds"], "own_minutes": f"{r['own_minutes']:.2f}",
             "rig_rounds": r["rig_rounds"], "rig_minutes": f"{r['rig_minutes']:.2f}",
             "stratum_rounds": len(s), "stratum_errors": sum(x["differs"] for x in s),
             "stratum_runs": " ".join(dict.fromkeys(x["run"] for x in s))}
        for clock in ("own", "rig"):
            e[f"{clock}_early_rounds"] = sum(1 for x in s if x[f"{clock}_rounds"] <= a.early)
            e[f"{clock}_rank"] = f"{midrank([x[f'{clock}_minutes'] for x in s], r[f'{clock}_minutes']):.3f}"
        erows.append(e)
    with open(os.path.join(a.out, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0]))
        w.writeheader()
        w.writerows(erows)

    # 3. tests
    tests = []
    rng = random.Random(a.seed)

    def run_tests(name, keep):
        """keep(r) -> True for the errors that count in this variant; strata as above."""
        sel = [r for r in errs if keep(r)]
        by = {}
        for r in sel:
            by.setdefault((r["card"], r["mem_mts"], r["core_mhz"]), []).append(r)
        for clock in ("own", "rig"):
            for unit, early in (("rounds", a.early), ("minutes", a.early_minutes)):
                col = f"{clock}_{unit}"
                pmfs, obs, exp, rr = [], 0, 0.0, []
                for key, es in by.items():
                    s = strata[key]
                    n = len(s)
                    ne = sum(1 for x in s if x[col] <= early)
                    k = len(es)
                    pmfs.append(hyper_pmf(n, ne, k))
                    obs += sum(1 for x in es if x[col] <= early)
                    exp += k * ne / n
                    rr += [(ne / n, x[col] <= early) for x in es]
                dist = convolve(pmfs)
                p_ge = sum(dist[obs:])
                p_le = sum(dist[:obs + 1])
                th, lo, hi = rate_ratio(rr)
                # mean rank, by minutes since the change (rounds tie too often), Monte Carlo
                svals = {key: [x[f"{clock}_minutes"] for x in strata[key]] for key in by}
                ranks = {key: sorted(midrank(svals[key], v) for v in svals[key]) for key in by}
                u_obs = sum(midrank(svals[key], x[f"{clock}_minutes"]) for key, es in by.items() for x in es)
                kk = sum(len(es) for es in by.values())
                le = ge = 0
                for _ in range(a.draws):
                    u = sum(sum(rng.sample(ranks[key], len(es))) for key, es in by.items())
                    le += u <= u_obs + 1e-12
                    ge += u >= u_obs - 1e-12
                tests.append({"errors": name, "since": "the card's own change" if clock == "own" else
                              "any card's change", "early": f"within {early} {unit}", "n_errors": kk,
                              "informative": sum(1 for f, _ in rr if 0 < f < 1),
                              "observed_early": obs, "expected_early": f"{exp:.2f}",
                              "p_at_least_observed": f"{p_ge:.3f}", "p_at_most_observed": f"{p_le:.3f}",
                              "rate_ratio": "" if th is None else f"{th:.2f}",
                              "rate_ratio_95_low": "" if lo is None else f"{lo:.3g}",
                              "rate_ratio_95_high": "" if hi is None else f"{hi:.3g}",
                              "mean_rank": f"{u_obs / kk:.3f}", "mean_rank_expected": "0.500",
                              "p_mean_rank_as_early": f"{le / a.draws:.3f}",
                              "p_mean_rank_as_late": f"{ge / a.draws:.3f}"})

    run_tests("all", lambda r: True)
    run_tests("without R201's cards 2 and 6 (the two that raised the question)",
              lambda r: not (r["run"] == "r201" and r["card"] in (2, 6)))
    with open(os.path.join(a.out, "tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0]))
        w.writeheader()
        w.writerows(tests)

    # 4. what the clean soaks add: early card-rounds after a card's own change, at values that never differed
    clean_keys = {k for k, s in strata.items() if not any(x["differs"] for x in s)}
    early_clean = [r for r in rows if (r["card"], r["mem_mts"], r["core_mhz"]) in clean_keys and r["own_rounds"] <= a.early]
    changes_clean = {(r["run"], r["card"], r["own_change"]) for r in early_clean}
    bound_clean = 1 - 0.05 ** (1 / len(early_clean))   # 95 % upper bound on a round's chance, 0 errors seen

    # 5. how long a soak must run: each value that erred, its rate and what a rung of RUNG rounds sees at that rate
    srows = []
    for (g, mem, core), s in sorted(strata.items()):
        k = sum(x["differs"] for x in s)
        if not k:
            continue
        per_run = {}
        for x in s:
            per_run.setdefault(x["run"], []).append(x)
        rate = k / len(s)
        srows.append({"card": g, "mem_mts": mem, "core_mhz": core, "rounds": len(s), "differing": k,
                      "runs": " ".join(per_run), "rate_one_in": f"{len(s) / k:.0f}",
                      "only_soaks_that_stopped_at_the_error": "yes" if all(r.startswith("try") for r in per_run)
                      else "no",
                      f"chance_a_{a.rung}_round_rung_is_clean": f"{(1 - rate) ** a.rung:.3f}",
                      "rounds_for_95_pct_to_see_one": math.ceil(math.log(0.05) / math.log(1 - rate))})
    with open(os.path.join(a.out, "strata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    # print a summary for the README
    print(f"{len(rows)} card-rounds in {len(runs)} runs, {len(errs)} differing checks, {len(strata)} strata "
          f"(a card at a value), {len(clean_keys)} of them never differed")
    print("\nEach differing check (rounds / minutes since the card's own change; since any card's change):")
    for e in erows:
        print(f"  {e['run']} card {e['card']} round {e['round']} at +{e['mem_mts']}/+{e['core_mhz']}: own "
              f"{e['own_rounds']} / {e['own_minutes']}, rig {e['rig_rounds']} / {e['rig_minutes']}; stratum "
              f"{e['stratum_rounds']} rounds ({e['stratum_runs']}), {e['stratum_errors']} errors, early (<= {a.early}) "
              f"own {e['own_early_rounds']}, rig {e['rig_early_rounds']}")
    print("\nTotals by bucket (card-rounds, differing):")
    for b in out:
        if b["run"] == "all":
            print(f"  since {b['since']}, {b['unit']} {b['bucket']}: {b['card_rounds']}, {b['differing']}")
    print("\nTests:")
    for x in tests:
        print("  " + "; ".join(f"{k}={v}" for k, v in x.items()))
    print(f"\nClean strata: {len(early_clean)} card-rounds within {a.early} rounds of the card's own change, after "
          f"{len(changes_clean)} changes, none differed: a round's chance under 1 in {1 / bound_clean:.0f} (95 %)")
    print(f"\nThe values that erred (a rung of {a.rung} rounds):")
    for s in srows:
        print("  " + "; ".join(f"{k}={v}" for k, v in s.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--r200", action="append", default=[], metavar="TRY=WORK")
    b.add_argument("--r200-offsets", metavar="R200 tables/soak.csv")
    b.add_argument("--kit", action="append", default=[], metavar="NAME=WORK")
    tb = sub.add_parser("table")
    tb.add_argument("rounds")
    tb.add_argument("out")
    tb.add_argument("--early", type=int, default=20, help="rounds that count as early (default 20)")
    tb.add_argument("--early-minutes", type=float, default=5.0, help="minutes that count as early (default 5)")
    tb.add_argument("--draws", type=int, default=100000)
    tb.add_argument("--seed", type=int, default=118)
    tb.add_argument("--rung", type=int, default=152, help="rounds in a 30-minute rung (R201b's soak: 152)")
    a = ap.parse_args()
    if a.cmd == "build":
        if a.r200 and not a.r200_offsets:
            ap.error("--r200 needs --r200-offsets")
        build(a)
    else:
        table(a)


if __name__ == "__main__":
    sys.exit(main())
