#!/usr/bin/env python3
"""C1128 (Q1861): the tables from run/run.sh's output (verify-drift's <arm>/drift.rows.tsv per arm).

Usage: python3 -I summarize.py <copy of WORK> <output folder>
Writes into the output folder:
  summary.tsv     per test arm, reading (reply rows 1-256, the outputs of the calls after the prompt; reply row 0, the
                  prompt's last position; whole, every position) and request (and all three pooled): positions, the
                  same bits, the same top token, the KL divergence (mean, median, 99th percentile, largest) and the
                  |logit gap move| and |probability gap move| of the reference's two most likely tokens (median, 90th
                  and 99th percentiles, largest)
  flips.tsv       per test arm, the rows where C964 flipped on the P106 (code 124, prose 67, deep 21), read here:
                  the reference's two candidates, both arms' probabilities, the moves, the KL divergence and each
                  move's percentile rank among the arm's pooled reply rows 1-256; with C964's P106 moves
  rule.tsv        the reading rule (README) applied to the four-token arm
  contested.tsv   the four-token arm's reply rows 1-256 where the reference's top token is below 0.6 or 0.5 (where a
                  near tie can sit): the moves' median and percentiles and C964's moves' percentile ranks among them;
                  and the near ties (the reference's two candidates closer than a gap) and how many changed top token
  depth.tsv       the four-token arms' whole reading of the deep request by depth (prompt positions, then the reply)
  offsets.tsv     the four-token arms' reply rows 1-256 by place in their four-token call (0 = the call's first token)
  reference.tsv   per request, the reference load against the P106's plain reply: positions whose top token is the
                  reply's token, the first that is not (rows 0-255 of the reply reading)
  reply_rows.tsv  every reply row of every test arm: the reference's candidates and the per-position measures
Prints the rule's verdict.
"""
import bisect
import csv
import math
import os
import sys

WORK, OUT = sys.argv[1], sys.argv[2]
ARMS = ["four", "floor", "one_rs3", "four_rs0", "four_exact"]
REQS = ["code", "prose", "deep"]
FLIP_ROW = {"code": 124, "prose": 67, "deep": 21}
C964_GAP = {"code": -0.0219, "prose": -0.0849, "deep": -0.0190}     # probability gap moves on the P106 (C964)
C964_LOGIT = {"code": -0.0467, "prose": -0.2459, "deep": -0.0943}   # logit gap moves, from C964's probabilities


def read(arm):
    path = os.path.join(WORK, arm, "drift.rows.tsv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["req"] = REQS[int(r["req"])]
        r["row"] = int(r["row"])
    return rows


def q(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def rank(sorted_vals, x):
    """percentile rank: the share of values strictly below x, plus half the ties"""
    lo, hi = bisect.bisect_left(sorted_vals, x), bisect.bisect_right(sorted_vals, x)
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(sorted_vals)


def fnum(x, d=6):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}g}" if isinstance(x, float) else str(x))


def select(rows, reading, req=None):
    out = []
    for r in rows:
        if reading == "reply" and not (r["part"] == "reply" and r["row"] >= 1):
            continue
        if reading == "reply0" and not (r["part"] == "reply" and r["row"] == 0):
            continue
        if reading == "whole" and r["part"] != "whole":
            continue
        if req and r["req"] != req:
            continue
        out.append(r)
    return out


def stats(sel):
    kld = sorted(float(r["kld"]) for r in sel)
    pair = sorted(abs(float(r["pairshift"])) for r in sel)
    gap = sorted(abs(float(r["gapshift"])) for r in sel)
    n = len(sel)
    return {
        "positions": n,
        "same_bits": sum(r["samebits"] == "1" for r in sel),
        "same_top": sum(r["sametop"] == "1" for r in sel),
        "kld_mean": sum(kld) / n if n else float("nan"),
        "kld_median": q(kld, 0.5), "kld_p99": q(kld, 0.99), "kld_max": kld[-1] if n else float("nan"),
        "logit_move_median": q(pair, 0.5), "logit_move_p90": q(pair, 0.9), "logit_move_p99": q(pair, 0.99),
        "logit_move_max": pair[-1] if n else float("nan"),
        "prob_move_median": q(gap, 0.5), "prob_move_p90": q(gap, 0.9), "prob_move_p99": q(gap, 0.99),
        "prob_move_max": gap[-1] if n else float("nan"),
    }


os.makedirs(OUT, exist_ok=True)
base = read("base")
data = {a: read(a) for a in ARMS}
data = {a: v for a, v in data.items() if v is not None}

# summary
keys = None
with open(os.path.join(OUT, "summary.tsv"), "w") as f:
    for a, rows in data.items():
        for reading in ("reply", "reply0", "whole"):
            for req in REQS + [None]:
                sel = select(rows, reading, req)
                if not sel:
                    continue
                s = stats(sel)
                if keys is None:
                    keys = list(s)
                    f.write("arm\treading\trequest\t" + "\t".join(keys) + "\n")
                f.write(f"{a}\t{reading}\t{req or 'all'}\t" + "\t".join(fnum(s[k]) for k in keys) + "\n")

# flips
with open(os.path.join(OUT, "flips.tsv"), "w") as f:
    f.write("arm\trequest\trow\tref_a\tref_b\tpref_a\tpref_b\tparm_a\tparm_b\tprob_move\tlogit_move\tkld\tsame_top\t"
            "prob_move_rank\tlogit_move_rank\tkld_rank\tc964_prob_move\tc964_logit_move\n")
    for a, rows in data.items():
        pool = select(rows, "reply")
        pair = sorted(abs(float(r["pairshift"])) for r in pool)
        gap = sorted(abs(float(r["gapshift"])) for r in pool)
        kld = sorted(float(r["kld"]) for r in pool)
        bl = {(r["req"], r["row"]): r for r in base if r["part"] == "reply"}
        for req in REQS:
            r = next(x for x in pool if x["req"] == req and x["row"] == FLIP_ROW[req])
            b = bl[(req, r["row"])]
            f.write("\t".join([a, req, str(r["row"]), b["top1"], b["top2"], r["pref_a"], r["pref_b"], r["ptest_a"],
                               r["ptest_b"], fnum(float(r["gapshift"])), fnum(float(r["pairshift"])), fnum(float(r["kld"])),
                               r["sametop"], f"{rank(gap, abs(float(r['gapshift']))):.1f}",
                               f"{rank(pair, abs(float(r['pairshift']))):.1f}", f"{rank(kld, float(r['kld'])):.1f}",
                               str(C964_GAP[req]), str(C964_LOGIT[req])]) + "\n")

# offsets in the four-token call
with open(os.path.join(OUT, "offsets.tsv"), "w") as f:
    f.write("arm\tplace_in_call\tpositions\tsame_bits\tsame_top\tkld_mean\tlogit_move_median\tlogit_move_p99\n")
    for a in ("four", "four_rs0", "four_exact"):
        if a not in data:
            continue
        pool = select(data[a], "reply")
        for o in range(4):
            sel = [r for r in pool if (r["row"] - 1) % 4 == o]
            s = stats(sel)
            f.write(f"{a}\t{o}\t{s['positions']}\t{s['same_bits']}\t{s['same_top']}\t{fnum(s['kld_mean'])}\t"
                    f"{fnum(s['logit_move_median'])}\t{fnum(s['logit_move_p99'])}\n")

# the whole reading of the deep request by depth (the prompt's positions, then the reply's)
with open(os.path.join(OUT, "depth.tsv"), "w") as f:
    f.write("arm\tfirst_pos\tlast_pos\tpositions\tsame_bits\tsame_top\tkld_mean\tkld_median\tkld_p99\tkld_max\t"
            "logit_move_median\tlogit_move_p99\n")
    n_prompt = 8189
    edges = [0, 256, 1024, 2048, 4096, 6144, n_prompt, n_prompt + 256]
    for a in ("four", "four_rs0", "four_exact"):
        if a not in data:
            continue
        deep = [r for r in data[a] if r["part"] == "whole" and r["req"] == "deep"]
        for lo, hi in zip(edges, edges[1:]):
            s = stats([r for r in deep if lo <= r["row"] < hi])
            f.write(f"{a}\t{lo}\t{hi - 1}\t{s['positions']}\t{s['same_bits']}\t{s['same_top']}\t{fnum(s['kld_mean'])}\t"
                    f"{fnum(s['kld_median'])}\t{fnum(s['kld_p99'])}\t{fnum(s['kld_max'])}\t{fnum(s['logit_move_median'])}\t"
                    f"{fnum(s['logit_move_p99'])}\n")

# the reference against the P106's plain reply
with open(os.path.join(OUT, "reference.tsv"), "w") as f:
    f.write("request\tpositions\ttop_is_reply_token\tfirst_row_not\n")
    for req in REQS:
        sel = [r for r in base if r["part"] == "reply" and r["req"] == req and int(r["next"]) >= 0]
        agree = [r["top1"] == r["next"] for r in sel]
        first = next((r["row"] for r, ok in zip(sel, agree) if not ok), "")
        f.write(f"{req}\t{len(sel)}\t{sum(agree)}\t{first}\n")

# every reply row
with open(os.path.join(OUT, "reply_rows.tsv"), "w") as f:
    f.write("arm\trequest\trow\tnext\tref_a\tref_b\tpref_a\tpref_b\tkld\tsame_top\tsame_bits\tprob_move\tlogit_move\n")
    bl = {(r["req"], r["row"]): r for r in base if r["part"] == "reply"}
    for a, rows in data.items():
        for r in rows:
            if r["part"] != "reply":
                continue
            b = bl[(r["req"], r["row"])]
            f.write("\t".join([a, r["req"], str(r["row"]), r["next"], b["top1"], b["top2"], r["pref_a"], r["pref_b"],
                               r["kld"], r["sametop"], r["samebits"], r["gapshift"], r["pairshift"]]) + "\n")

# the rule
four = select(data["four"], "reply")
floor = select(data["floor"], "reply") if "floor" in data else []
pair = sorted(abs(float(r["pairshift"])) for r in four)
fpair = sorted(abs(float(r["pairshift"])) for r in floor)
med, p99 = q(pair, 0.5), q(pair, 0.99)
fp99 = q(fpair, 0.99) if fpair else float("nan")
same = sum(r["samebits"] == "1" for r in four) / len(four)
flips = sorted(abs(v) for v in C964_LOGIT.values())
if not (fp99 < med / 10):
    verdict = "not resolved: the floor's 99th percentile is not below a tenth of the four-token arm's median"
    if same >= 0.99:
        verdict = "exact: the four-token pass gives the same bits at %.1f %% of the positions" % (100 * same)
elif same >= 0.99:
    verdict = "exact: the four-token pass gives the same bits at %.1f %% of the positions" % (100 * same)
elif med >= flips[0] / 3:
    verdict = "rounding everywhere: the median position moves by at least a third of the smallest flip's move"
elif med < flips[0] / 10 and all(v > p99 for v in flips):
    verdict = "a tail: the median is below a tenth of the smallest flip's move and all three flips lie above the 99th percentile"
else:
    verdict = "between: " + ", ".join(f"{k} at percentile {rank(pair, abs(v)):.1f}" for k, v in C964_LOGIT.items())
with open(os.path.join(OUT, "rule.tsv"), "w") as f:
    f.write("quantity\tvalue\n")
    for k, v in [("four: positions", len(four)), ("four: same bits share", same), ("four: median |logit move|", med),
                 ("four: 99th percentile |logit move|", p99), ("floor: 99th percentile |logit move|", fp99),
                 ("smallest C964 flip |logit move|", flips[0]), ("third of it", flips[0] / 3), ("tenth of it", flips[0] / 10)] + \
                [(f"C964 {k} flip: percentile rank among four's moves", rank(pair, abs(v))) for k, v in C964_LOGIT.items()] + \
                [("verdict", verdict)]:
        f.write(f"{k}\t{fnum(v) if not isinstance(v, str) else v}\n")
print(verdict)

# contested positions and near ties (reported, not judged)
with open(os.path.join(OUT, "contested.tsv"), "w") as f:
    f.write("selection\tpositions\tsame_top\tprob_move_median\tprob_move_p90\tprob_move_p99\tlogit_move_median\t"
            "logit_move_p99\tc964_code_prob_rank\tc964_prose_prob_rank\tc964_deep_prob_rank\tc964_code_logit_rank\t"
            "c964_prose_logit_rank\tc964_deep_logit_rank\n")
    sels = [(f"all reply rows 1-256", four)]
    sels += [(f"reference top below {lim}", [r for r in four if float(r["pref_a"]) < lim]) for lim in (0.6, 0.5)]
    sels += [(f"reference gap below {g}", [r for r in four if float(r["pref_a"]) - float(r["pref_b"]) < g])
             for g in (0.03, 0.0849)]
    for name, sel in sels:
        s = stats(sel)
        g = sorted(abs(float(r["gapshift"])) for r in sel)
        p = sorted(abs(float(r["pairshift"])) for r in sel)
        f.write(f"{name}\t{s['positions']}\t{s['same_top']}\t{fnum(s['prob_move_median'])}\t{fnum(s['prob_move_p90'])}\t"
                f"{fnum(s['prob_move_p99'])}\t{fnum(s['logit_move_median'])}\t{fnum(s['logit_move_p99'])}\t" +
                "\t".join(f"{rank(g, abs(C964_GAP[k])):.1f}" for k in REQS) + "\t" +
                "\t".join(f"{rank(p, abs(C964_LOGIT[k])):.1f}" for k in REQS) + "\n")
