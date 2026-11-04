#!/usr/bin/env python3
"""Check and summarize llama-server's LLAMA_SPEC_ADAPT_LOG files (Miner Llama, Q687).

A log holds a header line per server start ({"hdr": 1, ...}: the switch, n_max, the verify widths allowed, the cost
table T(k) for k = 1 .. n_max + 1, the draft cost, the S mode and its settings) and one line per drafted step: the
drafted count n_draft, what the rule keeps (n_rule), what was verified (n_keep), the drafter's probability of each
drafted token given the drafted prefix (p), the S the rule used (s), E(n_rule), C(n_rule), the accepted count
(n_acc) and the running speed-up after the step (s_next).

  adapt_log.py [--check] [--csv FILE] LOG [LOG ...]

--check recomputes every step from its header: the rule's choice from p and s (exactly as common_spec_adapt_choose
does it), S for the "run" mode by replaying the running speed-up over the earlier steps of the same slot and run,
n_keep (n_rule with the switch on, n_draft off), n_acc <= n_keep, and that the verify width is an allowed one when
the switch is on. It prints each problem and exits 1 if there is any.

The summary is one row per run: steps, steps by verify width, drafted, verified and accepted tokens per step, the
acceptance, the modelled step cost and speed-up over one-token decoding (from the cost table), and the rule's
expected tokens per step against the realised ones. For a log written with the switch off it also replays the rule
on the same drafts: a step that kept n_rule of the drafted tokens would have accepted min(n_acc, n_rule) of them,
because acceptance stops at the first rejected token; the next draft would start elsewhere, so this is an estimate.

Standard library only.
"""

import argparse
import csv
import json
import struct
import sys
from collections import Counter, defaultdict


def cost(hdr, k):
    t = hdr["cost_ms"]
    if k <= len(t):
        return t[k - 1]
    slope = t[-1] - t[-2] if len(t) >= 2 else t[0]
    return t[-1] + (k - len(t)) * slope


def lengths(hdr):
    return sorted(set(w - 1 for w in hdr["widths"]) | {0})


def choose(hdr, p, s_run):
    """common_spec_adapt_choose: returns (n_keep, s, e_tau, cost_ms, ties), ties = tests decided within rounding."""
    t1 = cost(hdr, 1)
    mode = hdr["s_mode"]

    def s_of(e, c):
        if mode == "step":
            return e * t1 / c
        if mode == "run":
            return s_run
        return hdr["s_fixed"]

    n_keep, e, c = 0, 1.0, hdr["draft_ms"] + t1
    s = s_of(e, c)
    path = 1.0
    ties = 0
    for ln in lengths(hdr):
        if ln <= n_keep:
            continue
        if ln > len(p):
            break
        gain, pp = 0.0, path
        for i in range(n_keep, ln):
            pp *= max(0.0, min(1.0, p[i]))
            gain += pp
        dt = cost(hdr, ln + 1) - cost(hdr, n_keep + 1)
        thr = hdr["scale"] * s * dt / t1
        if abs(gain - thr) <= 1e-9 * max(1.0, abs(thr)):
            ties += 1
        if not gain > thr:
            break
        n_keep, e, c, path = ln, e + gain, c + dt, pp
        s = s_of(e, c)
    return n_keep, s, e, c, ties


class RunState:
    """common_spec_adapt_state: exponential averages of a step's tokens and modelled cost."""

    def __init__(self, hdr):
        self.hdr = hdr
        self.tok = hdr["s0"]
        self.cost = cost(hdr, 1)

    def s(self):
        return cost(self.hdr, 1) * self.tok / self.cost

    def update(self, n_keep, n_acc):
        a = self.hdr["ema"]
        self.tok = (1.0 - a) * self.tok + a * (1.0 + n_acc)
        self.cost = (1.0 - a) * self.cost + a * (self.hdr["draft_ms"] + cost(self.hdr, n_keep + 1))


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(b))


def read(paths):
    """yields (file, header, step) with each step under the latest header of its file"""
    for path in paths:
        hdr = None
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("hdr"):
                    hdr = d
                    continue
                if hdr is None or d.get("run") != hdr.get("run"):
                    raise SystemExit(f"{path}:{n}: a step without its run's header")
                yield path, n, hdr, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--check", action="store_true", help="recompute every step from its header")
    ap.add_argument("--csv", help="write the summary rows to this CSV file")
    args = ap.parse_args()

    problems = 0
    n_ties = 0
    states = {}
    runs = defaultdict(lambda: {"hdr": None, "steps": 0, "k": Counter(), "drafted": 0, "kept": 0, "acc": 0,
                                "ms": 0.0, "e_tau": 0.0, "tok": 0, "noprob": 0,
                                "rule_k": Counter(), "rule_kept": 0, "rule_tok": 0, "rule_ms": 0.0})

    def problem(where, msg):
        nonlocal problems
        problems += 1
        if problems <= 50:
            print(f"PROBLEM {where}: {msg}")

    for path, n, hdr, d in read(args.logs):
        where = f"{path}:{n}"
        key = (path, hdr["run"])
        r = runs[key]
        r["hdr"] = hdr
        sk = (path, hdr["run"], d["slot"])
        if sk not in states:
            states[sk] = RunState(hdr)
        st = states[sk]

        p = [f32(x) for x in d["p"]]  # the drafter's floats, as the server used them
        has_p = len(p) == d["n_draft"]
        t1 = cost(hdr, 1)

        if args.check:
            if has_p:
                s_run = st.s()
                if hdr["s_mode"] == "run" and not close(d["s"], s_run, 1e-9):
                    problem(where, f"s {d['s']} against the replayed running speed-up {s_run}")
                n_rule, s, e, c, ties = choose(hdr, p, d["s"] if hdr["s_mode"] == "run" else s_run)
                n_ties += ties
                if n_rule != d["n_rule"] and not ties:
                    problem(where, f"n_rule {d['n_rule']}, the rule gives {n_rule} (p {p}, s {d['s']})")
                elif n_rule == d["n_rule"]:
                    if not close(d["e_tau"], e) or not close(d["cost_ms"], c) or not close(d["s"], s):
                        problem(where, f"e_tau/cost_ms/s {d['e_tau']}/{d['cost_ms']}/{d['s']}, recomputed {e}/{c}/{s}")
            elif d["n_rule"] != d["n_draft"]:
                problem(where, f"no probabilities, but n_rule {d['n_rule']} != n_draft {d['n_draft']}")
            want = d["n_rule"] if hdr["enabled"] else d["n_draft"]
            if d["n_keep"] != want:
                problem(where, f"n_keep {d['n_keep']}, expected {want} (switch {'on' if hdr['enabled'] else 'off'})")
            if hdr["enabled"] and has_p and (d["n_keep"] + 1) not in hdr["widths"]:
                problem(where, f"verify width {d['n_keep'] + 1} not among {hdr['widths']}")
            if not 0 <= d["n_acc"] <= d["n_keep"]:
                problem(where, f"n_acc {d['n_acc']} outside [0, n_keep {d['n_keep']}]")
            if d["on"] != hdr["enabled"]:
                problem(where, "on differs from the header")

        st.update(d["n_keep"], d["n_acc"])
        if args.check and not close(d["s_next"], st.s(), 1e-9):
            problem(where, f"s_next {d['s_next']} against the replayed {st.s()}")

        r["steps"] += 1
        r["k"][d["n_keep"] + 1] += 1
        r["drafted"] += d["n_draft"]
        r["kept"] += d["n_keep"]
        r["acc"] += d["n_acc"]
        r["tok"] += 1 + d["n_acc"]
        r["ms"] += hdr["draft_ms"] + cost(hdr, d["n_keep"] + 1)
        r["noprob"] += 0 if has_p else 1
        if has_p:
            P, x = [], 1.0
            for q in p[: d["n_keep"]]:
                x *= max(0.0, min(1.0, q))
                P.append(x)
            r["e_tau"] += 1.0 + sum(P)
        else:
            r["e_tau"] += 1.0 + d["n_keep"]
        # the rule on the same draft (exact as run when the switch was on)
        n_rule = d["n_rule"]
        r["rule_k"][n_rule + 1] += 1
        r["rule_kept"] += n_rule
        r["rule_tok"] += 1 + min(d["n_acc"], n_rule)
        r["rule_ms"] += hdr["draft_ms"] + cost(hdr, n_rule + 1)

    rows = []
    for (path, run), r in runs.items():
        hdr, n = r["hdr"], max(1, r["steps"])
        t1 = cost(hdr, 1)
        widths = " ".join(f"{k}:{c}" for k, c in sorted(r["k"].items()))
        rule_widths = " ".join(f"{k}:{c}" for k, c in sorted(r["rule_k"].items()))
        rows.append({
            "log": path,
            "run": run,
            "switch": "on" if hdr["enabled"] else "off",
            "n_max": hdr["n_max"],
            "s_mode": hdr["s_mode"] if hdr["s_mode"] != "fixed" else f"fixed {hdr['s_fixed']}",
            "scale": hdr["scale"],
            "steps": r["steps"],
            "no_prob_steps": r["noprob"],
            "verify_widths": widths,
            "drafted_per_step": round(r["drafted"] / n, 3),
            "verified_per_step": round(r["kept"] / n, 3),
            "accepted_per_step": round(r["acc"] / n, 3),
            "acceptance": round(r["acc"] / r["kept"], 3) if r["kept"] else "",
            "tokens_per_step": round(r["tok"] / n, 3),
            "expected_tokens_per_step": round(r["e_tau"] / n, 3),
            "model_ms_per_step": round(r["ms"] / n, 1),
            "model_speedup": round(r["tok"] * t1 / r["ms"], 3) if r["ms"] else "",
            "rule_verify_widths": rule_widths,
            "rule_tokens_per_step": round(r["rule_tok"] / n, 3),
            "rule_model_ms_per_step": round(r["rule_ms"] / n, 1),
            "rule_model_speedup": round(r["rule_tok"] * t1 / r["rule_ms"], 3) if r["rule_ms"] else "",
        })

    cols = list(rows[0].keys()) if rows else []
    if rows:
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(row[c]) for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    if args.check:
        steps = sum(r["steps"] for r in runs.values())
        print(f"check: {steps} steps in {len(runs)} runs, {problems} problems, {n_ties} tests within rounding")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
