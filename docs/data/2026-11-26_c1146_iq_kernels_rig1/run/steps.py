#!/usr/bin/env python3
"""steps.py: C1146's step pair from one or more deep-lane runs (Q1871).

  python3 -I steps.py --control u0 lane1/requests.csv lane2/requests.csv [--tsv steps.tsv] > steps.md

Each requests.csv is the deep lane's (bench/deep_lane): one line per request with its arm, load, kind, pass, depth,
gen_n, gen_ms, draft_accepted and reply hashes. A step is one pass of the main model: a request's steps are its
generated tokens minus the accepted draft tokens, and ms a step is its generation time over its steps (deep_table.py's
definition). Profiled arms (names starting with g) are left out. Per depth and pass (1: the first request at a new
depth, 2: the same request again, s: the sampler pass): every load's ms a step, then per arm its loads' mean, the
spread (fastest to slowest load) and the difference to the control; the class is gain or loss when the difference is
larger than the larger of the two spreads and both arms have two loads, negligible when it is within them, and no
class otherwise. Also: whether every load wrote the same reply (text hash) per depth and pass, the arms named with
--other (another model file: pmix-q2-fast's p0) checked apart from the rest.
Standard library only.
"""
import argparse
import csv
import sys
from collections import OrderedDict, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="+")
    ap.add_argument("--control", default="u0")
    ap.add_argument("--other", action="append", default=[], help="an arm with another model file")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    rows = []
    for p in a.requests:
        with open(p, newline="") as f:
            rows += [r for r in csv.DictReader(f) if r["load"] != "warm" and not r["arm"].startswith("g")
                     and r["kind"] in ("decode", "sample")]
    step = OrderedDict()  # (depth, pass) -> arm -> [(load, ms, steps)]
    reply = defaultdict(set)  # (depth, pass, file) -> text hashes over every load
    for r in rows:
        steps = int(r["gen_n"]) - int(r["draft_accepted"] or 0)
        ms = float(r["gen_ms"]) / steps
        key = (int(r["depth"]), r["pass"])
        step.setdefault(key, OrderedDict()).setdefault(r["arm"], []).append((r["load"], ms, steps))
        reply[key + (r["arm"] if r["arm"] in a.other else "",)].add(r["text_md5"])
    P = print
    long = []
    P("# The step pair: ms a step, every load\n")
    P(f"Arms: {', '.join(sorted({r['arm'] for r in rows}))}; control `{a.control}`. A step is one pass of the main model"
      " (its generated tokens minus the accepted draft tokens); ms a step = generation time / steps.\n")
    P("| Depth | Pass | Arm | Per load (ms a step, steps) | Mean | Spread | Against the control | % | Class |")
    P("|---:|---|---|---|---:|---:|---:|---:|---|")
    for key in sorted(step, key=lambda k: (k[0], {"1": 0, "2": 1, "s": 2}[k[1]])):
        arms = step[key]
        c = arms.get(a.control)
        cm = sum(x[1] for x in c) / len(c) if c else None
        cs = (max(x[1] for x in c) - min(x[1] for x in c)) if c and len(c) > 1 else None
        for arm, loads in arms.items():
            m = sum(x[1] for x in loads) / len(loads)
            s = max(x[1] for x in loads) - min(x[1] for x in loads) if len(loads) > 1 else None
            per = "; ".join(f"{ld}: {ms:.2f} ({n})" for ld, ms, n in loads)
            for ld, ms, n in loads:
                long.append((str(key[0]), key[1], arm, ld, str(n), f"{ms:.3f}"))
            if arm == a.control or cm is None:
                diff = pct = cls = ""
                if arm == a.control:
                    cls = "control"
            else:
                d = m - cm
                diff, pct = f"{d:+.2f}", f"{d / cm * 100:+.2f}"
                if cs is None or s is None:
                    cls = "no class (one load)"
                else:
                    lim = max(cs, s)
                    cls = "gain" if d < -lim else "loss" if d > lim else "negligible"
            P(f"| {key[0]:,} | {key[1]} | {arm} | {per} | {m:.2f} | {'—' if s is None else f'{s:.2f}'} | {diff} | {pct} | {cls} |")
    P("\n## Replies\n")
    for key in sorted(reply, key=lambda k: (k[2], k[0], k[1])):
        n = len(reply[key])
        who = f"`{key[2]}`" if key[2] else "the main file's arms"
        P(f"- {who}, depth {key[0]:,}, pass {key[1]}: " + ("the same reply on every load" if n == 1 else
          f"**{n} different replies**: " + ", ".join(sorted(h[:8] for h in reply[key]))))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("depth\tpass\tarm\tload\tsteps\tms_per_step\n")
            for r in long:
                f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
