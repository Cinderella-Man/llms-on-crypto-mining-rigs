"""The head's acceptance by what llama-server's lookups would have guessed at each position: score.py's per-position
scores on held-out agent sessions, split by the guess source cost_sim.py replays (the copy lookup, the request's
continuation of a 2-token or longer context, of a 1-token context, or none), with each lookup's own accuracy at those
positions next to it. It shows where the head competes in the server's guess order: with --pipe-guess staircase from
fork branch pascal-p106-v3-cheapguess on, the head takes only the positions the copy lookup and the 2-token
continuations leave.

usage: python source_split.py ROWS.jsonl SCORES.npz [--window g9] [--metric uniform]
ROWS.jsonl: the rows score.py scored (prompt_tokens, gen_tokens, gen_ranges; seq ids from the id's numeric end).
Needs numpy.
"""
import argparse
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np

_spec = importlib.util.spec_from_file_location("cost_sim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      "cost_sim.py"))
cost_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cost_sim)


def labels(r):
    """{gen index j: (source, lookup right)} for the generated positions of a row"""
    seq = r["prompt_tokens"] + r["gen_tokens"]
    n_prompt = len(r["prompt_tokens"])
    ranges = r.get("gen_ranges") or [(n_prompt, len(seq))]
    g = cost_sim.ServerGuesser()
    out, k = {}, 0
    for s, e in ranges:
        while k < s:
            g.push(seq[k])
            k += 1
        for a in range(s, e):
            if a > s:
                g.count_to(len(g.hist) - 2)
                cp, c2, c1 = g.copy_guess(), g.cont_guess(2), g.cont_guess(1)
                if cp is not None:
                    out[a - n_prompt] = ("copy", cp == seq[a])
                elif c2 is not None:
                    out[a - n_prompt] = ("cont2", c2 == seq[a])
                elif c1 is not None:
                    out[a - n_prompt] = ("cont1", c1 == seq[a])
                else:
                    out[a - n_prompt] = ("none", False)
            g.push(seq[a])
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("scores")
    ap.add_argument("--window", default="g9")
    ap.add_argument("--metric", default="uniform", choices=("uniform", "coupled", "top"))
    args = ap.parse_args()
    z = np.load(args.scores)
    key = f"{args.window}/{args.metric}"
    if key not in z.files:
        raise SystemExit(f"{key} not in {args.scores}: " + ", ".join(sorted({f.split('/')[0] for f in z.files})))
    seq, j, v = z["seq"], z["j"], z[key]
    conf = z[f"{args.window}/conf"]
    by_seq = defaultdict(list)
    for i in range(len(seq)):
        by_seq[int(seq[i])].append(i)
    acc = defaultdict(lambda: [0, 0.0, 0])   # source -> [positions, head sum, lookup right]
    per = defaultdict(lambda: ([], [], []))  # source -> (head right, head's top probability, lookup right) per position
    for line in open(args.rows):
        r = json.loads(line)
        sid = int(r["id"].split("_")[-1])
        if sid not in by_seq:
            continue
        lab = labels(r)
        for i in by_seq[sid]:
            x = lab.get(int(j[i]))
            if x is None:
                continue
            a = acc[x[0]]
            a[0] += 1
            a[1] += float(v[i])
            a[2] += x[1]
            p = per[x[0]]
            p[0].append(float(v[i]))
            p[1].append(float(conf[i]))
            p[2].append(float(x[1]))
    total = sum(a[0] for a in acc.values())
    print(f"{total} scored positions, window {args.window}, metric {args.metric}")
    print("| source | positions | share | head right | lookup right |")
    print("|---|---|---|---|---|")
    for s in ("copy", "cont2", "cont1", "none"):
        n, h, lk = acc[s]
        if n:
            print(f"| {s} | {n} | {n / total:.3f} | {h / n:.3f} | {lk / n:.3f} |")
    rest = [acc[s] for s in ("cont2", "cont1", "none")]
    n = sum(a[0] for a in rest)
    print(f"| not copy (the old order's head positions) | {n} | {n / total:.3f} | {sum(a[1] for a in rest) / n:.3f} | |")
    # the head in place of the lookup wherever its top probability is at least t: right guesses per position, and the
    # share of positions the head takes
    ts = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)
    print(f"\nThe head where its top probability (at the scored alpha) is at least t, else the lookup: right per position "
          f"(share the head takes)\n")
    print("| source | " + " | ".join(f"t {t:g}" for t in ts) + " |")
    print("|---|" + "---|" * len(ts))
    for s in ("copy", "cont2", "cont1"):
        h, c, lk = (np.array(x) for x in per[s])
        if not len(h):
            continue
        cells = []
        for t in ts:
            use = c >= t
            cells.append(f"{np.where(use, h, lk).mean():.3f} ({use.mean():.2f})")
        print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
