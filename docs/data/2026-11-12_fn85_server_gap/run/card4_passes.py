#!/usr/bin/env python3
"""card4_passes.py: each load's draft evaluations on the draft's embedding card with 5 ms or more of GPU time (FN85,
Q1410), by request, so that the first such evaluation of each server's life can be compared with the second request's
last-step wait on that card.

  python3 -I card4_passes.py OUT --loads base,...,f-2 --step-split STEP_SPLIT_DIR [--card 4] [--min-gpu 5] > card4.md

A draft evaluation is one outside the main passes (step_split's nine-card passes). The drafting passes' ordinary
evaluations on the embedding card take 0.01 to 0.3 ms; the heavier ones are listed with their time from the request's
first evaluation, their GPU time and their host span.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--loads", required=True)
    ap.add_argument("--step-split", required=True)
    ap.add_argument("--card", type=int, default=4)
    ap.add_argument("--min-gpu", type=float, default=5.0)
    a = ap.parse_args()
    sys.path.insert(0, a.step_split)
    import step_split as ss  # noqa: E402

    print(f"| load | request | draft evaluations on card {a.card} with {a.min_gpu:g} ms or more of GPU time after the "
          "request's prompt (ms from its first evaluation: GPU, host) |")
    print("|---|---|---|")
    for load in a.loads.split(","):
        rows = ss.load_rows(os.path.join(a.out, load, "host_profile.csv"))
        segs = ss.cut(rows, 2000.0)
        sp = [ss.main_passes(s, list(range(9))) for s in segs]
        req = [(s, p) for s, p in zip(segs, sp) if len(p) >= 10][-2:]
        for name, (s, ps) in zip(["greedy", "long"], req):
            inm = set(i for p in ps for i in range(p[0], p[1] + 1))
            # after the prompt: past the first main pass whose GPU time is at most 1.5 times the median main pass
            med = sorted(sum(s[i][3] for i in range(p[0], p[1] + 1)) for p in ps)[len(ps) // 2]
            k0 = next(j for j, p in enumerate(ps) if sum(s[i][3] for i in range(p[0], p[1] + 1)) <= 1.5 * med)
            t0 = s[ps[k0][0]][0]
            heavy = [f"{s[i][0] - s[0][0]:.1f} ({s[i][3]:.2f}, {s[i][1] - s[i][0]:.1f})" for i in range(len(s))
                     if s[i][2] == a.card and i not in inm and s[i][3] >= a.min_gpu and s[i][0] > t0]
            print(f"| {load} | {name} | {len(heavy)}: " + ", ".join(heavy) + " |")


if __name__ == "__main__":
    main()
