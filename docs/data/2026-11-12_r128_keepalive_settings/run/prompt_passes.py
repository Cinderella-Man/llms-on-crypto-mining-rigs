#!/usr/bin/env python3
"""prompt_passes.py: R128 (Q1419), the second request's prompt (2,760 tokens) main pass by main pass, from each load's
host profile (GGML_CUDA_PROFILE_HOST), to find where the loads' prompt times part.

  python3 -I prompt_passes.py ALL --loads f,f-2,fka250,feager --step-split STEP_SPLIT_DIR > prompt_passes.md

ALL/<load>/host_profile.csv (run/r128_tables.sh's all/ folder). The request is the profile's last stretch of
evaluations (S37's step_split.cut with a 2,000 ms gap). Its main passes are step_split.main_passes' (cards 0 to 8 in
turn); the prompt's are those before the first main pass in which no card's GPU time reaches --min-gpu ms (the first
decode pass). Per pass: when card 0 enters it (ms from the request's first evaluation), the time from card 0's entry to
card 8's end, each card's GPU time, and the evaluations between this pass and the next (the draft's). An evaluation
ends at entry + the longer of its host span and its GPU time.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all")
    ap.add_argument("--loads", required=True)
    ap.add_argument("--step-split", required=True)
    ap.add_argument("--min-gpu", type=float, default=30.0)
    a = ap.parse_args()
    sys.path.insert(0, a.step_split)
    import step_split as ss  # noqa: E402

    end = lambda r: r[0] + max(r[1] - r[0], r[3])  # noqa: E731
    devs = list(range(9))
    print("| load | pass | card 0 enters, ms | span to card 8's end, ms | GPU ms, cards 0 to 8 | GPU ms, all cards | "
          "evaluations after it (cards) |")
    print("|---|---:|---:|---:|---|---:|---|")
    totals = []
    for load in a.loads.split(","):
        rows = ss.load_rows(os.path.join(a.all, load, "host_profile.csv"))
        segs = [s for s in ss.cut(rows, 2000.0) if len(ss.main_passes(s, devs)) >= 10]
        s = segs[-1]
        ps = ss.main_passes(s, devs)
        t0 = s[0][0]
        n = 0
        for j, (i0, i1) in enumerate(ps):
            g = [s[i][3] for i in range(i0, i1 + 1)]
            if max(g) < a.min_gpu:
                break
            nxt = ps[j + 1][0] if j + 1 < len(ps) else len(s)
            other = [s[i][2] for i in range(i1 + 1, nxt)]
            print(f"| {load} | {j + 1} | {s[i0][0] - t0:.1f} | {end(s[i1]) - s[i0][0]:.1f} | "
                  f"{' / '.join(f'{x:.1f}' for x in g)} | {sum(g):.1f} | {len(other)} ({','.join(map(str, sorted(set(other))))}) |")
            n = j + 1
        last = ps[n - 1][1]
        totals.append((load, n, end(s[last]) - t0, sum(s[i][3] for i0, i1 in ps[:n] for i in range(i0, i1 + 1))))
    print()
    print("| load | prompt passes | first evaluation to the last prompt pass's end, ms | GPU ms of the prompt passes, all cards |")
    print("|---|---:|---:|---:|")
    for load, n, span, g in totals:
        print(f"| {load} | {n} | {span:.1f} | {g:.1f} |")


if __name__ == "__main__":
    main()
