#!/usr/bin/env python3
"""host_steps.py: one speculative step's time, split, from the fork's GGML_CUDA_PROFILE_HOST file (C338, Q900).

  python3 -I host_steps.py HOST_PROFILE.csv [--main-devices 0,1,2,3,4,5,6,7,8] [--steps N] > steps.md

The file has one line per graph evaluation and card: device, eval, host_enter_ms, host_exit_ms, gpu_us (the backend's
graph_compute entered and left on the host; the GPU time between two events around its work). A main-model pass is one
evaluation on each main device in order (0 ... 8 for nine cards); anything between two main passes is the draft (the MTP
draft's evaluations on its cards) or host work. Per step, from one main pass's first card to the next main pass's first
card:
  main_host_ms   host time from the main pass's first card's entry to its last card's exit (the verify chain as the host
                 sees it, with the hops)
  main_gpu_ms    the GPU time summed over the main pass's cards
  after_ms       from the last main card's exit to the next evaluation's entry: the wait for the cards, the sampling
                 data's copy to the host (with -bs the candidates, not the logits), sampling, the acceptance test, the
                 server's bookkeeping, the draft's set-up
  draft_host_ms  from the first to the last draft evaluation's exit in that step, and draft_gpu_ms their GPU time
  before_ms      from the last draft evaluation's exit to the next main pass's entry (the draft's sampling, the batch)
  step_ms        the whole step
The first steps (prompt and warm-up) are skipped: --skip (default 3). Prints the median, p10 and p90 of each over the
steps, and the totals.
"""
import argparse
import csv
import statistics


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--main-devices", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--skip", type=int, default=3)
    a = ap.parse_args()
    main_devs = [int(x) for x in a.main_devices.split(",")]
    rows = []
    with open(a.profile) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["host_enter_ms"]), float(r["host_exit_ms"]), int(r["device"]), float(r["gpu_us"]) / 1000.0))
    rows.sort()
    # main passes: a run of evaluations on main_devs in order, starting at main_devs[0]
    passes, i = [], 0
    while i < len(rows):
        if rows[i][2] == main_devs[0] and i + len(main_devs) <= len(rows) and \
                [rows[i + k][2] for k in range(len(main_devs))] == main_devs:
            passes.append((i, i + len(main_devs) - 1))
            i += len(main_devs)
        else:
            i += 1
    steps = []
    for p, q in zip(passes, passes[1:]):
        s, e = p
        nxt = q[0]
        main_host = rows[e][1] - rows[s][0]
        main_gpu = sum(rows[k][3] for k in range(s, e + 1))
        between = rows[e + 1:nxt]
        after = (rows[e + 1][0] if e + 1 < len(rows) else rows[nxt][0]) - rows[e][1]
        if between:
            draft_host = between[-1][1] - between[0][0]
            draft_gpu = sum(b[3] for b in between)
            before = rows[nxt][0] - between[-1][1]
        else:
            draft_host = draft_gpu = 0.0
            before = 0.0
        steps.append(dict(step_ms=rows[nxt][0] - rows[s][0], main_host_ms=main_host, main_gpu_ms=main_gpu,
                          after_ms=after, draft_host_ms=draft_host, draft_gpu_ms=draft_gpu, before_ms=before,
                          draft_evals=len(between)))
    steps = steps[a.skip:]
    keys = ["step_ms", "main_host_ms", "main_gpu_ms", "after_ms", "draft_host_ms", "draft_gpu_ms", "before_ms", "draft_evals"]
    print(f"{len(rows)} evaluations, {len(passes)} main passes on devices {a.main_devices}, {len(steps)} steps after skipping {a.skip}")
    print()
    print("| part | median | p10 | p90 | total over the steps |")
    print("|---|---:|---:|---:|---:|")
    for k in keys:
        xs = [s[k] for s in steps]
        tot = sum(xs)
        print(f"| {k} | {statistics.median(xs):.1f} | {pct(xs, 0.1):.1f} | {pct(xs, 0.9):.1f} | {tot:.0f} |")


if __name__ == "__main__":
    main()
