#!/usr/bin/env python3
"""draft_passes.py: the MTP draft's passes inside a speculative step, from the fork's GGML_CUDA_PROFILE_HOST file (Q961).

  python3 -I draft_passes.py HOST_PROFILE.csv [--main-devices 0,1,2,3,4,5,6,7,8] [--skip 3] > passes.md

The file has one line per graph evaluation and card: device, eval, host_enter_ms, host_exit_ms, gpu_us (the backend's
graph_compute entered and left on the host; the GPU time between two events around its work). A main-model pass is one
evaluation on each main device in order; the evaluations between two main passes are the draft's. A draft pass is one
evaluation per draft card, in order: with -otd token_embd=CUDA4 and the draft's layer and head on card 8, a card-4
evaluation (the token's embedding row) and a card-8 evaluation; with the row cache on (LLAMA_DRAFT_EMBD_CACHE) or the
embeddings on the host, the card-8 evaluation alone. Per draft pass k of a step (0 = the catch-up pass over the verified
tokens, 1 ... = the drafting passes):
  lead_ms   from the pass's first evaluation's exit to its last card's entry: with two cards, the crossing (card 4's
            row to the host and on to card 8, and card 8's other inputs); 0 with one card
  first_ms  the host span of the pass's first evaluation when it is not the last card's (card 4's), else 0
  gpu_ms    the GPU time of the pass, summed over its cards
  wait_ms   from the pass's last card's exit to the next pass's first entry (the host waits for the sampled id, samples,
            builds the next batch); for the step's last pass, to the next main pass's first entry
  idle_ms   wait_ms minus the part of the pass's last-card GPU time still running at its exit (gpu - host span): the
            host's own turnaround after the pass's GPU work, when the next pass depends on it
Prints, per pass index, the median, p10 and p90 of each, and per step the draft's host span (first draft entry to last
draft exit), its GPU time and the sum of the crossings.
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
        between = rows[p[1] + 1:q[0]]
        if not between:
            continue
        last_dev = max(b[2] for b in between)  # the draft's layer and head: the highest card of its evaluations
        dps, cur = [], []
        for b in between:
            cur.append(b)
            if b[2] == last_dev:
                dps.append(cur)
                cur = []
        if cur:
            dps.append(cur)
        out = []
        for k, dp in enumerate(dps):
            last = dp[-1]
            nxt_enter = dps[k + 1][0][0] if k + 1 < len(dps) else rows[q[0]][0]
            lead = last[0] - dp[0][1] if len(dp) > 1 else 0.0
            first = dp[0][1] - dp[0][0] if len(dp) > 1 else 0.0
            gpu = sum(x[3] for x in dp)
            wait = nxt_enter - last[1]
            idle = wait - max(0.0, last[3] - (last[1] - last[0]))
            out.append(dict(lead_ms=lead, first_ms=first, gpu_ms=gpu, wait_ms=wait, idle_ms=idle, n_evals=len(dp)))
        steps.append(dict(passes=out, draft_host_ms=between[-1][1] - between[0][0],
                          draft_gpu_ms=sum(b[3] for b in between), step_ms=rows[q[0]][0] - rows[p[0]][0]))
    steps = steps[a.skip:]
    print(f"{len(rows)} evaluations, {len(passes)} main passes on devices {a.main_devices}, {len(steps)} steps after "
          f"skipping {a.skip}")
    n_pass = statistics.median([len(s["passes"]) for s in steps]) if steps else 0
    print(f"draft passes a step (median): {n_pass:.0f}; evaluations a pass (median): "
          f"{statistics.median([d['n_evals'] for s in steps for d in s['passes']]):.0f}")
    print()
    print("| pass | part | median, ms | p10 | p90 |")
    print("|---|---|---:|---:|---:|")
    for k in range(int(n_pass)):
        ds = [s["passes"][k] for s in steps if len(s["passes"]) > k]
        for key in ["lead_ms", "first_ms", "gpu_ms", "wait_ms", "idle_ms"]:
            xs = [d[key] for d in ds]
            print(f"| {k} | {key} | {statistics.median(xs):.2f} | {pct(xs, 0.1):.2f} | {pct(xs, 0.9):.2f} |")
    print()
    print("| per step | median, ms | p10 | p90 | total over the steps |")
    print("|---|---:|---:|---:|---:|")
    per = {
        "step_ms": [s["step_ms"] for s in steps],
        "draft_host_ms": [s["draft_host_ms"] for s in steps],
        "draft_gpu_ms": [s["draft_gpu_ms"] for s in steps],
        "lead_sum_ms": [sum(d["lead_ms"] for d in s["passes"]) for s in steps],
        "lead_sum_drafting_ms": [sum(d["lead_ms"] for d in s["passes"][1:]) for s in steps],
        "idle_sum_drafting_ms": [sum(d["idle_ms"] for d in s["passes"][1:-1]) for s in steps],
    }
    for key, xs in per.items():
        print(f"| {key} | {statistics.median(xs):.2f} | {pct(xs, 0.1):.2f} | {pct(xs, 0.9):.2f} | {sum(xs):.0f} |")


if __name__ == "__main__":
    main()
