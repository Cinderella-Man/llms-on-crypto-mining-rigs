#!/usr/bin/env python3
"""align_t0.py: the steady clock's time of a GGML_CUDA_PROFILE_HOST file's 0 ms, from the LLAMA_STEP_TRACE file of the
same load (R86, Q1313), for step_gap.py --profile-t0 when the server log lacks the profile's anchor line.

  python3 -I align_t0.py TRACE.jsonl HOST_PROFILE.csv [--main-device 0] > align.md

The profile's anchor ("its 0 ms is <t> ms on the steady clock") is logged at INFO by ggml; llama-server at its default
verbosity does not print ggml's INFO lines, so the log can lack it. Both files time the same thread on the same clock,
so the offset between them is one constant, and the trace bounds it: every graph_compute the profile times runs inside
its context's process_ubatch, between that ubatch's inputs set (v.ub.in, d.ub.in) and its graphs queued (v.ub.comp,
d.ub.comp). The main device's evaluations are matched to the main context's ubatches from the end of both files (a
load's warm-up evaluations come before the trace's first line); each pair gives offset >= ub.in - entry and
offset <= ub.comp - exit, and the intersection over all pairs is the offset's interval. With its middle, every
evaluation after the trace's first mark is then checked to lie inside one ubatch window, the main or the draft
context's. Prints the interval, the ubatches that bound it, the check per card, and the line
t0 = <the trace header's t0 + the middle>, the value step_gap.py takes as --profile-t0.
"""
import argparse
import bisect
import csv
import json
import sys


def windows(trace, role):
    """(ub.in, ub.comp) of every ubatch of one context, in order"""
    out = []
    for o in trace:
        start = None
        for name, t in o["t"]:
            if name == f"{role}.ub.in":
                start = t
            elif name == f"{role}.ub.comp" and start is not None:
                out.append((start, t))
                start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("--main-device", type=int, default=0)
    a = ap.parse_args()

    with open(a.trace) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    header, trace = rows[0], rows[1:]
    prof = []
    with open(a.profile) as f:
        for r in csv.DictReader(f):
            prof.append((int(r["device"]), float(r["host_enter_ms"]), float(r["host_exit_ms"])))
    prof.sort(key=lambda r: r[1])

    vw, dw = windows(trace, "v"), windows(trace, "d")
    main = [p for p in prof if p[0] == a.main_device]

    def bounds(s):
        """the offset's interval when the main device's evaluation k + s is the trace's main ubatch k"""
        lo, hi, arg_lo, arg_hi, n = -1e30, 1e30, None, None, 0
        for k, (w0, w1) in enumerate(vw):
            if not 0 <= k + s < len(main):
                continue
            _, e, x = main[k + s]
            n += 1
            if w0 - e > lo:
                lo, arg_lo = w0 - e, k
            if w1 - x < hi:
                hi, arg_hi = w1 - x, k
        return lo, hi, arg_lo, arg_hi, n

    print(f"# {a.trace} beside {a.profile}\n")
    print(f"- main context's ubatches in the trace: {len(vw)}; card {a.main_device}'s evaluations in the profile: "
          f"{len(main)}; the draft context's ubatches in the trace: {len(dw)}")
    print("\n| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |\n|---:|---:|---:|---:|---|")
    fit = []
    for s in range(-4, 5):
        lo, hi, arg_lo, arg_hi, n = bounds(s)
        print(f"| {s} | {n} | {lo:.3f} (ubatch {arg_lo}) | {hi:.3f} (ubatch {arg_hi}) | {'yes' if lo <= hi else 'no'} |")
        if lo <= hi:
            fit.append((n, s, lo, hi, arg_lo, arg_hi))
    if not fit:
        print("\n- NO OFFSET fits every pair at any shift")
        return 1
    n, s, lo, hi, arg_lo, arg_hi = max(fit)
    print(f"\n- shift {s} ({n} pairs): card {a.main_device}'s evaluations before the trace's first ubatch are the load's "
          f"warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit")
    print(f"- card {a.main_device} bounds the offset (trace ms = profile ms + offset) to {lo:.3f} … {hi:.3f} ms, width "
          f"{hi - lo:.3f} ms: from below by ubatch {arg_lo}'s inputs set, from above by ubatch {arg_hi}'s graphs queued")

    # Inside that interval, the offset that puts the most evaluations of any card inside one ubatch window of the
    # trace (main or draft): the draft passes' windows are under 1 ms long, and an evaluation the host enters on the
    # deciding thread (the first split of a pass) falls inside its window only near the true offset. Evaluations a
    # helper thread enters after the window (a staged hop's later splits) fall outside at any offset and count for none.
    vs, ds = [w[0] for w in vw], [w[0] for w in dw]

    def where(te, tx):
        for ws, st, name in ((vw, vs, "main"), (dw, ds, "draft")):
            i = bisect.bisect_right(st, te) - 1
            if i >= 0 and ws[i][0] <= te and tx <= ws[i][1]:
                return name
        return "none"

    t_first = trace[0]["t"][0][1]
    step = 0.005
    grid = [lo + step * i for i in range(int((hi - lo) / step) + 1)]
    score = [sum(where(e + o, x + o) != "none" for _, e, x in prof if e + o >= t_first) for o in grid]
    best = max(score)
    plateau = [o for o, c in zip(grid, score) if c == best]
    off = (plateau[0] + plateau[-1]) / 2.0
    print(f"- the most evaluations inside a window, {best}, at offsets {plateau[0]:.3f} … {plateau[-1]:.3f} ms "
          f"({len(plateau)} grid points of {step} ms, of {len(grid)}); the offset: their middle, {off:.3f}")

    count = {}
    for dev, e, x in prof:
        if e + off < t_first:
            continue
        w = where(e + off, x + off)
        count[(dev, w)] = count.get((dev, w), 0) + 1
    print("\n| card | in a main ubatch | in a draft ubatch | in none |\n|---|---:|---:|---:|")
    for dev in sorted({d for d, _ in count}):
        print(f"| {dev} | " + " | ".join(str(count.get((dev, w), 0)) for w in ["main", "draft", "none"]) + " |")
    print(f"\nt0 = {header['t0_steady_ms'] + off:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
