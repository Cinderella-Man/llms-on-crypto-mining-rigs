#!/usr/bin/env python3
"""timeline.py: one request's evaluations around the start and the end of its generation (FN85, Q1410), on the
profile's clock with the server's T_s and T_end placed as fn85.py places them.

  python3 -I timeline.py OUT LOAD REQUEST --step-split STEP_SPLIT_DIR [--head-before 1500] [--head-passes 2]
      [--tail-passes 2] > timeline.md

REQUEST is greedy or long (the first or second request of the load); T_s and T_end come from fn85.py's anchor (the
slot's start delta ms before the request's first evaluation). Each line: the evaluation's card, its entry,
exit and GPU time, its end (entry + the longer of its host span and its GPU time), the gap from the previous
evaluation's entry, and its entry against T_s or T_end; idle stretches of 100 ms or more with no evaluation running
on any card are marked.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("load")
    ap.add_argument("request")
    ap.add_argument("--step-split", required=True)
    ap.add_argument("--head-before", type=float, default=1500.0)
    ap.add_argument("--head-passes", type=int, default=2)
    ap.add_argument("--tail-passes", type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, a.step_split)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import step_split as ss  # noqa: E402
    import fn85  # noqa: E402

    names = ["greedy", "long"]
    d = os.path.join(a.out, a.load)
    rows = ss.load_rows(os.path.join(d, "host_profile.csv"))
    segs = ss.cut(rows, 2000.0)
    seg_passes = [ss.main_passes(s, list(range(9))) for s in segs]
    req = [(s, p) for s, p in zip(segs, seg_passes) if len(p) >= 10][-2:]
    tasks = fn85.parse_log(os.path.join(d, f"server-{a.load}.log"))[-2:]
    k = names.index(a.request)
    s, ps = req[k]
    task = tasks[k]
    between = set()
    for s2, ps2 in req:
        for p, q in zip(ps2, ps2[1:]):
            between.update(x[2] for x in s2[p[1] + 1:q[0]])
    delta, how = fn85.anchor(s, ps, task, max(between), 2.0)
    t_s = s[0][0] - delta + task["prompt_ms"]
    t_end = t_s + task["predicted_ms"]
    main_of = {}
    for j, p in enumerate(ps):
        for i in range(p[0], p[1] + 1):
            main_of[i] = j
    # the head: from head_before ms before T_s to the entry of the head_passes-th main pass after T_s
    after_ts = [j for j, p in enumerate(ps) if s[p[0]][0] >= t_s]
    head_end = s[ps[after_ts[min(len(after_ts) - 1, a.head_passes)]][0]][0] if after_ts else t_s
    tail_start = s[ps[-a.tail_passes][0]][0]

    def show(lo, hi, ref, ref_name):
        print(f"| card | eval | main pass | entry | exit | GPU | end | from the previous entry | entry - {ref_name} |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        prev, run_end = None, None
        for i, (en, ex, dev, gpu) in enumerate(s):
            if en < lo or en > hi:
                continue
            end = en + max(ex - en, gpu)
            if run_end is not None and en - run_end >= 100.0:
                print(f"| | **idle {en - run_end:.1f} ms** (no evaluation running) | | | | | | | |")
            mp = main_of.get(i, "")
            kind = "main" if mp != "" else "draft"
            print(f"| {dev} | {kind} | {mp} | {en:.3f} | {ex:.3f} | {gpu:.3f} | {end:.3f} | "
                  f"{'' if prev is None else f'{en - prev:.3f}'} | {en - ref:+.1f} |")
            prev = en
            run_end = end if run_end is None else max(run_end, end)

    print(f"## {a.load}, {a.request}: T_s {t_s:.3f}, T_end {t_end:.3f} (profile ms); prompt {task['prompt_ms']:.2f} ms, "
          f"generation {task['predicted_ms']:.2f} ms, {task['n_gen']} tokens, {len(ps)} main passes; the slot starts "
          f"{delta:.1f} ms before the first evaluation ({how})")
    print()
    print(f"### The head: {a.head_before:.0f} ms before T_s to main pass {after_ts[min(len(after_ts) - 1, a.head_passes)] if after_ts else '-'}")
    print()
    show(t_s - a.head_before, head_end, t_s, "T_s")
    print()
    print(f"### The tail: main pass {len(ps) - a.tail_passes} to the request's last evaluation (T_end {t_end:.3f})")
    print()
    show(tail_start, float("inf"), t_end, "T_end")
    last_end = max(en + max(ex - en, gpu) for en, ex, dev, gpu in s)
    print()
    print(f"The request's last evaluation ends at {last_end:.3f}: T_end - that = {t_end - last_end:+.1f} ms.")


if __name__ == "__main__":
    main()
