#!/usr/bin/env python3
"""Summarizes llama-pipespec traces (--pipe-trace, written by rig_run.sh as <job>.trace).

Each trace row is one sampled token: microseconds since the first submit, guesses in flight when it was sampled, and
the outcome for the token after it: h (a right guess was already in flight), m (a wrong one: dropped), n (no guess),
p (plain decoding), r (a token of one of the raced samples), f (the next token already in flight, fed on the devices
with --pipe-feed). The time from one sampled token to the next is attributed
to the earlier token's outcome:
  after h  the next token was already on its way: one pipeline cycle;
  after m  the next token starts from card 1: a full pass, like plain decoding, plus the cost of the dropped guesses;
  after n  the same full pass without dropped guesses;
  after r  the next raced token, usually another sample's: with K samples in flight, a full pass divided by up to K;
  after f  a full pass without the host between the cards: the last card's sample goes straight to card 1.
Usage: traces.py <dir or .trace files>...
"""
import os
import statistics
import sys


def load(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            t, d, o = line.split()
            rows.append((int(t), int(d), o))
    return rows


def main(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(os.path.join(a, x) for x in os.listdir(a) if x.endswith(".trace"))
        else:
            files.append(a)
    print(f"{'job':28} {'tokens':>6} {'tok/s':>7} {'median ms':>9}  {'after h':>13} {'after m':>13} {'after n':>13} {'after r':>13} {'after f':>13}")
    for p in files:
        rows = load(p)
        if len(rows) < 3:
            continue
        gaps = {"h": [], "m": [], "n": [], "p": [], "r": [], "f": []}
        all_gaps = []
        for (t0, _, o0), (t1, _, _) in zip(rows, rows[1:]):
            g = (t1 - t0) / 1000.0
            gaps[o0].append(g)
            all_gaps.append(g)
        span = (rows[-1][0] - rows[0][0]) / 1e6
        rate = (len(rows) - 1) / span if span > 0 else 0.0

        def cell(k):
            v = gaps[k]
            return f"{len(v):4d} x {statistics.median(v):5.2f}" if v else f"{'-':>13}"

        name = os.path.basename(p)[: -len(".trace")]
        print(f"{name:28} {len(rows):6d} {rate:7.2f} {statistics.median(all_gaps):9.2f}  {cell('h')} {cell('m')} {cell('n')} {cell('r')} {cell('f')}")
    print("after X: number of intervals x median ms")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
