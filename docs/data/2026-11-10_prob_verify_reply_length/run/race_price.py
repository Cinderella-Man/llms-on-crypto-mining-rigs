#!/usr/bin/env python3
"""What a first-finish race and a length cap would save on these replies (Q93, Q1195).

Reads replies.tsv (one row per reply: load, arm, task, seed_index, length, ...). The same prompt was sent with
several seeds, so for every prompt the replies of one arm are samples of one request. For N = 1..5 the script takes
every N-subset of a prompt's samples (all of them, not a draw) and the shortest reply in each: the tokens a race of
N samples generates before its first finisher ends. The second finisher is the same with the second shortest.
Counts only; no time is read.

usage: python3 race_price.py replies.tsv > race_price.md
Standard library only.
"""
import itertools
import statistics
import sys


def kth(xs, n, k):
    c = list(itertools.combinations(xs, n))
    return sum(sorted(t)[k] for t in c) / len(c)


def main(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]
    arms = sorted({r["arm"] for r in rows})
    print("# The price of a first-finish race and of a length cap on these replies\n")
    print("Tokens of whole replies (reasoning and answer). A prompt's samples are its replies in every load of the")
    print("arm. \"Share\" is against one sample (N = 1).\n")
    print("## First finisher of N samples of one request\n")
    print("| Arm | Prompts | Samples a prompt | N = 1 | N = 2 | N = 3 | N = 4 | N = 5 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        g = {}
        for r in rows:
            if r["arm"] == arm:
                g.setdefault(r["task"], []).append(int(r["length"]))
        base = sum(sum(v) / len(v) for v in g.values())
        cells = []
        for n in (1, 2, 3, 4, 5):
            tot = sum(kth(v, n, 0) for v in g.values())
            cells.append("%.0f (%.3f)" % (tot, tot / base))
        sizes = sorted({len(v) for v in g.values()})
        print("| `%s` | %d | %s | %s |" % (arm, len(g), ", ".join(map(str, sizes)), " | ".join(cells)))
    print("\nExpected tokens summed over the prompts, with the share in brackets.\n")
    print("## Second finisher of N (a vote of two needs it)\n")
    print("| Arm | N = 2 | N = 3 | N = 4 | N = 5 |")
    print("|---|---:|---:|---:|---:|")
    for arm in arms:
        g = {}
        for r in rows:
            if r["arm"] == arm:
                g.setdefault(r["task"], []).append(int(r["length"]))
        base = sum(sum(v) / len(v) for v in g.values())
        cells = []
        for n in (2, 3, 4, 5):
            tot = sum(kth(v, n, 1) for v in g.values())
            cells.append("%.0f (%.3f)" % (tot, tot / base))
        print("| `%s` | %s |" % (arm, " | ".join(cells)))
    print("\n## Per prompt, every arm's samples together\n")
    print("| Prompt | Samples | Shortest | Median | Mean | Longest |")
    print("|---|---:|---:|---:|---:|---:|")
    g = {}
    for r in rows:
        g.setdefault(r["task"], []).append(int(r["length"]))
    for task, v in sorted(g.items(), key=lambda kv: -sum(kv[1])):
        print("| %s | %d | %d | %.1f | %.1f | %d |" % (task, len(v), min(v), statistics.median(v), sum(v) / len(v), max(v)))
    print("\n## A cap on a reply's length\n")
    print("| Cap, tokens | Tokens kept, share of all | Replies over the cap |")
    print("|---:|---:|---:|")
    allv = [int(r["length"]) for r in rows]
    tot = sum(allv)
    for cap in (1000, 2000, 4000, 8000):
        kept = sum(min(x, cap) for x in allv)
        print("| %d | %.3f | %d of %d |" % (cap, kept / tot, sum(x > cap for x in allv), len(allv)))
    print("\nA capped reply is counted at the cap: what the model writes after a forced end of its reasoning is not")
    print("in these numbers.")


if __name__ == "__main__":
    main(sys.argv[1])
