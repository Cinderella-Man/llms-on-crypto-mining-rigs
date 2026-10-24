"""Build the guesser's vocabulary: the most frequent token ids in Flint's generated answers, with Flint's control tokens
forced in, and report its coverage per prompt class and on held-out transcripts.

usage: python vocab.py OUT.npy GEN.jsonl [GEN.jsonl ...] [--size 32000] [--test TEST_GEN.jsonl] [--force IDS]
OUT.npy: int64 array of target token ids, sorted by frequency (draft id = index). --force: comma-separated ids to
include whatever their count (by default Flint's ids 150000-150143, its added and control tokens).
"""
import argparse
import json
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--size", type=int, default=32000)
    ap.add_argument("--test")
    ap.add_argument("--force", default="")
    args = ap.parse_args()
    counts, by_cat = Counter(), {}
    for path in args.gen:
        for line in open(path):
            r = json.loads(line)
            counts.update(r["gen_tokens"])
            by_cat.setdefault(r["category"], Counter()).update(r["gen_tokens"])
    force = [int(x) for x in args.force.split(",") if x] or list(range(150000, 150144))
    force = [t for t in force if counts[t] > 0 or not args.force]
    keep = [t for t, _ in counts.most_common()]
    chosen = list(dict.fromkeys(force + keep))[:args.size]
    chosen.sort(key=lambda t: -counts[t])
    vocab = np.array(chosen, np.int64)
    np.save(args.out, vocab)
    s = set(chosen)
    total = sum(counts.values())
    print(f"{len(counts)} distinct ids in {total} generated tokens; vocabulary {len(vocab)}")
    print(f"coverage, all: {sum(c for t, c in counts.items() if t in s) / total:.4f}")
    for cat, c in sorted(by_cat.items()):
        n = sum(c.values())
        print(f"coverage, {cat}: {sum(v for t, v in c.items() if t in s) / n:.4f} ({n} tokens)")
    if args.test:
        tc, tcat = Counter(), {}
        for line in open(args.test):
            r = json.loads(line)
            tc.update(r["gen_tokens"])
            tcat.setdefault(r["category"], Counter()).update(r["gen_tokens"])
        n = sum(tc.values())
        print(f"held-out coverage, all: {sum(v for t, v in tc.items() if t in s) / n:.4f} ({n} tokens)")
        for cat, c in sorted(tcat.items()):
            m = sum(c.values())
            print(f"held-out coverage, {cat}: {sum(v for t, v in c.items() if t in s) / m:.4f}")


if __name__ == "__main__":
    main()
