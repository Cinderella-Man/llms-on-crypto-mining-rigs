"""Build the guesser's vocabulary: the most frequent token ids in Flint's generated answers, with Flint's control tokens
forced in, and report its coverage per prompt class and on held-out transcripts.

usage: python vocab.py OUT.npy GEN.jsonl [GEN.jsonl ...] [--size 32000] [--test TEST_GEN.jsonl] [--force IDS]
                       [--only IDS_FILE]
OUT.npy: int64 array of target token ids, sorted by frequency (draft id = index). --force: comma-separated ids to
include whatever their count (by default Flint's ids 150000-150143, its added and control tokens). Agent sessions'
rows (sessions_seq.jsonl, with gen_ranges) count only their generated spans, not the tool results.
"""
import argparse
import json
from collections import Counter

import numpy as np


def generated(r):
    """The tokens Flint generated in a row: gen_tokens, or for an agent session (gen_ranges over the whole sequence)
    the tokens of its spans, without the tool results."""
    if "gen_ranges" not in r:
        return r["gen_tokens"]
    seq = r["prompt_tokens"] + r["gen_tokens"]
    return [t for s, e in r["gen_ranges"] for t in seq[s:e]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--size", type=int, default=32000)
    ap.add_argument("--test")
    ap.add_argument("--force", default="")
    ap.add_argument("--only", help="file of answer ids (one per line): count only these answers (e.g. licence-clean ones)")
    args = ap.parse_args()
    only = {x.strip() for x in open(args.only) if x.strip()} if args.only else None
    counts, by_cat = Counter(), {}
    for path in args.gen:
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:   # a file still being written
                continue
            if only is not None and r["id"] not in only:
                continue
            gen = generated(r)
            counts.update(gen)
            by_cat.setdefault(r["category"], Counter()).update(gen)
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
            tc.update(generated(r))
            tcat.setdefault(r["category"], Counter()).update(generated(r))
        n = sum(tc.values())
        print(f"held-out coverage, all: {sum(v for t, v in tc.items() if t in s) / n:.4f} ({n} tokens)")
        for cat, c in sorted(tcat.items()):
            m = sum(c.values())
            print(f"held-out coverage, {cat}: {sum(v for t, v in c.items() if t in s) / m:.4f}")


if __name__ == "__main__":
    main()
