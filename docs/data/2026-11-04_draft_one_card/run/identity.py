#!/usr/bin/env python3
"""identity.py: whether a switch drafts the same tokens on the cards (Q961).

  python3 -I identity.py REF_REPLY.json REPLY.json [REPLY.json ...]

Each file is llama-server's reply to the same greedy chat request (run/greedy_request.json) from one load. Prints, per
reply, whether its text and its draft counts (timings.draft_n, timings.draft_n_accepted) equal the reference's: with
greedy sampling the drafts decide the counts, so equal counts and text mean the switch drafted the same tokens.
Exit status 0 when all equal.
"""
import json
import sys


def key(path):
    with open(path) as f:
        r = json.load(f)
    t = r.get("timings", {})
    return r["choices"][0]["message"].get("content"), t.get("draft_n"), t.get("draft_n_accepted"), t.get("predicted_n")


def main():
    ref = key(sys.argv[1])
    print(f"{sys.argv[1]}: draft_n {ref[1]}, accepted {ref[2]}, predicted {ref[3]} (the reference)")
    bad = 0
    for p in sys.argv[2:]:
        k = key(p)
        same = k == ref
        bad += not same
        print(f"{p}: draft_n {k[1]}, accepted {k[2]}, predicted {k[3]}: {'the same text and counts' if same else 'DIFFERS'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
