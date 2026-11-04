"""Copy spans as text, for reading (transcript text: the output is private, keep it out of the repository).

    WORK=/path/to/work VOCAB=/path/vocab.gguf python3 samples.py [N]

Writes $WORK/samples.txt: the N longest copy spans (min length 8, prompt and own earlier text as source) and N random
ones, with the region they sit in, their length and the tokens before them.
"""
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import covered_flags, load_vocab, match_profile, runs  # noqa: E402


def main():
    work = os.environ["WORK"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    table = load_vocab(os.environ["VOCAB"])
    spans = []
    with gzip.open(os.path.join(work, "prep", "tokens.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            g = r["out"]
            pl, _ = match_profile(r["ctx"], g, tail=16, own=True)
            for s, ln in runs(covered_flags(pl, 8)):
                spans.append((ln, r["id"], r["rnames"][r["reg"][s]], s, g[max(0, s - 6):s], g[s:s + ln]))

    def text(ids):
        return b"".join(table[t] for t in ids).decode("utf-8", "replace")

    rng = random.Random(1)
    pick = sorted(spans, reverse=True)[:n] + rng.sample(spans, min(n, len(spans)))
    with open(os.path.join(work, "samples.txt"), "w") as f:
        for ln, rid, region, s, before, span in pick:
            f.write("=== %s  %s  starts at token %d  length %d\n--- before: %r\n%s\n\n" % (
                rid, region, s, ln, text(before), text(span[:600])))
    print("%d spans; wrote %d samples" % (len(spans), len(pick)))


if __name__ == "__main__":
    main()
