"""Decode-path drift between two greedy_requests.py outputs: over the tokens both builds generated alike, the
difference in the chosen token's log-probability (probs requests), per prompt and over all. Standard library only.
usage: decode_drift.py REF.jsonl OTHER.jsonl"""
import json, statistics, sys
def load(p):
    return {(r["name"], r["mode"]): r for r in map(json.loads, filter(str.strip, open(p)))}
a, b = load(sys.argv[1]), load(sys.argv[2])
allv = []
for (n, m), ra in a.items():
    if m != "probs":
        continue
    rb = b[(n, m)]
    ta, tb = ra["tokens"], rb["tokens"]
    k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    d = [abs(ra["probs"][i][0][1] - rb["probs"][i][0][1]) for i in range(k)]
    allv += d
    print(f"{n}: shared {k}, mean |d logprob| {statistics.mean(d):.4f}, median {statistics.median(d):.4f}, "
          f"p90 {sorted(d)[int(0.9 * len(d))]:.4f}, max {max(d):.4f}")
print(f"all: {len(allv)} tokens, mean {statistics.mean(allv):.4f}, median {statistics.median(allv):.4f}")
