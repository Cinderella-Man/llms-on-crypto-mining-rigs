#!/usr/bin/env python3
"""Where two greedy runs of greedy_check.py differ: per prompt, the generated tokens whose top-10 probabilities differ,
the tokens that differ, and whether the text is the same.

  greedy_diff.py OUT/greedy-ref.json OUT/greedy-TAG.json
"""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for i, (x, y) in enumerate(zip(a, b)):
    pa = x["response"]["completion_probabilities"]
    pb = y["response"]["completion_probabilities"]
    diff = [k for k, (u, v) in enumerate(zip(pa, pb)) if u != v]
    tokdiff = [k for k, (u, v) in enumerate(zip(pa, pb)) if u.get("token") != v.get("token")]
    print(f"prompt {i}: {len(diff)} of {len(pa)} generated tokens with different probabilities "
          f"(first {diff[0] if diff else None}); {len(tokdiff)} with a different token "
          f"(first {tokdiff[0] if tokdiff else None}); text identical: "
          f"{x['response']['content'] == y['response']['content']}")
    if diff:
        u, v = pa[diff[0]], pb[diff[0]]
        top = lambda t: [round(p.get("logprob", p.get("prob", 0.0)), 6) for p in t.get("top_logprobs", t.get("top_probs", []))[:3]]
        print(f"  first difference: token {u.get('token')!r} {top(u)} against {v.get('token')!r} {top(v)}")
