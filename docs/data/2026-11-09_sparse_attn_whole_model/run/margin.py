#!/usr/bin/env python3
# margin.py RESP_DIR SET DEPTH ARM_A ARM_B [PROBE_A PROBE_B]: where two arms' 128-token texts part, and how close the
# two candidates were. Finds the first generated token that differs between ARM_A's and ARM_B's texts, then reads
# that position in the two arms' probes (run with PROBE_TOKENS=128, so each token has its top 10 log-probabilities;
# PROBE_A and PROBE_B name the arms whose probes to read when they are not ARM_A and ARM_B). Prints, per probe: whether
# its tokens up to that position are the text's, the log-probability of each arm's token there, and the lead of one
# over the other. A lead far below the differences that two correct computations show at that depth is a near-tie.
import json, sys

resp, name, depth, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
pa, pb = (sys.argv[6], sys.argv[7]) if len(sys.argv) > 7 else (a, b)


def load(arm, probe=False):
    return json.load(open(f"{resp}/{'probe-' if probe else ''}{name}-{arm}-{depth}.json"))


ta, tb = load(a)["tokens"], load(b)["tokens"]
i = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), None)
if i is None:
    sys.exit(f"{name} {depth}: {a} and {b} write the same {len(ta)} tokens")
print(f"{name} {depth}: {a} and {b} part at generated token {i + 1} of {len(ta)}: {a} writes {ta[i]}, {b} writes {tb[i]}")
for arm, text in ((pa, ta), (pb, tb)):
    p = load(arm, probe=True)["probs"]
    if len(p) <= i:
        print(f"  probe {arm}: only {len(p)} tokens")
        continue
    same = [x["id"] for x in p[:i]] == text[:i]
    c = {x["id"]: (x["logprob"], x["token"]) for x in p[i]["top_logprobs"]}
    la, lb = c.get(ta[i], (None, ""))[0], c.get(tb[i], (None, ""))[0]
    lead = "" if la is None or lb is None else f", lead of {ta[i]} over {tb[i]}: {la - lb:+.4f}"
    print(f"  probe {arm}: the same {i} tokens before it: {same}; writes {p[i]['id']} ({p[i]['token']!r}); "
          f"log-probability of {ta[i]} ({c.get(ta[i], (None, '?'))[1]!r}): {la}, of {tb[i]} "
          f"({c.get(tb[i], (None, '?'))[1]!r}): {lb}{lead}")
    print("    top 5: " + ", ".join(f"{t} {v[1]!r} {v[0]:.4f}" for t, v in sorted(c.items(), key=lambda kv: -kv[1][0])[:5]))
