#!/usr/bin/env python3
# steps.py RESP_DIR SET DEPTH N ARM=AGAINST ...: Q1171's table of how far two arms' probes are apart token by token.
# Per pair one line: for each of the first N generated tokens, the largest difference in log-probability of a
# candidate both arms list in their top 10, or "-" once the tokens before it differ (the contexts are then not the
# same). Token 1 comes from the prompt's last batch; token 2 is the first one decoded a token at a time.
import json, sys

resp, name, depth, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def load(arm):
    return json.load(open(f"{resp}/probe-{name}-{arm}-{depth}.json"))["probs"]


print("set\tdepth\tarm\tagainst\t" + "\t".join(f"token_{j + 1}" for j in range(n)))
for pair in sys.argv[5:]:
    arm, against = pair.split("=")
    a, b = load(arm), load(against)
    row = []
    for j in range(min(n, len(a), len(b))):
        if [p["id"] for p in a[:j]] != [p["id"] for p in b[:j]]:
            row.append("-")
            continue
        ca = {c["id"]: c["logprob"] for c in a[j]["top_logprobs"]}
        cb = {c["id"]: c["logprob"] for c in b[j]["top_logprobs"]}
        row.append(f"{max(abs(ca[t] - cb[t]) for t in ca.keys() & cb.keys()):.3g}")
    print(f"{name}\t{depth}\t{arm}\t{against}\t" + "\t".join(row))
