#!/usr/bin/env python3
# compare.py RESP_DIR OUT_DIR [ARM=AGAINST ...]: Q1171's tables from the files whole_requests.py wrote. Each arm is
# compared with the arm it should or should not equal: on with off, fault with on, offfault with off, and any pair
# given on the command line (on.s1=on).
#   texts.tsv  per set, depth and arm, from the 128-token requests: prompt tokens, the prompt's md5, generated tokens,
#              drafted and accepted tokens, the text's md5, the arm it is compared with, and "same" or the first
#              generated token (counted from 1) that differs
#   probe.tsv  per set, depth and arm, from the 16-token probes: how many of the 16 tokens are the same from the
#              start; then over the positions with the same context (up to and including the first differing token):
#              the candidates both arms list (of the top 10 per position), how many of them have the same
#              log-probability to the last digit, the largest difference in log-probability, and the smallest lead of
#              a chosen token over the next candidate in the arm compared with, which is what a difference has to
#              beat to move a token (candidates above the chosen token are the end-of-text tokens the requests ban)
import glob, hashlib, json, os, sys

resp, out = sys.argv[1], sys.argv[2]
main, probe = {}, {}
for f in sorted(glob.glob(resp + "/*.json")):
    r = json.load(open(f))
    (probe if os.path.basename(f).startswith("probe-") else main)[(r["set"], r["depth"], r["arm"])] = r
AGAINST = {"on": "off", "fault": "on", "offfault": "off"}
AGAINST.update(p.split("=") for p in sys.argv[3:])
order = {"off": 0, "on": 1, "fault": 2, "offfault": 3}
sets = sorted({k[0] for k in main}, key=lambda s: s != "plain")


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def top(p):
    # one generated token's candidates: {token id: log-probability}
    return {c["id"]: c["logprob"] for c in p.get("top_logprobs", [])}


with open(out + "/texts.tsv", "w") as f:
    f.write("set\tdepth\tarm\tprompt_tokens\tprompt_md5\tgen_tokens\tdraft_tokens\tdraft_accepted\ttext_md5\t"
            "against\tverdict\n")
    for s in sets:
        for depth in sorted({k[1] for k in main if k[0] == s}):
            for arm in sorted({k[2] for k in main if k[:2] == (s, depth)}, key=lambda a: (order.get(a, 9), a)):
                r, ref = main[(s, depth, arm)], main.get((s, depth, AGAINST.get(arm)))
                c = r["counts"]
                if ref is None:
                    v = ""
                elif r["prompt_md5"] != ref["prompt_md5"]:
                    v = "another prompt"
                else:
                    i = first_diff(r["tokens"], ref["tokens"])
                    v = "same" if i is None and r["content"] == ref["content"] else f"differs from token {i + 1}"
                f.write(f"{s}\t{depth}\t{arm}\t{c['prompt_n']}\t{r['prompt_md5']}\t{c['predicted_n']}\t"
                        f"{c.get('draft_n', '')}\t{c.get('draft_n_accepted', '')}\t"
                        f"{hashlib.md5(r['content'].encode()).hexdigest()}\t{AGAINST.get(arm, '') if ref else ''}\t{v}\n")

with open(out + "/probe.tsv", "w") as f:
    f.write("set\tdepth\tarm\tagainst\tprobe_tokens\tsame_from_start\tpositions_compared\tcandidates_compared\t"
            "identical_logprobs\tmax_abs_dlogprob\tsmallest_lead\n")
    for s in sets:
        for depth in sorted({k[1] for k in probe if k[0] == s}):
            for arm in sorted({k[2] for k in probe if k[:2] == (s, depth) and k[2] in AGAINST}, key=lambda a: (order.get(a, 9), a)):
                a, b = probe[(s, depth, arm)], probe.get((s, depth, AGAINST[arm]))
                if b is None or not a.get("probs") or not b.get("probs"):
                    continue
                ta, tb = [p["id"] for p in a["probs"]], [p["id"] for p in b["probs"]]
                i = first_diff(ta, tb)
                same = min(len(ta), len(tb)) if i is None else i
                n = min(len(ta), len(tb), same + 1)
                dmax, lead, cands, ident = 0.0, float("inf"), 0, 0
                for j in range(n):
                    ca, cb = top(a["probs"][j]), top(b["probs"][j])
                    for t in ca.keys() & cb.keys():
                        cands += 1
                        ident += ca[t] == cb[t]
                        dmax = max(dmax, abs(ca[t] - cb[t]))
                    chosen = b["probs"][j]["logprob"]
                    below = [v for t, v in cb.items() if v < chosen and t != b["probs"][j]["id"]]
                    if below:
                        lead = min(lead, chosen - max(below))
                f.write(f"{s}\t{depth}\t{arm}\t{AGAINST[arm]}\t{len(ta)}\t{same}\t{n}\t{cands}\t{ident}\t{dmax:.3g}\t"
                        f"{lead:.3g}\n")

for name in ("texts.tsv", "probe.tsv"):
    print(open(out + "/" + name).read())
