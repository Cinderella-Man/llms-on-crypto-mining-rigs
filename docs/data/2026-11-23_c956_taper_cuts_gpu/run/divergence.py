# C956: where two passes' greedy replies to the sized prompts part, and by how much. For each request of a pair of
# passes (client.py's sizes.jsonl of two loads or passes): whether the prompt call's pieces are the same (the
# "ubatch taper:" lines, matched to the requests in order as compare.py does; the continuation and the edit also
# count the call that built the cache they read on), the largest difference of a top-3
# log-probability over the reply tokens before the first different token (the same context in both), the first
# different token, and there the two candidates' log-probabilities in each pass (top-1 minus the other pass's token).
# Usage: python3 -I divergence.py <load dir> <run> <reference> [<run> <reference> ...] > divergence.tsv
#   a run or reference is <load>/<pass>, e.g. head/v5c head/srv
import json
import os
import re
import sys

ld = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$")
cache = {}


def load(name):
    lo, ps = name.split("/")
    if lo not in cache:
        rows = [json.loads(x) for x in open(os.path.join(ld, lo, "sizes.jsonl"))]
        with open(os.path.join(ld, lo, "server.log"), errors="replace") as f:
            lines = f.read().splitlines()
        s = int(open(os.path.join(ld, lo, "split.txt")).read().split()[0])
        calls, pend = [], []
        for line in lines[:s]:
            line = ansi.sub("", line)
            m = pat.search(line)
            if m:
                pend.append(f"{m.group(1)}: {m.group(5).strip()}")
            elif "prompt eval time =" in line:
                calls.append(pend)
                pend = []
        for i, r in enumerate(rows):
            r["calls"] = " | ".join(calls[i]) if i < len(calls) else "?"
        cache[lo] = rows
    return {r["req"]: r for r in cache[lo] if r["pass"] == ps}


def lp(tops, tok):
    for i, v in tops:
        if i == tok:
            return v
    return None


print("run\treference\treq\tpieces\tfirst_different_token\tlargest_logprob_difference_before\t"
      "reference_top1_minus_run_token\trun_top1_minus_reference_token\tfirst_token_top3_logprob_difference")
for run, ref in pairs:
    a, b = load(run), load(ref)
    for req in b:
        x, y = a.get(req), b.get(req)
        if x is None or "error" in x or "error" in y:
            print(f"{run}\t{ref}\t{req}\t-\tERROR\t\t\t\t")
            continue
        dep = {"cont-577": "fresh-577", "edit-1534": "fresh-1534"}.get(req)
        same_pieces = "same" if x["calls"] == y["calls"] and (dep is None or a[dep]["calls"] == b[dep]["calls"]) else "differ"
        ta, tb = x["tokens"], y["tokens"]
        k = next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), None)
        upto = len(ta) if k is None else k
        dmax = 0.0
        for (ia, pa), (ib, pb) in zip(x["probs"][:upto + (0 if k is None else 1)], y["probs"][:upto + (0 if k is None else 1)]):
            for (i1, v1), (i2, v2) in zip(pa, pb):
                if i1 == i2:
                    dmax = max(dmax, abs(v1 - v2))
        first = max((abs(v1 - v2) for (i1, v1), (i2, v2) in zip(x["probs"][0][1], y["probs"][0][1]) if i1 == i2),
                    default=float("nan"))
        if k is None:
            print(f"{run}\t{ref}\t{req}\t{same_pieces}\tnone of {len(tb)}\t{dmax:.3g}\t-\t-\t{first:.3g}")
            continue
        g_ref = lp(y["probs"][k][1], tb[k]) - (lp(y["probs"][k][1], ta[k]) if lp(y["probs"][k][1], ta[k]) is not None else float("nan"))
        g_run = lp(x["probs"][k][1], ta[k]) - (lp(x["probs"][k][1], tb[k]) if lp(x["probs"][k][1], tb[k]) is not None else float("nan"))
        print(f"{run}\t{ref}\t{req}\t{same_pieces}\t{k}\t{dmax:.3g}\t{g_ref:.3g}\t{g_run:.3g}\t{first:.3g}")
