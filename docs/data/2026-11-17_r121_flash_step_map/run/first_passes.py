#!/usr/bin/env python3
"""first_passes.py: the first request at a depth against its repeat, launch group by launch group, from two pieces
of a group-mode GGML_CUDA_PROFILE trace (split_trace.py's d<depth>p1.csv and d<depth>p2.csv) (R121, Q1133).

  python3 -I first_passes.py P1.csv P2.csv [--event-us 7.6] [--other-regex mtp_] [--prompt-passes 1] [--top 15]
      [--tsv labels.tsv] > first_passes.md

A main pass is the j-th evaluation of the model's graph on every card (an evaluation with a launch group labelled
--other-regex is the MTP draft's and is left out). Each request's first --prompt-passes main passes read its prompt
(the deep lane's pass 1 reads 64 tokens in one micro-batch, pass 2 reads 4); the next one is the first verify pass and
the request's last main pass verifies the reply's last tokens. Per request: every main pass's GPU time on all cards
(the profiler's --event-us subtracted per launch group). Then the GPU time by launch-group label (node numbers
dropped) of four passes of both requests side by side: the first pass after the prompt, the early pass (pass 1's
slowest of its first four passes after the prompt, and pass 2's pass at the same index), a typical pass (the median
pass by GPU time among passes 5 to -5) and the last pass; sorted by the largest difference of the first, early or
last pass; how many labels grew; and per card.
Standard library only.
"""
import argparse
import csv
import re
import statistics
from collections import OrderedDict, defaultdict


def norm(label):
    return re.sub(r"\bnode_\d+", "node", label)


def passes_of(path, other_rx, event_us):
    ev = OrderedDict()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                key, t, us = (int(r["device"]), int(r["eval"])), float(r["host_ms"]), float(r["us"])
                int(r["group"])
            except (KeyError, TypeError, ValueError):
                continue
            if r.get("label") is None:
                continue
            e = ev.setdefault(key, {"d": key[0], "t": t, "labels": defaultdict(float), "other": False, "us": 0.0})
            e["labels"][norm(r["label"])] += (us - event_us) / 1000.0
            e["us"] += (us - event_us) / 1000.0
            if other_rx.search(r["label"]):
                e["other"] = True
    devs = sorted({e["d"] for e in ev.values()})
    model = {d: sorted((e for e in ev.values() if e["d"] == d and not e["other"]), key=lambda e: e["t"]) for d in devs}
    n = min(len(v) for v in model.values())
    return devs, [[model[d][j] for d in devs] for j in range(n)]


def labels(p):
    out = defaultdict(float)
    for e in p:
        for k, v in e["labels"].items():
            out[k] += v
    return out


def total(p):
    return sum(e["us"] for e in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p1")
    ap.add_argument("p2")
    ap.add_argument("--event-us", type=float, default=7.6)
    ap.add_argument("--other-regex", default="mtp_")
    ap.add_argument("--prompt-passes", type=int, default=1)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    rx = re.compile(a.other_regex)
    reqs = {}
    for name, path in (("pass 1", a.p1), ("pass 2", a.p2)):
        devs, ps = passes_of(path, rx, a.event_us)
        k = a.prompt_passes
        mid = ps[k + 4:-4] if len(ps) > k + 12 else ps[k:]
        typ = sorted(mid, key=total)[len(mid) // 2]
        reqs[name] = {"devs": devs, "passes": ps, "first": ps[k], "typical": typ, "last": ps[-1]}
        print(f"**{name}** ({path.rsplit('/', 1)[-1]}): {len(ps)} main passes on cards {devs[0]} to {devs[-1]}; GPU time "
              f"of every main pass, ms (all cards): " + ", ".join(f"{total(p):.1f}" for p in ps))
        print()
    # the early pass: pass 1's slowest of its first four passes after the prompt, and pass 2's pass at that index
    k = a.prompt_passes
    p1 = reqs["pass 1"]["passes"]
    ei = max(range(k, min(k + 4, len(p1))), key=lambda i: total(p1[i]))
    for name in reqs:
        reqs[name]["early"] = reqs[name]["passes"][min(ei, len(reqs[name]["passes"]) - 1)]
    print(f"The early pass: main pass {ei} of each request (pass 1's slowest of passes {k} to {k + 3}).")
    print()
    works = ("first", "early", "typical", "last")
    cols = [(r, w) for w in works for r in ("pass 1", "pass 2")]
    L = {c: labels(reqs[c[0]][c[1]]) for c in cols}
    keys = set().union(*L.values())
    diff = lambda w, k: L[("pass 1", w)].get(k, 0.0) - L[("pass 2", w)].get(k, 0.0)
    rows = sorted(keys, key=lambda k: -max(diff("first", k), diff("early", k), diff("last", k)))
    head = ["launch group (label)"] + [f"{r}, {w}" for r, w in cols] + [f"{w}: 1 − 2" for w in ("first", "early", "last")]
    print("**By label, ms** (sorted by the largest difference of the first, early or last pass)")
    print()
    print("| " + " | ".join(head) + " |")
    print("|---" + "|---:" * (len(head) - 1) + "|")
    for kk in rows[:a.top]:
        v = [L[c].get(kk, 0.0) for c in cols]
        print(f"| `{kk[:90]}` | " + " | ".join(f"{x:.3f}" for x in v) + " | " +
              " | ".join(f"{diff(w, kk):+.3f}" for w in ("first", "early", "last")) + " |")
    tot = {c: sum(L[c].values()) for c in cols}
    print(f"| **every label** | " + " | ".join(f"**{tot[c]:.2f}**" for c in cols) + " | " +
          " | ".join(f"**{tot[('pass 1', w)] - tot[('pass 2', w)]:+.2f}**" for w in ("first", "early", "last")) + " |")
    print()
    # how the difference spreads: the share of launch groups that grew
    for w in ("first", "early", "last"):
        a1, a2 = L[("pass 1", w)], L[("pass 2", w)]
        common = [kk for kk in a1 if kk in a2]
        grew = [kk for kk in common if a1[kk] > a2[kk] * 1.2 + 0.001]
        d = sum(a1[kk] - a2[kk] for kk in common)
        top10 = sum(sorted((a1[kk] - a2[kk] for kk in common), reverse=True)[:10])
        print(f"- the {w} pass: {len(common)} labels in both; {len(grew)} of them over 1.2 times pass 2's; the ten "
              f"largest differences hold {top10:.2f} of the {d:.2f} ms")
    print()
    print("**Per card, ms**")
    print()
    devs = reqs["pass 1"]["devs"]
    print("| card | " + " | ".join(f"{r}, {w}" for r, w in cols) + " |")
    print("|---" + "|---:" * len(cols) + "|")
    for i, d in enumerate(devs):
        print(f"| {d} | " + " | ".join(f"{reqs[r][w][i]['us']:.2f}" for r, w in cols) + " |")
    print()
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("label\t" + "\t".join(f"{r.replace(' ', '')}_{w}" for r, w in cols) + "\n")
            for kk in rows:
                f.write(kk + "\t" + "\t".join(f"{L[c].get(kk, 0.0):.4f}" for c in cols) + "\n")


if __name__ == "__main__":
    main()
