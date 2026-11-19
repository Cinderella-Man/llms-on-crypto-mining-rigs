#!/usr/bin/env python3
"""compare.py: R197's arm against arm. Reads R176's step_map.py output (--tsv, long form) once for arm base and once
for arm ro (each arm's group-profile loads named w<k>g1, w<k>g2 there), and prints, per verify width and depth, each
load by value:

  1. the verify pass (the sum of its launch groups, nine cards), base against ro, each arm's spread (its two loads
     apart), the paired differences (ro's first load less base's first, the second less the second) and the class:
     gain when both differences are below zero and each is larger than both arms' spreads, loss when both are above
     zero and larger than both spreads, negligible otherwise
  2. the pass by family of launch groups (step_map.py's families), base against ro
  3. Ornith's dense Q5_0 and Q8_0 products (launch groups with a plain MUL_MAT of a q5_0 or q8_0 tensor), their sum
     and the launch groups that moved most
  4. the one-token step on this boot (the host-profile load, w1h1) beside R176's on v4.0
  5. the break-even a draft needs (est.): the step of k tokens over the one-token step of this boot, the step built
     from this run's verify pass (group profile, put on the host profile's footing by R176's ratio of the two at the
     same width) plus R176's parts outside the verify pass's GPU time (its rest and the draft's span, the draft's GPU
     time scaled by this boot's one-token GPU time over R176's); and the same for a verify pass alone
  6. the replies: each load's token ids (md5) against the other arm's, the one-token load's and R176's, and the first
     token where two differ (from the deep lane's <load>.jsonl)

  python3 -I compare.py --base map_b_p2.tsv --ro map_r_p2.tsv --request p2 --requests requests.csv --replies OUT \\
      --r176 <R176>/map.tsv.gz --r176-requests <R176>/requests.csv [--tsv compare_p2.tsv] > compare_p2.md
Standard library only.
"""
import argparse
import csv
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

ARMS = ("base", "ro")


def read_map(path):
    op = gzip.open if path.endswith(".gz") else open
    m = defaultdict(dict)  # (table, width, depth, request, load) -> name -> value
    with op(path, "rt", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                v = float(r["value"])
            except ValueError:
                continue
            m[(r["table"], int(r["width"]), int(r["depth"]), r["request"], r["load"])][r["name"]] = v
    return m


def f2(x):
    return "—" if x is None or x != x else f"{x:.2f}"


def vals(xs):
    return ", ".join(f2(x) for x in xs) if xs else "—"


def pct(a, b):
    return "—" if not a or not b else f"{(statistics.fmean(b) / statistics.fmean(a) - 1) * 100:+.2f} %"


def classify(b, r):
    """b, r: the two loads of each arm in run order -> (paired differences, spreads, class)"""
    if len(b) < 2 or len(r) < 2:
        return [], [], "one load of an arm: no class"
    d = [y - x for x, y in zip(b, r)]
    sp = [abs(b[0] - b[1]), abs(r[0] - r[1])]
    lim = max(sp)
    if all(x < 0 for x in d) and min(abs(x) for x in d) > lim:
        return d, sp, "gain"
    if all(x > 0 for x in d) and min(abs(x) for x in d) > lim:
        return d, sp, "loss"
    return d, sp, "negligible"


def dense_q5q8(label):
    ops, _, rest = label.partition(" ")
    return "MUL_MAT" in ops.split("+") and re.search(r":(q5_0|q8_0):", rest) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ro", required=True)
    ap.add_argument("--request", default="p2")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--r176", required=True)
    ap.add_argument("--r176-requests", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    R = a.request
    maps = {"base": read_map(a.base), "ro": read_map(a.ro)}
    r176 = read_map(a.r176)
    out = []

    def keep(table, w, d, arm, load, name, v):
        out.append((table, w, d, f"d{d}{R}", arm, load, name, v))

    keys = set()
    for m in maps.values():
        keys |= {(w, d) for (t, w, d, q, l) in m if t == "families" and q == f"d{d}{R}" and w > 1}
    widths = sorted({w for w, _ in keys})
    depths = sorted({d for _, d in keys})

    def loads(arm, w, d):
        return sorted(l for (t, ww, dd, q, l) in maps[arm] if t == "families" and ww == w and dd == d
                      and q == f"d{d}{R}")

    def fam(arm, w, d, load):
        return maps[arm].get(("families", w, d, f"d{d}{R}", load), {})

    def labels(arm, w, d, load):
        return maps[arm].get(("verify_labels", w, d, f"d{d}{R}", load), {})

    def passes(arm, w, d):
        return [sum(fam(arm, w, d, l).values()) for l in loads(arm, w, d)]

    P = print
    names = {"p1": "the first request at the depth", "p2": "the repeat"}
    P(f"# R197: arm base (Ornith's v5 preset) against arm ro (+ the read-once switches), request {R} "
      f"({names.get(R, R)}); each load by value, in run order")
    P()

    # 1. the pass
    P("## 1. The verify pass, GPU ms a pass, nine cards (the group profile's launch groups summed)")
    P()
    P("| width | depth | base | ro | ro − base, paired | change of the means | spread base, ro | class |")
    P("|---:|---:|---:|---:|---:|---:|---:|---|")
    for w in widths:
        for d in depths:
            b, r = passes("base", w, d), passes("ro", w, d)
            dd, sp, cl = classify(b, r)
            P(f"| {w} | {d:,} | {vals(b)} | {vals(r)} | {vals(dd)} | {pct(b, r)} | {vals(sp)} | **{cl}** |")
            for arm, xs in (("base", b), ("ro", r)):
                for l, x in zip(loads(arm, w, d), xs):
                    keep("pass", w, d, arm, l, "the pass", x)
            for i, x in enumerate(dd):
                keep("pass", w, d, "ro-base", f"pair{i + 1}", "the pass", x)
            keep("class", w, d, "ro-base", "", cl, 0.0)
    P()

    # 2. families
    P("## 2. The pass by family of launch groups, ms a pass, nine cards (families that moved 0.05 ms or more)")
    P()
    for w in widths:
        for d in depths:
            bl, rl = loads("base", w, d), loads("ro", w, d)
            fams = []
            for arm, ls in (("base", bl), ("ro", rl)):
                for l in ls:
                    for k in fam(arm, w, d, l):
                        if k not in fams:
                            fams.append(k)
            rows = []
            for k in fams:
                b = [fam("base", w, d, l).get(k, 0.0) for l in bl]
                r = [fam("ro", w, d, l).get(k, 0.0) for l in rl]
                dd = [y - x for x, y in zip(b, r)]
                for arm, ls, xs in (("base", bl, b), ("ro", rl, r)):
                    for l, x in zip(ls, xs):
                        keep("family", w, d, arm, l, k, x)
                if dd and max(abs(x) for x in dd) >= 0.05:
                    rows.append((statistics.fmean(dd), k, b, r, dd))
            rows.sort()
            P(f"### {w} tokens, depth {d:,}")
            P()
            P("| family | base | ro | ro − base, paired |")
            P("|---|---:|---:|---:|")
            for _, k, b, r, dd in rows:
                P(f"| {k} | {vals(b)} | {vals(r)} | {vals(dd)} |")
            if not rows:
                P("| (none moved 0.05 ms) | | | |")
            P()

    # 3. the dense Q5_0 and Q8_0 products
    P("## 3. The dense Q5_0 and Q8_0 products (launch groups with a plain MUL_MAT of a q5_0 or q8_0 tensor), ms a pass")
    P()
    P("| width | depth | base | ro | ro − base, paired | change of the means | groups |")
    P("|---:|---:|---:|---:|---:|---:|---:|")
    for w in widths:
        for d in depths:
            bl, rl = loads("base", w, d), loads("ro", w, d)
            b = [sum(v for k, v in labels("base", w, d, l).items() if dense_q5q8(k)) for l in bl]
            r = [sum(v for k, v in labels("ro", w, d, l).items() if dense_q5q8(k)) for l in rl]
            n = len({k for l in bl for k in labels("base", w, d, l) if dense_q5q8(k)})
            dd = [y - x for x, y in zip(b, r)]
            P(f"| {w} | {d:,} | {vals(b)} | {vals(r)} | {vals(dd)} | {pct(b, r)} | {n} |")
            for arm, ls, xs in (("base", bl, b), ("ro", rl, r)):
                for l, x in zip(ls, xs):
                    keep("dense_q5q8", w, d, arm, l, "sum", x)
    P()
    for w in widths:
        for d in depths[:1] + depths[-1:] if len(depths) > 1 else depths:
            bl, rl = loads("base", w, d), loads("ro", w, d)
            labs = set()
            for l in bl:
                labs |= set(labels("base", w, d, l))
            for l in rl:
                labs |= set(labels("ro", w, d, l))
            rows = []
            for k in labs:
                b = [labels("base", w, d, l).get(k, 0.0) for l in bl]
                r = [labels("ro", w, d, l).get(k, 0.0) for l in rl]
                dd = [y - x for x, y in zip(b, r)]
                if dd:
                    rows.append((-abs(statistics.fmean(dd)), k, b, r, dd))
            rows.sort()
            P(f"### {w} tokens, depth {d:,}: the {a.top} launch groups that moved most")
            P()
            P("| launch group (label) | base | ro | ro − base, paired |")
            P("|---|---:|---:|---:|")
            for _, k, b, r, dd in rows[:a.top]:
                P(f"| `{k}` | {vals(b)} | {vals(r)} | {vals(dd)} |")
            P()
            if d == depths[0] and R == "p2":
                for _, k, b, r, dd in rows:
                    for arm, ls, xs in (("base", bl, b), ("ro", rl, r)):
                        for l, x in zip(ls, xs):
                            keep("label", w, d, arm, l, k, x)

    # 4. the one-token step
    def split(m, w, d, load):
        return m.get(("split", w, d, f"d{d}{R}", load), {})

    P("## 4. The one-token step, ms (the host-profile load; nothing synchronized)")
    P()
    P("| depth | this boot (w1h1): step | its GPU time, nine cards | the rest | R176 on v4.0 at stock (w1h1, w1h2): step | GPU time | change of the step |")
    P("|---:|---:|---:|---:|---:|---:|---:|")
    s1, g1, g176 = {}, {}, {}
    for d in depths:
        s = split(maps["base"], 1, d, "w1h1")
        o = [split(r176, 1, d, l) for l in ("w1h1", "w1h2")]
        s1[d], g1[d] = s.get("step"), s.get("verify pass: GPU time, nine cards")
        g176[d] = statistics.fmean([x["verify pass: GPU time, nine cards"] for x in o if x])
        st176 = [x.get("step") for x in o if x]
        P(f"| {d:,} | {f2(s1[d])} | {f2(g1[d])} | {f2(s.get('verify pass: the rest'))} | {vals(st176)} | "
          f"{vals([x.get('verify pass: GPU time, nine cards') for x in o if x])} | {pct(st176, [s1[d]] if s1[d] else [])} |")
        for k in ("step", "verify pass: GPU time, nine cards", "verify pass: the rest"):
            if s.get(k) is not None:
                keep("one_token", 1, d, "base", "w1h1", k, s[k])
    P()

    # tokens a step of this run's replies
    reqs = list(csv.DictReader(open(a.requests, newline="")))
    tps = defaultdict(list)  # (arm, w, d) -> tokens a step per load
    for r in reqs:
        if r["kind"] != "decode" or f"p{r['pass']}" != R:
            continue
        m = re.match(r"w(\d)([br])(\d)$", r["load"])
        if m and r["tokens_per_step"]:
            tps[("base" if m.group(2) == "b" else "ro", int(m.group(1)), int(r["depth"]))].append(
                float(r["tokens_per_step"]))

    # 5. the break-even
    P("## 5. The break-even a draft needs, tokens a step (est.)")
    P()
    P("The step of k tokens (est.) = this run's verify pass × R176's ratio of the host profile's verify GPU time to the "
      "group profile's pass at the same width + R176's verify rest + R176's draft span less its GPU time + R176's draft "
      "GPU time × (this boot's one-token GPU time / R176's). Break-even = that step / this boot's one-token step. "
      "\"Verify alone\" leaves the draft's span out: the price of a verify for any draft. Tokens a step and ms a token "
      "are this greedy reply's.")
    P()
    P("| width | depth | arm | the step (est.) | break-even | verify alone: break-even | tokens a step (this reply) | ms a token (est.) | one-token step | R176 measured: step, break-even |")
    P("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for w in widths:
        for d in depths:
            o = split(r176, w, d, f"w{w}h1")
            gp = sum(r176.get(("families", w, d, f"d{d}{R}", f"w{w}g1"), {}).values())
            if not o or not gp or not s1.get(d) or not g1.get(d):
                continue
            ratio = o["verify pass: GPU time, nine cards"] / gp
            g = g1[d] / g176[d]
            span = o.get("the draft's span", 0.0)
            dgpu = o.get("its GPU time", 0.0)
            for arm in ARMS:
                ps = passes(arm, w, d)
                if not ps:
                    continue
                v = statistics.fmean(ps) * ratio
                step = v + o["verify pass: the rest"] + (span - dgpu) + dgpu * g
                alone = v + o["verify pass: the rest"]
                t = tps.get((arm, w, d), [])
                ms = step / statistics.fmean(t) if t else None
                P(f"| {w} | {d:,} | {arm} | {f2(step)} | {f2(step / s1[d])} | {f2(alone / s1[d])} | "
                  f"{vals(t)} | {f2(ms)} | {f2(s1[d])} | {f2(o['step'])}, {f2(o['step'] / split(r176, 1, d, 'w1h1')['step'])} |")
                keep("break_even", w, d, arm, "", "step (est.)", step)
                keep("break_even", w, d, arm, "", "break-even (est.)", step / s1[d])
                keep("break_even", w, d, arm, "", "verify alone: break-even (est.)", alone / s1[d])
                if ms:
                    keep("break_even", w, d, arm, "", "ms a token, this reply (est.)", ms)
    P()

    # 6. the replies
    P("## 6. The replies (greedy, 256 tokens): md5 of the token ids, first 8 characters")
    P()
    ids = {}  # (load, depth, pass) -> ids
    for r in reqs:
        if r["kind"] == "decode":
            ids.setdefault((r["load"], int(r["depth"]), r["pass"]), None)
    for f in os.listdir(a.replies):
        if not f.endswith(".jsonl") or not re.match(r"w\d", f):
            continue
        with open(os.path.join(a.replies, f)) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("kind") == "decode":
                    ids[(rec["load"], int(rec["depth"]), str(rec["pass"]))] = rec.get("ids")
    md5 = {(r["load"], int(r["depth"]), r["pass"]): r["ids_md5"] for r in reqs if r["kind"] == "decode"}
    old = {(r["load"], int(r["depth"]), r["pass"]): r["ids_md5"]
           for r in csv.DictReader(open(a.r176_requests, newline="")) if r["kind"] == "decode"}
    p = R[1:]

    def first_diff(x, y):
        if x is None or y is None:
            return "—"
        for i, (s, t) in enumerate(zip(x, y)):
            if s != t:
                return str(i)
        return "none" if len(x) == len(y) else str(min(len(x), len(y)))

    P("| width | depth | base | ro | the arms' replies | first token where ro's first load differs from base's | one-token reply (w1h1) | R176's reply at this width (v4.0) |")
    P("|---:|---:|---|---|---|---:|---|---|")
    for d in depths:
        one = md5.get(("w1h1", d, p), "")
        P(f"| 1 | {d:,} | {one[:8]} | | | | | {old.get(('w1h1', d, p), '')[:8]} |")
    for w in widths:
        for d in depths:
            bl = sorted(l for (l, dd, pp) in md5 if dd == d and pp == p and re.match(rf"w{w}b\d$", l))
            rl = sorted(l for (l, dd, pp) in md5 if dd == d and pp == p and re.match(rf"w{w}r\d$", l))
            bm, rm = [md5[(l, d, p)] for l in bl], [md5[(l, d, p)] for l in rl]
            same = "the same" if len(set(bm + rm)) == 1 else ("each arm its own" if len(set(bm)) == 1 and
                                                                len(set(rm)) == 1 else "differ within an arm")
            fd = first_diff(ids.get((bl[0], d, p)) if bl else None, ids.get((rl[0], d, p)) if rl else None)
            one = md5.get(("w1h1", d, p), "")
            o = old.get((f"w{w}g1", d, p), "")
            P(f"| {w} | {d:,} | {', '.join(x[:8] for x in bm)} | {', '.join(x[:8] for x in rm)} | {same} | {fd} | "
              f"{'base and ro' if one and set(bm + rm) == {one} else ', '.join(k for k, xs in (('base', bm), ('ro', rm)) if one and set(xs) == {one}) or 'neither'} | "
              f"{'base and ro' if o and set(bm + rm) == {o} else ', '.join(k for k, xs in (('base', bm), ('ro', rm)) if o and set(xs) == {o}) or 'neither'} |")
    P()
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("table\twidth\tdepth\trequest\tarm\tload\tname\tvalue\n")
            for t, w, d, q, arm, l, n, v in out:
                f.write(f"{t}\t{w}\t{d}\t{q}\t{arm}\t{l}\t{n}\t{v:.4f}\n")


if __name__ == "__main__":
    main()
