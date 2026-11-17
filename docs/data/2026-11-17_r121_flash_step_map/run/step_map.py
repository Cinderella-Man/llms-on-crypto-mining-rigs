#!/usr/bin/env python3
"""step_map.py: R121's map of a Flash-Next decode step on Miner Llama v4.0 by part, depth and verify width, each load
by value, from the outputs of the tools that read one load each (R121, Q1393).

  python3 -I step_map.py --requests requests.csv \
      --split w4h1=split/w4h1.tsv ... --verify w4g1=verify/w4g1 ... [--request p2] [--top 12] \
      [--tsv map.tsv] > map.md

Inputs:
  --requests  the deep lane's requests.csv (every load's requests: arm, load, kind, pass, depth, gen_n, gen_ms,
              draft_n, draft_accepted, ...)
  --split     LOAD=steps.tsv: S37's step_split.py --tsv over the load's GGML_CUDA_PROFILE_HOST file, its requests
              named d<depth>p1, d<depth>p2 and d<depth>s (the first request at the depth, its repeat, the sampled one)
  --verify    LOAD=dir: per request a folder dir/<request> (d0p1, d0p2, d0s, d4096p1, ...: the deep lane's order) with
              R49's verify_parts.py --out over that request's piece of the load's group-mode GGML_CUDA_PROFILE trace
              (split_trace.py), its one request being verify-r1
A load's width is read from its name: w1, w4 or w5 (verify passes of 1, 4 or 5 tokens).

Tables (--request picks the request the map is drawn from; p2, the repeat, by default):
  1. the step: the server's ms a step (gen_ms over the steps, steps = gen_n - accepted, or gen_n without a draft) of
     every load, and the host profile's split (medians over the typical steps): the step, the verify pass's GPU time
     (nine cards), the verify pass's rest (hops, rows back, acceptance, the server's work before the draft), the
     draft's span, its GPU time, the server's turn before the next verify
  2. the verify pass's GPU time by part (R41's parts, all cards) per width and depth
  3. card 8 (the output head) by position and the rollback snapshots (GATED_DELTA_NET, GDN_CONV, the PLE copies)
  4. by launch-group label: what grows from depth 0 to the deepest depth (per width), and from 4 tokens to 5 (per
     depth), each load's difference by value (loads paired in order: the first load of one against the first of the
     other)
--tsv writes every value in long form: table, width, depth, request, load, name, ms.
Standard library only.
"""
import argparse
import csv
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict


def width_of(load):
    m = re.match(r"w(\d+)", load)
    return int(m.group(1)) if m else 0


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f2(x):
    return "" if x is None or x != x else f"{x:.2f}"


def vals(xs):
    return ", ".join(f2(x) for x in xs) if xs else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--verify", action="append", default=[])
    ap.add_argument("--request", default="p2")
    ap.add_argument("--min-steps", type=int, default=40)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    out_rows = []

    def keep(table, width, depth, req, load, name, ms):
        out_rows.append((table, width, depth, req, load, name, ms))

    # the deep lane's requests
    with open(a.requests, newline="") as f:
        reqs = list(csv.DictReader(f))
    depths = sorted({int(r["depth"]) for r in reqs if r["kind"] in ("decode", "sample")})
    names = []
    for d in depths:
        names += [f"d{d}p1", f"d{d}p2", f"d{d}s"]
    server = defaultdict(dict)  # load -> request name -> ms a step
    for r in reqs:
        if r["kind"] not in ("decode", "sample"):
            continue
        n = f"d{r['depth']}{'s' if r['kind'] == 'sample' else 'p' + r['pass']}"
        gen_n, acc, ms = int(r["gen_n"]), int(r["draft_accepted"] or 0), float(r["gen_ms"])
        steps = gen_n - acc if int(r["draft_n"] or 0) > 0 else gen_n
        if steps > 0:
            server[r["load"]][n] = ms / steps
            keep("server_step", width_of(r["arm"]), int(r["depth"]), n, r["load"], "ms a step", ms / steps)
            keep("server_step", width_of(r["arm"]), int(r["depth"]), n, r["load"], "steps", steps)

    # the host profile's split
    split = defaultdict(dict)  # load -> request -> {part: median}
    for spec in a.split:
        load, path = spec.split("=", 1)
        rows = [r for r in read_tsv(path) if r["typical"] == "1"]
        byreq = defaultdict(list)
        for r in rows:
            byreq[r["request"]].append(r)
        for req, rs in byreq.items():
            npass = max(int(r["n_dp"]) for r in rs)
            def med(fn):
                xs = [fn(r) for r in rs]
                return statistics.median(xs) if xs else float("nan")
            parts = OrderedDict()
            parts["step"] = med(lambda r: float(r["step_ms"]))
            parts["verify pass: GPU time, nine cards"] = med(lambda r: float(r["verify_gpu_ms"]))
            parts["verify pass: the rest"] = med(lambda r: float(r["verify_rest_ms"]))
            if npass:
                parts["the draft's span"] = med(lambda r: float(r["step_ms"]) - float(r["verify_span_ms"]))
                parts["  its GPU time"] = med(lambda r: sum(float(r[f"gpu{k}_ms"]) + float(r[f"emb_gpu{k}_ms"] or 0)
                                                           for k in range(int(r["n_dp"]))))
                parts["  the server's turn before the next verify"] = med(
                    lambda r: float(r[f"turn{int(r['n_dp']) - 1}_ms"]))
            parts["typical steps"] = len(rs)
            split[load][req] = parts
            for k, v in parts.items():
                keep("split", width_of(load), int(req[1:].split("p")[0].rstrip("s")), req, load, k.strip(), v)

    # the group profile's verify passes
    verify = defaultdict(dict)  # load -> request -> {"parts": {}, "acct": {}, "labels": {}, "per_card": {}}
    for spec in a.verify:
        load, top = spec.split("=", 1)
        for n in names:
            d = os.path.join(top, n)  # verify_parts.py's --out over this request's piece (split_trace.py)
            if not os.path.isdir(d):
                print(f"WARNING: {load}: no {d}", file=sys.stderr)
                continue
            v = verify[load].setdefault(n, defaultdict(dict))
            for r in read_tsv(os.path.join(d, "parts.tsv")):
                if r["trace"] != "verify-r1":
                    continue
                if r["cards"] == "all cards":
                    v["parts"][r["part"]] = float(r["ms_per_pass"])
                if r["cards"] == "card 8" and r["part"].startswith("by position"):
                    v["pos"][r["part"][13:]] = float(r["ms_per_pass"])
            for r in read_tsv(os.path.join(d, "accounting.tsv")):
                if r["trace"] == "verify-r1":
                    v["acct"][f"{r['cards']}: {r['where']}"] = float(r["ms_per_pass"])
            for r in read_tsv(os.path.join(d, "labels.tsv")):  # cards 0 to 7 and card 8: summed per label
                if r["trace"] == "verify-r1" and r["cards"] != "all cards":
                    v["labels"][r["label"]] = v["labels"].get(r["label"], 0.0) + float(r["ms_per_pass"])
            for r in read_tsv(os.path.join(d, "per_card.tsv")):
                if r["trace"] == "verify-r1":
                    v["per_card"][r["card"]] = float(r["ms_per_pass"])
            cut = read_tsv(os.path.join(d, "cut.tsv"))
            if cut:
                v["cut"] = cut[0]
        for n, v in verify[load].items():
            for k, x in v["parts"].items():
                keep("verify_parts", width_of(load), int(n[1:].split("p")[0].rstrip("s")), n, load, k, x)
            for k, x in v["acct"].items():
                keep("verify_where", width_of(load), int(n[1:].split("p")[0].rstrip("s")), n, load, k, x)
            for k, x in v["labels"].items():
                keep("verify_labels", width_of(load), int(n[1:].split("p")[0].rstrip("s")), n, load, k, x)

    widths = sorted({width_of(l) for l in list(split) + list(verify) + list(server)} - {0})  # 0: the warm-up load
    P = print
    R = a.request

    def loads(width, kind):
        src = {"h": split, "g": verify, "u": server}[kind]
        return sorted(l for l in src if width_of(l) == width and re.match(rf"w\d+{kind}", l))

    # 1. the step
    P(f"## 1. The step, ms (request {R}: " + {"p1": "the first request at the depth", "p2": "the repeat",
                                              "s": "the sampled request"}.get(R, R) + "), each load by value")
    P()
    cols = ["width", "depth", "server, unprofiled", "server, host profile and trace", "server, group profile",
            "split: step", "verify GPU", "verify rest", "draft span", "draft GPU", "server turn"]
    P("| " + " | ".join(cols) + " |")
    P("|---" * 2 + "|---:" * (len(cols) - 2) + "|")
    for w in widths:
        for d in depths:
            n = f"d{d}{R}"
            un = [server[l].get(n) for l in sorted(server) if width_of(l) == w and re.match(r"w\d+u", l)]
            hs = [server[l].get(n) for l in loads(w, "h") if l in server]
            gs = [server[l].get(n) for l in loads(w, "g") if l in server]
            sp = [split[l].get(n, {}) for l in loads(w, "h")]
            row = [f"{w}", f"{d:,}", vals([x for x in un if x]), vals([x for x in hs if x]), vals([x for x in gs if x])]
            for k in ("step", "verify pass: GPU time, nine cards", "verify pass: the rest", "the draft's span",
                      "  its GPU time", "  the server's turn before the next verify"):
                row.append(vals([s[k] for s in sp if k in s]))
            P("| " + " | ".join(row) + " |")
    P()
    # 2. the verify pass by part
    for w in widths:
        gl = loads(w, "g")
        if not gl:
            continue
        P(f"## 2. Width {w}: the verify pass's GPU time by part, ms a pass, all nine cards ({', '.join(gl)}; request {R})")
        P()
        parts = []
        for l in gl:
            for d in depths:
                for k in verify[l].get(f"d{d}{R}", {}).get("parts", {}):
                    if k not in parts:
                        parts.append(k)
        P("| part | " + " | ".join(f"depth {d:,}" for d in depths) + f" | {depths[-1]:,} − 0 |")
        P("|---" + "|---:" * (len(depths) + 1) + "|")
        for k in parts:
            row = [k]
            for d in depths:
                row.append(vals([verify[l].get(f"d{d}{R}", {}).get("parts", {}).get(k) for l in gl]))
            row.append(vals([verify[l].get(f"d{depths[-1]}{R}", {}).get("parts", {}).get(k, 0) -
                             verify[l].get(f"d0{R}", {}).get("parts", {}).get(k, 0) for l in gl]))
            P("| " + " | ".join(row) + " |")
        P()
        P(f"Where it goes (verify_parts.py's buckets), ms a pass:")
        P()
        keys = []
        for l in gl:
            for d in depths:
                for k in verify[l].get(f"d{d}{R}", {}).get("acct", {}):
                    if k not in keys:
                        keys.append(k)
        P("| cards: bucket | " + " | ".join(f"depth {d:,}" for d in depths) + " |")
        P("|---" + "|---:" * len(depths) + "|")
        for k in keys:
            xs = [[verify[l].get(f"d{d}{R}", {}).get("acct", {}).get(k) for l in gl] for d in depths]
            if all((x or 0) < 0.005 for row in xs for x in row):
                continue
            P(f"| {k} | " + " | ".join(vals(x) for x in xs) + " |")
        P()
    # 4. growth by label
    def growth(title, pairs):
        """pairs: [(load_a, req_a, load_b, req_b)] -> the labels with the largest mean growth b - a"""
        labs = set()
        for la, ra, lb, rb in pairs:
            labs |= set(verify[la].get(ra, {}).get("labels", {})) | set(verify[lb].get(rb, {}).get("labels", {}))
        rows = []
        for lab in labs:
            av = [verify[la].get(ra, {}).get("labels", {}).get(lab, 0.0) for la, ra, _, _ in pairs]
            bv = [verify[lb].get(rb, {}).get("labels", {}).get(lab, 0.0) for _, _, lb, rb in pairs]
            dv = [y - x for x, y in zip(av, bv)]
            rows.append((statistics.fmean(dv), lab, av, bv, dv))
        rows.sort(key=lambda r: -abs(r[0]))
        tot_a = [sum(verify[la].get(ra, {}).get("labels", {}).values()) for la, ra, _, _ in pairs]
        tot_b = [sum(verify[lb].get(rb, {}).get("labels", {}).values()) for _, _, lb, rb in pairs]
        P(title)
        P()
        P("| launch group (label) | before | after | after − before |")
        P("|---|---:|---:|---:|")
        for _, lab, av, bv, dv in rows[:a.top]:
            P(f"| `{lab}` | {vals(av)} | {vals(bv)} | {vals(dv)} |")
        P(f"| every label (the pass) | {vals(tot_a)} | {vals(tot_b)} | {vals([y - x for x, y in zip(tot_a, tot_b)])} |")
        P()
        return rows

    for w in widths:
        gl = loads(w, "g")
        if len(depths) > 1 and gl:
            rows = growth(f"## 4a. Width {w}: what grows from depth 0 to {depths[-1]:,}, ms a pass (each load: "
                          f"{', '.join(gl)}; request {R})",
                          [(l, f"d0{R}", l, f"d{depths[-1]}{R}") for l in gl])
            for _, lab, av, bv, dv in rows:
                for l, x in zip(gl, dv):
                    keep("growth_depth", w, depths[-1], R, l, lab, x)
            if len(depths) > 2:
                growth(f"### Width {w}: depth 0 to {depths[1]:,}", [(l, f"d0{R}", l, f"d{depths[1]}{R}") for l in gl])
    g4, g5 = loads(4, "g"), loads(5, "g")
    if g4 and g5:
        for d in depths:
            rows = growth(f"## 4b. Depth {d:,}: what grows from 4 tokens to 5, ms a pass (loads paired in order: "
                          f"{', '.join(f'{x} → {y}' for x, y in zip(g4, g5))}; request {R})",
                          [(x, f"d{d}{R}", y, f"d{d}{R}") for x, y in zip(g4, g5)])
            for _, lab, av, bv, dv in rows:
                for l, x in zip(g5, dv):
                    keep("growth_width", 5, d, R, l, lab, x)
    g1 = loads(1, "g")
    if g1 and g4:
        for d in depths:
            growth(f"## 4c. Depth {d:,}: from 1 token to 4, ms a pass (loads paired in order: "
                   f"{', '.join(f'{x} → {y}' for x, y in zip(g1, g4))}; request {R})",
                   [(x, f"d{d}{R}", y, f"d{d}{R}") for x, y in zip(g1, g4)])
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("table\twidth\tdepth\trequest\tload\tname\tvalue\n")
            for t, w, d, r, l, n, v in out_rows:
                f.write(f"{t}\t{w}\t{d}\t{r}\t{l}\t{n}\t{v:.4f}\n" if isinstance(v, float) else
                        f"{t}\t{w}\t{d}\t{r}\t{l}\t{n}\t{v}\n")


if __name__ == "__main__":
    main()
