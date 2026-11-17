#!/usr/bin/env python3
"""flint_map.py: R176's map of Flint Pascal Q8 GGUF's decode step on v4.0 (Q1510), from one deep-lane run of r176.sh.

  python3 -I flint_map.py --out-dir OUT --md map.md --tsv map.tsv [--event-us 7.6] [--gap-ms 3000] [--band 0.2]

OUT is the deep lane's results folder: requests.csv and requests-<load>.csv (every request, in order), hp-<load>.csv
(GGML_CUDA_PROFILE_HOST: device, eval, host_enter_ms, host_exit_ms, gpu_us), gp-<load>.csv (GGML_CUDA_PROFILE in group
mode: device, eval, host_ms, group, label, us) and st-<load>.jsonl (LLAMA_STEP_TRACE). Each file is read streaming and
cut into pieces at idle gaps of more than --gap-ms (the lane's PAUSE): one piece per request. A load's last N pieces
are its N requests, in the order of requests-<load>.csv (warm-up, then per depth the prime and the two decode passes);
a file with fewer pieces than requests is reported and not used.

Per request piece (the passes of several tokens are width_curve.py's, step by step):
  - host profile: each card's evaluations; its typical evaluations (GPU time within --band of the card's median: the
    one-token evaluations of a decode, the full-width verify passes of a synthetic-acceptance arm) give the card's
    median GPU time and host time a pass; the pass's GPU time is the sum over the nine cards;
  - group profile: each card's evaluations, --event-us subtracted per launch group (the profiler's own cost per
    timed span on a P106, C140); the typical evaluations (total within --band of the card's median) give the mean of
    every part and launch-group label a pass, summed over the cards;
  - step trace: the server's steps, their count and the median interval between step starts.
Parts (first match on the label wins; the rest are other weights, where the label names a weight with its shape, or
other operations without weights):
  routed experts (MUL_MAT_ID, *_exps), shared expert (*_shexp), router (ffn_gate_inp), attention products (attn_*:
  q, k, v and the output projection), dense feed-forward (ffn_gate, ffn_up, ffn_down of the dense layers), attention
  (FLASH_ATTN_EXT), output layer (output).
Writes the tables as Markdown (--md) and every value in long form (--tsv: table, load, depth, pass, name, value).
Standard library only; one piece in memory at a time.
"""
import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys

PARTS = [
    ("routed experts", r"^MUL_MAT_ID|_exps:"),
    ("shared expert", r"_shexp:"),
    ("router", r"ffn_gate_inp:"),
    ("attention products (q, k, v, out)", r" attn_[a-z_]+:"),
    ("dense feed-forward", r" ffn_(gate|up|down):"),
    ("attention (FLASH_ATTN_EXT)", r"FLASH_ATTN_EXT"),
    ("output layer", r" output:"),
]
NODE = re.compile(r" node_\d+")  # a graph node's number: the same operation in every layer gets one label
PART_NAMES = [p for p, _ in PARTS] + ["other weights", "other ops without weights"]
_part_cache = {}


def part_of(label):
    p = _part_cache.get(label)
    if p is None:
        p = next((n for n, rx in PARTS if re.search(rx, label)), None)
        if p is None:
            _, _, w = label.partition(" ")
            p = "other weights" if re.search(r":[a-z0-9_]+:\d+x\d+", w) else "other ops without weights"
        _part_cache[label] = p
    return p


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pieces(path, tcol_names, gap_ms):
    """Yields (header, rows) per piece of a profile CSV, cut at gaps of more than gap_ms in the time column."""
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        tcol = next(header.index(n) for n in tcol_names if n in header)
        rows, tmax = [], None
        for r in rd:
            if len(r) < len(header):
                continue
            try:
                t = float(r[tcol])
            except ValueError:
                continue
            if tmax is not None and t - tmax > gap_ms and rows:
                yield header, rows
                rows = []
            rows.append(r)
            tmax = t if tmax is None else max(tmax, t)
        if rows:
            yield header, rows


def typical(vals, band):
    m = med(vals)
    return [v for v in vals if abs(v - m) <= band * m] if m == m and m > 0 else []


def host_piece(header, rows, band):
    ix = {n: header.index(n) for n in ("device", "eval", "host_enter_ms", "host_exit_ms", "gpu_us")}
    by = collections.defaultdict(list)
    for r in rows:
        try:
            g = float(r[ix["gpu_us"]]) / 1000.0
            h = float(r[ix["host_exit_ms"]]) - float(r[ix["host_enter_ms"]])
        except ValueError:
            continue
        by[int(r[ix["device"]])].append((g, h))
    out = {"cards": {}}
    for d, evs in sorted(by.items()):
        m = med([g for g, _ in evs])
        typ = [(g, h) for g, h in evs if m > 0 and abs(g - m) <= band * m]
        out["cards"][d] = {"evals": len(evs), "typical": len(typ), "gpu": med([g for g, _ in typ]),
                           "host": med([h for _, h in typ]), "gpu_all": sum(g for g, _ in evs)}
    out["pass_gpu"] = sum(c["gpu"] for c in out["cards"].values())
    out["pass_host"] = sum(c["host"] for c in out["cards"].values())
    return out


def group_piece(header, rows, band, event_us):
    ix = {n: header.index(n) for n in ("device", "eval", "label", "us")}
    evs = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        try:
            us = float(r[ix["us"]]) - event_us
        except ValueError:
            continue
        evs[(int(r[ix["device"]]), int(r[ix["eval"]]))][NODE.sub("", r[ix["label"]])] += us
    by_card = collections.defaultdict(list)
    for (d, _), lab in evs.items():
        by_card[d].append(lab)
    out = {"cards": {}, "parts": collections.defaultdict(float), "labels": collections.defaultdict(float),
           "card_parts": {}}
    for d, labs in sorted(by_card.items()):
        tots = [sum(l.values()) for l in labs]
        m = med(tots)
        typ = [l for l, t in zip(labs, tots) if m > 0 and abs(t - m) <= band * m]
        n = len(typ) or 1
        cp = collections.defaultdict(float)
        for l in typ:
            for lab, us in l.items():
                out["labels"][lab] += us / 1000.0 / n
                cp[part_of(lab)] += us / 1000.0 / n
        for p, v in cp.items():
            out["parts"][p] += v
        out["card_parts"][d] = dict(cp)
        out["cards"][d] = {"evals": len(labs), "typical": len(typ), "total": sum(cp.values())}
    out["pass_gpu"] = sum(c["total"] for c in out["cards"].values())
    return out


def step_pieces(path, gap_ms):
    with open(path) as f:
        f.readline()
        steps = []
        for line in f:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if "t" not in s or not s["t"]:
                continue
            steps.append(s)
    out, cur, last = [], [], None
    for s in steps:
        t0 = s["t"][0][1]
        if last is not None and t0 - last > gap_ms and cur:
            out.append(cur)
            cur = []
        cur.append(s)
        last = s["t"][-1][1]
    if cur:
        out.append(cur)
    return out


def step_summary(steps):
    starts = [s["t"][0][1] for s in steps]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    marks = collections.Counter(m for s in steps for m, _ in s["t"])
    ns = collections.Counter()
    for s in steps:
        for k, v in (s.get("n") or {}).items():
            if isinstance(v, (int, float)):
                ns[k] += v
    q = sorted(gaps)
    pct = lambda f: q[min(len(q) - 1, int(f * len(q)))] if q else float("nan")
    return {"steps": len(steps), "interval": med(gaps), "p10": pct(0.1), "p90": pct(0.9), "marks": marks, "n": ns,
            "span": (starts[-1] - starts[0]) if len(starts) > 1 else 0.0}


PIPE = re.compile(r"pipelined = +(\d+) guesses, (\d+) right, (\d+) wrong")


def pipe_lines(path):
    """The server log's pipelined-guess counts, one per request that generated with guesses, in order."""
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = PIPE.search(line)
            if m:
                out.append(tuple(int(x) for x in m.groups()))
    return out


def read_requests(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def name_of(r):
    return f"{r['kind']}:{r['depth']}:{r['pass']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--event-us", type=float, default=7.6)
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--band", type=float, default=0.2)
    a = ap.parse_args()
    O = a.out_dir
    allreq = read_requests(os.path.join(O, "requests.csv"))
    loads = []
    for r in allreq:
        if r["load"] not in loads:
            loads.append(r["load"])
    tsv = []
    md = []

    def put(table, load, depth, pas, name, value):
        tsv.append((table, load, depth, pas, name, value))

    req = {}  # (load, name) -> request row
    host, group, step, pipe = {}, {}, {}, {}
    notes = []
    for load in loads:
        p = os.path.join(O, f"requests-{load}.csv")
        rs = read_requests(p) if os.path.exists(p) else [r for r in allreq if r["load"] == load]
        names = [name_of(r) for r in rs]
        for r in rs:
            req[(load, name_of(r))] = r
        for kind, path, store in (("hp", f"hp-{load}.csv", host), ("gp", f"gp-{load}.csv", group)):
            fp = os.path.join(O, path)
            if not os.path.exists(fp):
                continue
            ps = []
            for header, rows in pieces(fp, ("host_ms", "host_enter_ms"), a.gap_ms):
                if kind == "hp":
                    ps.append(host_piece(header, rows, a.band))
                else:
                    ps.append(group_piece(header, rows, a.band, a.event_us))
                del rows
            notes.append(f"{path}: {len(ps)} pieces for {len(rs)} requests; card 0's evaluations per piece: "
                         + " ".join(str(pc["cards"].get(0, {}).get("evals", 0)) for pc in ps))
            if len(ps) < len(rs):
                notes.append(f"{path}: fewer pieces than requests, not used")
                continue
            for n, pc in zip(names, ps[-len(rs):]):
                store[(load, n)] = pc
        fp = os.path.join(O, f"server-{load}.log")
        if os.path.exists(fp):
            pl = pipe_lines(fp)
            gen = [n for n, r in zip(names, rs) if float(r["gen_n"] or 0) > 0]
            notes.append(f"server-{load}.log: {len(pl)} pipelined-guess lines for {len(gen)} requests that generate")
            if pl and len(pl) <= len(gen):
                for n, v in zip(gen[-len(pl):], pl):
                    pipe[(load, n)] = v
        fp = os.path.join(O, f"st-{load}.jsonl")
        if os.path.exists(fp):
            sps = step_pieces(fp, a.gap_ms)
            notes.append(f"st-{load}.jsonl: {len(sps)} pieces for {len(rs)} requests; steps per piece: "
                         + " ".join(str(len(sp)) for sp in sps))
            if len(sps) >= len(rs):
                for n, sp in zip(names, sps[-len(rs):]):
                    step[(load, n)] = step_summary(sp)

    depths = sorted({int(r["depth"]) for r in allreq if r["kind"] == "decode"})
    f2 = lambda v: "" if v is None or v != v else f"{v:.2f}"

    # 1. every request
    md.append("## 1. Every request (the server's own times)\n")
    md.append("| Load | Request | Depth | Prompt tokens read | Prompt ms | Tokens out | Generation ms | ms a token | Drafted | Accepted | Tokens a step | ms a step | Reply ids md5 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in allreq:
        gen_n = float(r["gen_n"] or 0)
        gen_ms = float(r["gen_ms"] or 0)
        tps = float(r["tokens_per_step"]) if r.get("tokens_per_step") else (1.0 if gen_n else 0.0)
        mpt = gen_ms / gen_n if gen_n else None
        mps = gen_ms / (gen_n / tps) if gen_n and tps else None
        md.append(f"| {r['load']} | {r['kind']} {r['pass']} | {r['depth']} | {r['prompt_n']} | {r['prompt_ms']} | {r['gen_n']} | "
                  f"{r['gen_ms']} | {f2(mpt)} | {r['draft_n']} | {r['draft_accepted']} | {r['tokens_per_step']} | {f2(mps)} | "
                  f"`{r['ids_md5'][:8]}` |")
        if r["kind"] == "decode":
            for k, v in (("ms_a_token", mpt), ("tokens_a_step", tps), ("ms_a_step", mps), ("gen_ms", gen_ms),
                         ("gen_n", gen_n), ("draft_n", float(r["draft_n"] or 0)),
                         ("draft_accepted", float(r["draft_accepted"] or 0))):
                put("requests", r["load"], r["depth"], r["pass"], k, v)

    def get(store, load, depth, pas):
        return store.get((load, f"decode:{depth}:{pas}"))

    def reqv(load, depth, pas):
        return req.get((load, f"decode:{depth}:{pas}"))

    hloads = [l for l in loads if l.startswith("h")]
    gloads = [l for l in loads if l.startswith("g")]

    # 2. the one-token step as served
    md.append("\n## 2. The one-token step of the preset (pipelined guesses), host profile\n")
    md.append("Per request: the server's ms a token under the host profile, which waits at each card's next evaluation "
              "for its previous one and so keeps the host from queuing guesses ahead (not the preset's speed); the "
              "server log's guesses; each card's evaluations a generated token (a dropped guess is an evaluation that "
              "gave no token); the pass's GPU time (each card's median over its typical evaluations, summed over the "
              "nine cards) and the cards' host time a pass (the same, between entering and leaving each card's graph "
              "evaluation). The step trace has no line per pipelined token.\n")
    md.append("| Load | Depth | Pass | ms a token (host profile) | Guesses, right, wrong (server log) | Card 0 evaluations a token | Card 8 evaluations a token | Pass GPU ms | Cards' host ms a pass | Steps (trace) | Step interval ms: p10, median, p90 |")
    md.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|")
    for l in hloads:
        for d in depths:
            for pas in ("1", "2"):
                r, h, s = reqv(l, d, pas), get(host, l, d, pas), get(step, l, d, pas)
                if not r:
                    continue
                gen_n = float(r["gen_n"])
                mpt = float(r["gen_ms"]) / gen_n
                e0 = h["cards"].get(0, {}).get("evals", 0) / gen_n if h else None
                e8 = h["cards"].get(8, {}).get("evals", 0) / gen_n if h else None
                pg = h["pass_gpu"] if h else None
                ph = h["pass_host"] if h else None
                pv = pipe.get((l, f"decode:{d}:{pas}"))
                md.append(f"| {l} | {d} | {pas} | {f2(mpt)} | {', '.join(map(str, pv)) if pv else ''} | {f2(e0)} | {f2(e8)} | "
                          f"{f2(pg)} | {f2(ph)} | {s['steps'] if s else ''} | "
                          f"{(f2(s['p10']) + ', ' + f2(s['interval']) + ', ' + f2(s['p90'])) if s else ''} |")
                if pv:
                    for k, v in zip(("guesses", "right", "wrong"), pv):
                        put("one_token_host", l, d, pas, k, v)
                if s:
                    put("one_token_host", l, d, pas, "step_interval_p10", s["p10"])
                    put("one_token_host", l, d, pas, "step_interval_p90", s["p90"])
                for k, v in (("evals_a_token_card0", e0), ("evals_a_token_card8", e8), ("pass_gpu", pg), ("pass_host", ph),
                             ("steps", s["steps"] if s else None), ("step_interval", s["interval"] if s else None)):
                    if v is not None:
                        put("one_token_host", l, d, pas, k, v)
                if h:
                    for c, cv in h["cards"].items():
                        put("one_token_host_card", l, d, pas, f"card{c}_gpu", cv["gpu"])
                        put("one_token_host_card", l, d, pas, f"card{c}_host", cv["host"])
                        put("one_token_host_card", l, d, pas, f"card{c}_evals", cv["evals"])

    # 3. parts of a one-token pass (group profile)
    def parts_table(title, loads_, pas):
        md.append(title)
        cols = [(l, d) for d in depths for l in loads_]
        md.append("| Part | " + " | ".join(f"{l}, {d}" for l, d in cols) + " |")
        md.append("|---|" + "---:|" * len(cols))
        for p in PART_NAMES + ["the pass"]:
            vals = []
            for l, d in cols:
                g = get(group, l, d, pas)
                v = (g["pass_gpu"] if p == "the pass" else g["parts"].get(p, 0.0)) if g else None
                vals.append(v)
                if v is not None:
                    put("parts", l, d, pas, p, v)
            md.append(f"| {'**' + p + '**' if p == 'the pass' else p} | " + " | ".join(f2(v) for v in vals) + " |")

    if gloads:
        parts_table("\n## 3. A one-token pass by part (group profile, the repeated request), GPU ms summed over the nine cards\n",
                    gloads, "2")
        parts_table("\n### 3a. The same, the first request at the depth\n", gloads, "1")
        # growth with depth by label
        if len(depths) > 1:
            d0, d1 = depths[0], depths[-1]
            md.append(f"\n### 3b. What grows from depth {d0} to {d1}, by launch-group label (the repeated request, ms a pass)\n")
            labs = collections.Counter()
            for l in gloads:
                g0, g1 = get(group, l, d0, "2"), get(group, l, d1, "2")
                if g0 and g1:
                    for lab in set(g0["labels"]) | set(g1["labels"]):
                        labs[lab] += abs(g1["labels"].get(lab, 0) - g0["labels"].get(lab, 0))
            top = [lab for lab, _ in labs.most_common(12)]
            md.append("| Label | Part | " + " | ".join(f"{l}: {d0}, {d1}, growth" for l in gloads) + " |")
            md.append("|---|---|" + "---|" * len(gloads))
            for lab in top:
                cells = []
                for l in gloads:
                    g0, g1 = get(group, l, d0, "2"), get(group, l, d1, "2")
                    if g0 and g1:
                        v0, v1 = g0["labels"].get(lab, 0.0), g1["labels"].get(lab, 0.0)
                        cells.append(f"{v0:.2f}, {v1:.2f}, {v1 - v0:+.2f}")
                        put("growth_label", l, f"{d0}-{d1}", "2", lab, v1 - v0)
                    else:
                        cells.append("")
                md.append(f"| `{lab}` | {part_of(lab)} | " + " | ".join(cells) + " |")
        # per card
        md.append("\n### 3c. Each card's GPU ms a one-token pass (group profile, the repeated request)\n")
        cols = [(l, d) for d in depths for l in gloads]
        md.append("| Card | " + " | ".join(f"{l}, {d}" for l, d in cols) + " |")
        md.append("|---:|" + "---:|" * len(cols))
        for c in range(9):
            vals = []
            for l, d in cols:
                g = get(group, l, d, "2")
                v = g["cards"].get(c, {}).get("total") if g else None
                vals.append(v)
                if v is not None:
                    put("card_gpu_group", l, d, "2", f"card{c}", v)
            md.append(f"| {c} | " + " | ".join(f2(v) for v in vals) + " |")
        # biggest labels at depth 0
        md.append(f"\n### 3d. The biggest launch-group labels of a one-token pass (the repeated request, depth {depths[0]})\n")
        g = get(group, gloads[0], depths[0], "2")
        if g:
            top = sorted(g["labels"].items(), key=lambda kv: -kv[1])[:25]
            md.append("| Label | Part | " + " | ".join(gloads) + " |")
            md.append("|---|---|" + "---:|" * len(gloads))
            for lab, _ in top:
                vals = []
                for l in gloads:
                    gg = get(group, l, depths[0], "2")
                    vals.append(gg["labels"].get(lab) if gg else None)
                md.append(f"| `{lab}` | {part_of(lab)} | " + " | ".join(f2(v) for v in vals) + " |")
        for l in gloads:
            for d in depths:
                for pas in ("1", "2"):
                    g = get(group, l, d, pas)
                    if g:
                        for lab, v in g["labels"].items():
                            put("labels", l, d, pas, lab, v)

    # 5. step-trace marks
    md.append("\n## 5. Step-trace marks per request (counts), for reading the step\n")
    for (l, n), s in sorted(step.items()):
        if not n.startswith("decode"):
            continue
        top = ", ".join(f"{m} {c}" for m, c in s["marks"].most_common(40))
        nn = ", ".join(f"{k} {v}" for k, v in sorted(s["n"].items()))
        md.append(f"- **{l} {n}**: {s['steps']} steps over {s['span']:.1f} ms; n: {nn}; marks: {top}")

    md.append("\n## Notes\n")
    md.extend(f"- {x}" for x in notes)
    with open(a.md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(a.tsv, "w") as f:
        f.write("table\tload\tdepth\tpass\tname\tvalue\n")
        for t in tsv:
            v = t[5]
            f.write("\t".join(map(str, t[:5])) + "\t" + (f"{v:.4f}" if isinstance(v, float) else str(v)) + "\n")
    print("\n".join(notes), file=sys.stderr)


if __name__ == "__main__":
    main()
