#!/usr/bin/env python3
"""first_request.py: why the first request at a depth generates slower than its repeat (R121, Q1133), from one
deep-lane load's LLAMA_STEP_TRACE file, its requests CSV and, optionally, its GGML_CUDA_PROFILE_HOST file.

  python3 -I first_request.py --trace st-LOAD.jsonl --requests requests-LOAD.csv [--profile hp-LOAD.csv
      --profile-t0 MS] [--depth 4096] [--gap-ms 2000] [--tsv steps.tsv] [--pairs-tsv pairs.tsv] > first.md

The deep lane (bench/deep_lane) sends per depth a prime (prompt only), decode pass 1 (the first request at the depth:
it reads the prompt's last 64 tokens on a live cache), decode pass 2 (the same prompt again: it restores a checkpoint
four tokens before the end and reads 4) and, with SAMPLER_PASS=1, a sampled pass. With PAUSE set the client waits
before every request, so a gap of more than --gap-ms between two trace lines cuts the trace into one piece per request.
The pieces are named from the requests CSV in order (a request that decoded nothing writes no line; the CSV's rows
with prompt_n = 0 are skipped).

A trace line is one llama-server update_slots call that decoded (C405's format: {"step", "n": {...}, "t": [[mark,
ms], ...]}). Within a request, the generation lines are those that verify a draft (n.drf > 0 or an s.acc mark) or,
without a draft, decode one token; the lines before the first of them are the prompt's. A line's span runs from its
first mark to the next line's first mark (the last line: to its own last mark), so the spans of a request's generation
lines add up to its generation time as this trace sees it: from the first generation line's start to the last line's
end. The server's own gen_ms starts at the token the prompt's last call sampled, so the prompt's last line's tail
belongs to it too; both are printed.

Per request: lines, generation lines, the generation span, the server's gen_ms, the graph builds and reuses of both
contexts (v.ub.build, v.ub.reuse, d.ub.build, d.ub.reuse), and with the profile the GPU time of the main passes (all
cards) and of the draft's evaluations inside the generation lines. Then pass 1 against pass 2 at each depth, step by
step (step i of one against step i of the other; the steps line up when both drafted and accepted alike):
  - the span's difference: the first three steps, the rest, the whole;
  - by stage (the stages below, summed over the generation lines): which stage holds the difference;
  - by consecutive mark pair (a#k -> b, k the occurrence of a in its line): the ten pairs with the largest
    difference.
Stages, by the marks that open them (each runs to the next stage's opening mark in the line):
  draft s.drf, checkpoints s.ckpt, batch s.bat, verify s.dec (its waits for the cards, v.sync -> v.sync.end, also
  counted apart), catch-up s.cu.in, acceptance s.pd, the line's own end s.upd.end, and "between" from the line's last
  mark to the next line's first. Marks before the first stage's mark count as "loop".
Standard library only.
"""
import argparse
import bisect
import csv
import json
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict

STAGES = [("s.drf", "draft"), ("s.ckpt", "checkpoints"), ("s.bat", "batch"), ("s.dec", "verify"),
          ("s.cu.in", "catch-up"), ("s.pd", "acceptance"), ("s.upd.end", "end")]
STAGE_OF = dict(STAGES)
ORDER = ["loop"] + [s for _, s in STAGES] + ["between"]


def read_trace(path):
    lines = []
    with open(path) as f:
        head = json.loads(f.readline())
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except ValueError:
                continue  # a line cut at the server's exit
            if o.get("t"):
                lines.append(o)
    return head, lines


def read_requests(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("prompt_n") not in (None, "", "0")]


def is_gen(o):
    n = o.get("n", {})
    names = {m for m, _ in o["t"]}
    return n.get("drf", 0) > 0 or "s.acc" in names


def cut(lines, gap_ms):
    pieces, cur = [], []
    for o in lines:
        if cur and o["t"][0][1] - cur[-1]["t"][-1][1] > gap_ms:
            pieces.append(cur)
            cur = []
        cur.append(o)
    if cur:
        pieces.append(cur)
    return pieces


def gen_lines(piece, speculative):
    if speculative:
        idx = [i for i, o in enumerate(piece) if is_gen(o)]
        first = idx[0] if idx else len(piece)
    else:
        # without a draft: the generation lines decode one token; the prompt's lines come first
        first = 0
        while first < len(piece) and piece[first].get("n", {}).get("tok", 0) != 1:
            first += 1
    return piece[:first], piece[first:]


def stage_times(o, nxt_t0):
    """{stage: ms} for one line, plus 'wait' (verify's waits for the cards) and the pairs {key: ms}."""
    out = defaultdict(float)
    pairs = defaultdict(float)
    marks = o["t"]
    cur = "loop"
    occ = Counter()
    in_verify = False
    sync0 = None
    for (a, ta), (b, tb) in zip(marks, marks[1:]):
        k = occ[a]
        occ[a] += 1
        if a in STAGE_OF:
            cur = STAGE_OF[a]
        out[cur] += tb - ta
        pairs[f"{a}#{k} -> {b}"] += tb - ta
        if a == "s.dec":
            in_verify = True
        if a == "s.dec.end":
            in_verify = False
        if in_verify and a == "v.sync":
            sync0 = ta
        if in_verify and b == "v.sync.end" and sync0 is not None:
            out["verify: waits for the cards"] += tb - sync0
            sync0 = None
    last = marks[-1][1]
    if nxt_t0 is not None:
        out["between"] += nxt_t0 - last
        pairs[f"{marks[-1][0]} -> next line"] += nxt_t0 - last
    return out, pairs


def builds(lines):
    c = Counter()
    for o in lines:
        for m, _ in o["t"]:
            if m in ("v.ub.build", "v.ub.reuse", "d.ub.build", "d.ub.reuse"):
                c[m] += 1
    return c


def read_profile(path, shift):
    """[(enter on the trace's clock, device, gpu ms)] sorted"""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["host_enter_ms"]) + shift, int(r["device"]), float(r["gpu_us"]) / 1000.0))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort()
    return rows


def ctx_windows(o):
    """[(start, end, 'v'|'d')] of the line's decode calls (from X.dec to X.dec.end)"""
    out, open_ = [], {}
    for m, t in o["t"]:
        if m in ("v.dec", "d.dec"):
            open_[m[0]] = t
        elif m in ("v.dec.end", "d.dec.end") and m[0] in open_:
            out.append((open_.pop(m[0]), t, m[0]))
    return out


def gpu_of(o, prof, keys, slack=0.05):
    """GPU ms of the evaluations entered inside this line's decode calls: {'v': ms, 'd': ms}"""
    res = {"v": 0.0, "d": 0.0}
    for a, b, c in ctx_windows(o):
        i = bisect.bisect_left(keys, a - slack)
        while i < len(prof) and prof[i][0] <= b + slack:
            res[c] += prof[i][2]
            i += 1
    return res


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--profile-t0", type=float, help="the profile's 0 ms on the steady clock (align_t0.py)")
    ap.add_argument("--depth", type=int, action="append", help="depths to compare (default: every depth)")
    ap.add_argument("--gap-ms", type=float, default=2000.0)
    ap.add_argument("--tsv")
    ap.add_argument("--pairs-tsv")
    a = ap.parse_args()
    head, lines = read_trace(a.trace)
    reqs = read_requests(a.requests)
    pieces = cut(lines, a.gap_ms)
    P = print
    P(f"trace {len(lines)} lines in {len(pieces)} pieces (cut at gaps over {a.gap_ms:g} ms); requests in the CSV "
      f"that read a prompt: {len(reqs)}")
    if len(pieces) != len(reqs):
        P(f"**the pieces ({len(pieces)}) do not match the requests ({len(reqs)}): the last {min(len(pieces), len(reqs))} "
          f"of each are paired**")
    n = min(len(pieces), len(reqs))
    pieces, reqs = pieces[-n:], reqs[-n:]
    speculative = any(is_gen(o) for o in lines)
    prof, keys = None, None
    if a.profile:
        if a.profile_t0 is None:
            sys.exit("--profile needs --profile-t0 (align_t0.py)")
        prof = read_profile(a.profile, a.profile_t0 - head["t0_steady_ms"])
        keys = [r[0] for r in prof]
    reqrows = []
    per = OrderedDict()
    for piece, r in zip(pieces, reqs):
        name = f"{r['kind']}{r['pass'] if r['kind'] != 'prime' else ''}@{r['depth']}"
        pl, gl = gen_lines(piece, speculative)
        spans, stages, pairs, gpus = [], [], [], []
        for i, o in enumerate(gl):
            nxt = gl[i + 1]["t"][0][1] if i + 1 < len(gl) else None
            span = (nxt if nxt is not None else o["t"][-1][1]) - o["t"][0][1]
            st, pr = stage_times(o, nxt)
            spans.append(span)
            stages.append(st)
            pairs.append(pr)
            if prof is not None:
                gpus.append(gpu_of(o, prof, keys))
        gen_span = (gl[-1]["t"][-1][1] - gl[0]["t"][0][1]) if gl else 0.0
        tail = (gl[0]["t"][0][1] - pl[-1]["t"][-1][1]) if (gl and pl) else float("nan")
        b = builds(gl)
        bp = builds(pl)
        acc = sum(o.get("n", {}).get("acc", 0) for o in gl)
        drf = sum(o.get("n", {}).get("drf", 0) for o in gl)
        per[name] = {"req": r, "pl": pl, "gl": gl, "spans": spans, "stages": stages, "pairs": pairs, "gpus": gpus}
        reqrows.append([name, str(len(pl)), str(len(gl)), str(drf), str(acc), fmt(gen_span), fmt(tail),
                        r.get("gen_ms", ""), r.get("draft_n", ""), r.get("draft_accepted", ""),
                        f"{b['v.ub.build']}/{b['v.ub.reuse']}", f"{b['d.ub.build']}/{b['d.ub.reuse']}",
                        f"{bp['v.ub.build']}/{bp['v.ub.reuse']}"] +
                       ([fmt(sum(g['v'] for g in gpus)), fmt(sum(g['d'] for g in gpus))] if prof is not None else []))
    cols = ["request", "prompt lines", "generation lines", "drafted (trace)", "accepted (trace)",
            "generation span, ms", "prompt's last line end to the first generation line, ms", "server gen_ms",
            "server drafted", "server accepted", "main graph built/reused", "draft graph built/reused",
            "prompt lines: main graph built/reused"]
    if prof is not None:
        cols += ["main passes' GPU time, ms (all cards)", "draft's GPU time, ms"]
    P()
    P("| " + " | ".join(cols) + " |")
    P("|---" + "|---:" * (len(cols) - 1) + "|")
    for row in reqrows:
        P("| " + " | ".join(row) + " |")
    P()
    if a.tsv:
        with open(a.tsv, "w") as f:
            hdr = ["request", "i", "span_ms"] + ORDER + ["verify: waits for the cards"]
            if prof is not None:
                hdr += ["main_gpu_ms", "draft_gpu_ms"]
            f.write("\t".join(hdr) + "\n")
            for name, d in per.items():
                for i, (s, st) in enumerate(zip(d["spans"], d["stages"])):
                    row = [name, str(i), f"{s:.3f}"] + [f"{st.get(k, 0.0):.3f}" for k in ORDER] + \
                          [f"{st.get('verify: waits for the cards', 0.0):.3f}"]
                    if prof is not None:
                        row += [f"{d['gpus'][i]['v']:.3f}", f"{d['gpus'][i]['d']:.3f}"]
                    f.write("\t".join(row) + "\n")
    # pass 1 against pass 2 at each depth
    depths = sorted({int(d["req"]["depth"]) for d in per.values()})
    pairs_out = []
    for dep in depths:
        if a.depth and dep not in a.depth:
            continue
        p1, p2 = per.get(f"decode1@{dep}"), per.get(f"decode2@{dep}")
        if not p1 or not p2:
            continue
        m = min(len(p1["spans"]), len(p2["spans"]))
        P(f"### Depth {dep}: pass 1 against pass 2")
        P()
        P(f"- generation lines: {len(p1['spans'])} and {len(p2['spans'])}; drafted and accepted per line equal in "
          f"{sum(1 for x, y in zip(p1['gl'], p2['gl']) if x.get('n') == y.get('n'))} of {m} paired lines")
        d = [x - y for x, y in zip(p1["spans"], p2["spans"])]
        P(f"- the span, pass 1 − pass 2, ms: all {m} paired steps {fmt(sum(d))}; steps 0-2 {fmt(sum(d[:3]))} "
          f"({', '.join(fmt(x) for x in d[:3])}); steps 3 on {fmt(sum(d[3:]))} (median {fmt(statistics.median(d[3:]) if len(d) > 3 else float('nan'))} "
          f"a step); the server's gen_ms {p1['req']['gen_ms']} − {p2['req']['gen_ms']} = "
          f"{float(p1['req']['gen_ms']) - float(p2['req']['gen_ms']):.2f}")
        P(f"- the span, median a step: pass 1 {fmt(statistics.median(p1['spans'][3:]))}, pass 2 "
          f"{fmt(statistics.median(p2['spans'][3:]))} (steps 3 on)")
        P()
        P("| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |")
        P("|---|---:|---:|---:|---:|---:|")
        for k in ORDER + ["verify: waits for the cards"]:
            s1 = [st.get(k, 0.0) for st in p1["stages"][:m]]
            s2 = [st.get(k, 0.0) for st in p2["stages"][:m]]
            dd = [x - y for x, y in zip(s1, s2)]
            if abs(sum(s1)) + abs(sum(s2)) < 0.005:
                continue
            P(f"| {k} | {fmt(sum(s1))} | {fmt(sum(s2))} | {fmt(sum(dd))} | {fmt(sum(dd[:3]))} | "
              f"{fmt(statistics.median(dd[3:]) if len(dd) > 3 else float('nan'), 3)} |")
        if prof is not None:
            for k, lab in (("v", "main passes' GPU time (all cards)"), ("d", "the draft's GPU time")):
                s1 = [g[k] for g in p1["gpus"][:m]]
                s2 = [g[k] for g in p2["gpus"][:m]]
                dd = [x - y for x, y in zip(s1, s2)]
                P(f"| {lab} | {fmt(sum(s1))} | {fmt(sum(s2))} | {fmt(sum(dd))} | {fmt(sum(dd[:3]))} | "
                  f"{fmt(statistics.median(dd[3:]) if len(dd) > 3 else float('nan'), 3)} |")
        P()
        agg = defaultdict(lambda: [0.0, 0.0])
        for i in range(m):
            for kk, v in p1["pairs"][i].items():
                agg[kk][0] += v
            for kk, v in p2["pairs"][i].items():
                agg[kk][1] += v
        top = sorted(agg.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))
        P("The ten mark pairs with the largest difference (summed over the paired steps):")
        P()
        P("| mark pair | pass 1, ms | pass 2, ms | difference |")
        P("|---|---:|---:|---:|")
        for kk, (x, y) in top[:10]:
            P(f"| `{kk}` | {fmt(x)} | {fmt(y)} | {fmt(x - y)} |")
        P()
        P("Step by step, pass 1 − pass 2, ms: " + ", ".join(fmt(x, 1) for x in d))
        P()
        for kk, (x, y) in top:
            pairs_out.append((dep, kk, x, y))
    if a.pairs_tsv:
        with open(a.pairs_tsv, "w") as f:
            f.write("depth\tpair\tpass1_ms\tpass2_ms\tdiff_ms\n")
            for dep, kk, x, y in pairs_out:
                f.write(f"{dep}\t{kk}\t{x:.3f}\t{y:.3f}\t{x - y:.3f}\n")


if __name__ == "__main__":
    main()
