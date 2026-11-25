#!/usr/bin/env python3
"""C1003 (Q1736): the tables of c1003.sh's deep-lane run. Standard library only; one load's profile in memory at a time.

Usage: python3 -I c1003_tables.py <OUT> <RES>

<OUT> is the deep lane's results folder (requests.csv, server-<load>.log, hp-<arm>.csv), <RES> holds tables.sh's
outputs (split/<load>.tsv: S37's step_split.py --tsv per request; pieces/<load>/<request>.csv: split_trace.py's
pieces) and receives:
  step.tsv     per load, depth and request (p1, p2: the whole prompt at temperature 0; s: with the preset's sampler):
               tokens, drafted, accepted, steps (tokens less accepted), the server's ms a step (gen_ms over the
               steps), tokens a step, tok/s
  replies.tsv  per depth and request: the md5 of the token ids in every load, and whether every load gave the same
  split.tsv    per profiled load and request: the medians over the typical steps of step_split.py's step, verify span,
               verify GPU time (nine cards), verify rest and each card's verify GPU time
  queued.tsv   per profiled load, card and request piece: entries, entries with an earlier evaluation of the card
               still running, the most at once, entries by that number (0, 1, 2, 3 or more)
  prompt.tsv   per load and prompt read (the prime at a depth, each request's own new tokens): tokens read, the
               server's prompt ms and tok/s
  prompt_cards.tsv  per profiled load and card, the profile's lines of the prime at the deepest depth (every line
               after the last generation request of the first depth and before the first of the deepest, sorted by
               host_enter_ms): evaluations, their GPU ms, entries with an earlier evaluation still running, the most
               at once, the host's ms inside those entries (host_exit_ms minus host_enter_ms), and the span from
               the first entry to the last exit
  lines.tsv    per profiled load and card: the profile's lines, the first and last evaluation id, ids missing between
               them and ids written twice
  profile.tsv  per profiled load and card: the count line the profile writes when the server exits (evaluations,
               written, entries with earlier ones still running, the most at once, entries the host waited at), and
               the way the load line names; empty when the server's log level hides those lines
"""
import collections
import csv
import glob
import os
import re
import statistics
import sys

OUT, RES = sys.argv[1], sys.argv[2]
FLUSH_RE = re.compile(r"ggml_cuda_profile_host_flush: CUDA(\d+): (\d+) evaluations, (\d+) written; (\d+) entered with "
                      r"earlier ones still running \(at most (\d+)\), the host waited at (\d+)")
WAY_RE = re.compile(r"(without waiting \(GGML_CUDA_PROFILE_HOST_NOWAIT=1\)|waiting for each card's earlier evaluations)")
NAME = {"1": "p1", "2": "p2", "s": "s"}


def write(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# step.tsv and replies.tsv
reqs = [r for r in csv.DictReader(open(os.path.join(OUT, "requests.csv")))
        if r["kind"] in ("decode", "sample") and r["load"] != "warm"]
loads = []
for r in reqs:
    if r["load"] not in loads:
        loads.append(r["load"])
rows, ids = [], collections.defaultdict(dict)
for r in reqs:
    n, acc, d, ms = int(r["gen_n"]), int(r["draft_accepted"] or 0), int(r["draft_n"] or 0), float(r["gen_ms"])
    steps = n - acc if d > 0 else n
    req = NAME[r["pass"]]
    rows.append([r["load"], r["depth"], req, n, d, acc, steps, f"{ms / steps:.2f}", f"{n / steps:.3f}",
                 f"{1000.0 * n / ms:.2f}"])
    ids[(int(r["depth"]), req)][r["load"]] = r["ids_md5"]
write("step.tsv", ["load", "depth", "request", "tokens", "drafted", "accepted", "steps", "ms_a_step",
                   "tokens_a_step", "tok_s"], rows)
rows = []
for (depth, req), m in sorted(ids.items()):
    same = "yes" if len(set(m.values())) == 1 and len(m) == len(loads) else "no"
    rows.append([depth, req] + [m.get(l, "") for l in loads] + [same])
write("replies.tsv", ["depth", "request"] + loads + ["same_in_every_load"], rows)

# split.tsv
rows = []
for f in sorted(glob.glob(os.path.join(RES, "split", "*.tsv"))):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f), delimiter="\t"):
        if r["typical"] == "1":
            by[r["request"]].append(r)
    for req, rs in by.items():
        cols = ["step_ms", "verify_span_ms", "verify_gpu_ms", "verify_rest_ms"] + [f"card{c}_gpu_ms" for c in range(9)]
        rows.append([rs[0]["load"], req, len(rs)] + [f"{statistics.median(float(x[c]) for x in rs):.2f}" for c in cols])
write("split.tsv", ["load", "request", "typical_steps", "step", "verify_span", "verify_gpu", "verify_rest"]
      + [f"card{c}" for c in range(9)], rows)

# queued.tsv
rows = []
for d in sorted(glob.glob(os.path.join(RES, "pieces", "*"))):
    if not os.path.isdir(d):
        continue
    load = os.path.basename(d)
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        hist = collections.defaultdict(collections.Counter)
        for r in csv.DictReader(open(p)):
            try:
                hist[int(r["device"])][int(r["queued"])] += 1
            except (KeyError, ValueError, TypeError):
                continue
        for dev in sorted(hist):
            h = hist[dev]
            rows.append([load, dev, os.path.basename(p)[:-4], sum(h.values()), sum(v for k, v in h.items() if k > 0),
                         max(h), h[0], h[1], h[2], sum(v for k, v in h.items() if k >= 3)])
write("queued.tsv", ["load", "device", "request", "entries", "entries_with_earlier_running", "most_at_once", "q0",
                     "q1", "q2", "q3_or_more"], rows)

# prompt.tsv
rows = []
for r in csv.DictReader(open(os.path.join(OUT, "requests.csv"))):
    if r["load"] == "warm" or r["kind"] not in ("prime", "decode", "sample"):
        continue
    n, ms = int(r["prompt_n"]), float(r["prompt_ms"])
    rows.append([r["load"], r["depth"], r["kind"], r["pass"], n, f"{ms:.3f}", f"{1000.0 * n / ms:.2f}"])
write("prompt.tsv", ["load", "depth", "kind", "pass", "tokens_read", "prompt_ms", "tok_s"], rows)

# prompt_cards.tsv
def times(path):
    return [float(r["host_enter_ms"]) for r in csv.DictReader(open(path))]


rows = []
for f in sorted(glob.glob(os.path.join(OUT, "hp-*.csv"))):
    load = os.path.basename(f)[3:-4] + "1"
    pd = os.path.join(RES, "pieces", load)
    firsts = sorted(x for x in os.listdir(pd) if x.endswith(".csv")) if os.path.isdir(pd) else []
    deps = sorted({int(x[1:].split("p")[0].split("s")[0]) for x in firsts})
    if len(deps) < 2:
        continue
    lo, hi = deps[0], deps[-1]
    t_lo = max(max(times(os.path.join(pd, f"d{lo}{q}.csv"))) for q in ("p1", "p2", "s"))
    t_hi = min(times(os.path.join(pd, f"d{hi}p1.csv")))
    cards = collections.defaultdict(lambda: [0, 0.0, 0, 0, 0.0, None, None])
    for r in csv.DictReader(open(f)):
        try:
            t = float(r["host_enter_ms"])
        except (KeyError, ValueError):
            continue
        if not t_lo < t < t_hi:
            continue
        c = cards[int(r["device"])]
        q, te = int(r["queued"]), float(r["host_exit_ms"])
        c[0] += 1
        c[1] += float(r["gpu_us"]) / 1000.0
        if q > 0:
            c[2] += 1
            c[4] += te - t
        c[3] = max(c[3], q)
        c[5] = t if c[5] is None else min(c[5], t)
        c[6] = te if c[6] is None else max(c[6], te)
    for dev in sorted(cards):
        c = cards[dev]
        rows.append([load, hi, dev, c[0], f"{c[1]:.1f}", c[2], c[3], f"{c[4]:.1f}", f"{c[6] - c[5]:.1f}"])
write("prompt_cards.tsv", ["load", "depth", "device", "evaluations", "gpu_ms", "entries_with_earlier_running",
                           "most_at_once", "host_ms_in_those_entries", "span_ms"], rows)

# lines.tsv
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "hp-*.csv"))):
    load = os.path.basename(f)[3:-4] + "1"
    ev = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(f)):
        try:
            ev[int(r["device"])][int(r["eval"])] += 1
        except (KeyError, ValueError, TypeError):
            continue
    for dev in sorted(ev):
        c = ev[dev]
        lo, hi = min(c), max(c)
        rows.append([load, dev, sum(c.values()), lo, hi, sum(1 for i in range(lo, hi + 1) if i not in c),
                     sum(1 for v in c.values() if v > 1)])
write("lines.tsv", ["load", "device", "lines", "first_eval", "last_eval", "missing_ids", "repeated_ids"], rows)

# profile.tsv
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "server-*.log"))):
    load = os.path.basename(f)[7:-4]
    way, flush = "", []
    for line in open(f, errors="replace"):
        m = WAY_RE.search(line)
        if m and not way:
            way = "no-wait" if "without" in m.group(1) else "waiting"
        m = FLUSH_RE.search(line)
        if m:
            flush.append(m.groups())
    for g in flush:
        rows.append([load, way] + list(g))
write("profile.tsv", ["load", "way", "device", "evaluations", "written", "entries_with_earlier_running", "most_at_once",
                      "waited"], rows)
print("tables in", RES)
