#!/usr/bin/env python3
"""host_split.py: a one-token decode step (no draft) split into the host's part and the cards' part, from llama-server's
LLAMA_STEP_TRACE file, with the nine cards' GPU time from step_split.py's table of the same load (R176, Q1510).

  python3 -I host_split.py TRACE.jsonl --split STEPS.tsv --load w1h1 [--names d0p1,d0p2,d8192p1,d8192p2]
      [--gap-ms 3000] [--min-steps 45] [--skip 3] [--tsv out.tsv] > host.md

The trace has one JSON line per server update that decoded (step_gap.py's docstring has the marks). A one-token
generation line holds: the server's turn (s.upd ... s.dec.in), the main context's decode (v.dec ... v.ub.in: the
inputs set; v.ub.comp: the nine cards' graphs queued; v.dec.end), then the wait for the cards (the first v.sync to
v.sync.end: the sampled token back from the last card), then the server's bookkeeping to s.upd.end. Lines are cut into
requests at gaps of more than --gap-ms (the deep lane's PAUSE); a piece with at least --min-steps one-token lines is a
generation request, named by --names in order. Per line i of a request (its first --skip lines and its last left out):
  step       s.upd of line i+1 - s.upd of line i
  host_pre   v.ub.in - s.upd: the server's turn and the decode's set-up up to the inputs set
  launch     v.ub.comp - v.ub.in: the host queuing the nine cards' graphs (the cards start as the first is queued)
  to_wait    the first v.sync - v.ub.comp
  wait       the first v.sync.end - the first v.sync
  host_post  s.upd of line i+1 - the first v.sync.end: the token read, the server's bookkeeping, the next update's start
  cards      launch + to_wait + wait: from the inputs set to the last card's token on the host
With --split (step_split.py --tsv of the load's GGML_CUDA_PROFILE_HOST file, its requests named as here) the nine
cards' GPU time (verify_gpu_ms, median over the request's typical steps) is set beside it, and
  rest = cards - GPU time: the hops between the cards, the launches' latency and the gaps (est.: medians of two files).
Prints one table of medians per request; --tsv writes them long (load, request, part, ms).
Standard library only.
"""
import argparse
import csv
import json
import statistics


def first(marks, name):
    for n, t in marks:
        if n == name:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--split")
    ap.add_argument("--load", default="load")
    ap.add_argument("--names", default="d0p1,d0p2,d8192p1,d8192p2")
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--min-steps", type=int, default=45)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    names = a.names.split(",")
    lines = []
    with open(a.trace) as f:
        f.readline()  # the header
        for raw in f:
            try:
                o = json.loads(raw)
            except ValueError:
                continue  # a line cut at the server's exit
            m = o.get("t") or []
            if m:
                lines.append(o)
    pieces, cur, last_end = [], [], None
    for o in lines:
        t0, t1 = o["t"][0][1], o["t"][-1][1]
        if cur and t0 - last_end > a.gap_ms:
            pieces.append(cur)
            cur = []
        cur.append(o)
        last_end = t1
    if cur:
        pieces.append(cur)
    gen = [p for p in pieces if sum(1 for o in p if o["n"].get("tok") == 1 and not o["n"].get("drf")) >= a.min_steps]
    gpu = {}
    if a.split:
        with open(a.split, newline="") as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if r["typical"] == "1"]
        for n in names:
            xs = [float(r["verify_gpu_ms"]) for r in rows if r["request"] == n]
            if xs:
                gpu[n] = statistics.median(xs)
    print(f"# {a.load}: {len(lines)} trace lines in {len(pieces)} pieces; {len(gen)} generation requests for "
          f"{len(names)} names")
    print()
    out = []
    for n, p in zip(names, gen):
        one = [o for o in p if o["n"].get("tok") == 1 and not o["n"].get("drf")]
        vals = {k: [] for k in ("step", "host_pre", "launch", "to_wait", "wait", "host_post", "cards")}
        for o, nxt in list(zip(one, one[1:]))[a.skip:]:
            m = o["t"]
            s, ub_in, ub_comp, sy, sy_end = (first(m, "s.upd"), first(m, "v.ub.in"), first(m, "v.ub.comp"),
                                              first(m, "v.sync"), first(m, "v.sync.end"))
            s_next = first(nxt["t"], "s.upd")
            if None in (s, ub_in, ub_comp, sy, sy_end, s_next):
                continue
            vals["step"].append(s_next - s)
            vals["host_pre"].append(ub_in - s)
            vals["launch"].append(ub_comp - ub_in)
            vals["to_wait"].append(sy - ub_comp)
            vals["wait"].append(sy_end - sy)
            vals["host_post"].append(s_next - sy_end)
            vals["cards"].append(sy_end - ub_in)
        med = {k: statistics.median(v) for k, v in vals.items() if v}
        print(f"## {a.load}, {n}: {len(vals['step'])} steps")
        print()
        print("| part | median ms | p10 | p90 |")
        print("|---|---:|---:|---:|")
        label = {"step": "the step", "host_pre": "host, before the inputs are set (server turn, decode set-up)",
                 "cards": "from the inputs set to the token on the host", "launch": "  of it, the nine graphs queued",
                 "to_wait": "  then to the wait", "wait": "  the wait for the last card",
                 "host_post": "host, after the token (bookkeeping, the next update's start)"}
        for k in ("step", "host_pre", "cards", "launch", "to_wait", "wait", "host_post"):
            v = sorted(vals[k])
            if not v:
                continue
            p10, p90 = v[round(0.1 * (len(v) - 1))], v[round(0.9 * (len(v) - 1))]
            print(f"| {label[k]} | {med[k]:.2f} | {p10:.2f} | {p90:.2f} |")
            out.append((n, k, med[k]))
        if n in gpu and "cards" in med:
            print(f"| the nine cards' GPU time (step_split.py, median of typical steps) | {gpu[n]:.2f} | | |")
            print(f"| the rest of the cards' part: hops, launch latency, gaps (est.) | {med['cards'] - gpu[n]:.2f} | | |")
            out.append((n, "gpu", gpu[n]))
            out.append((n, "rest", med["cards"] - gpu[n]))
        print()
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("load\trequest\tpart\tms\n")
            for n, k, v in out:
                f.write(f"{a.load}\t{n}\t{k}\t{v:.4f}\n")


if __name__ == "__main__":
    main()
