#!/usr/bin/env python3
"""width_curve.py: what a pass of k tokens costs Flint, step by step, from R176's speculative arms (r176.sh's s8, s4
and s2: n-gram drafts with every draft accepted, so each server step is one verify pass of 1 + drafted tokens).

  python3 -I width_curve.py --out-dir OUT --loads s8,s4,s2 --md widths.md --tsv widths.tsv [--gap-ms 3000]

Per load, the step trace (st-<load>.jsonl: one line per server step, with the step's batch size n.tok, its drafted
and accepted counts, and the host's marks) and the host profile (hp-<load>.csv: each card's GPU time per evaluation)
are cut into one piece per request at idle gaps of more than --gap-ms, and a load's last N pieces are its N requests
(requests-<load>.csv). In a decode request each server step evaluates every card once, so the k-th step pairs with
each card's k-th evaluation (a piece whose cards' evaluation counts differ from its step count is reported and left
out). Per step: its width (n.tok), the step's time (from its first mark to the next step's first mark; the last step
to its own last mark), the nine cards' GPU time and the rest (the step's time less the GPU time: the hops between
cards, the host's part of the pass, the sampling chain, the n-gram draft, the acceptance). Prompt steps (those
without a drafted count in the trace) are left out. Per load, depth, request and width: the steps' count and medians, and per card the
median GPU time. Standard library only.
"""
import argparse
import collections
import csv
import json
import os
import statistics


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def cut(items, t_first, t_last, gap_ms):
    out, cur, last = [], [], None
    for it in items:
        if last is not None and t_first(it) - last > gap_ms and cur:
            out.append(cur)
            cur = []
        cur.append(it)
        last = t_last(it) if last is None else max(last, t_last(it))
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loads", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    a = ap.parse_args()
    O = a.out_dir
    rows_out = []  # load, depth, pass, width, name, value
    per_step = []  # load, depth, pass, step, width, step_ms, gpu_ms, card gpus...
    notes = []
    agg = collections.OrderedDict()  # (load, depth, pass, width) -> list of (step_ms, gpu, [cards])
    for load in a.loads.split(","):
        with open(os.path.join(O, f"requests-{load}.csv"), newline="") as f:
            reqs = list(csv.DictReader(f))
        with open(os.path.join(O, f"st-{load}.jsonl")) as f:
            f.readline()
            steps = [s for s in (json.loads(l) for l in f if l.strip()) if s.get("t")]
        with open(os.path.join(O, f"hp-{load}.csv"), newline="") as f:
            hp = [r for r in csv.DictReader(f)]
        sp = cut(steps, lambda s: s["t"][0][1], lambda s: s["t"][-1][1], a.gap_ms)
        hpp = cut(hp, lambda r: float(r["host_enter_ms"]), lambda r: float(r["host_exit_ms"]), a.gap_ms)
        notes.append(f"{load}: {len(sp)} step pieces, {len(hpp)} profile pieces, {len(reqs)} requests")
        if len(sp) < len(reqs) or len(hpp) < len(reqs):
            notes.append(f"{load}: fewer pieces than requests, left out")
            continue
        for r, ss, hh in zip(reqs, sp[-len(reqs):], hpp[-len(reqs):]):
            if r["kind"] != "decode":
                continue
            d, pas = r["depth"], r["pass"]
            by = collections.defaultdict(list)
            for x in hh:
                by[int(x["device"])].append(float(x["gpu_us"]) / 1000.0)
            counts = {c: len(v) for c, v in by.items()}
            n_ev = set(counts.values())
            # the profile writes an evaluation at the card's next one, so a file's very last evaluation is not in it
            if len(by) != 9 or len(n_ev) != 1 or not (len(ss) - 1 <= n_ev.pop() <= len(ss)):
                notes.append(f"{load} depth {d} pass {pas}: {len(ss)} steps, cards' evaluations {counts}: left out")
                continue
            n_pair = min(len(v) for v in by.values())
            if n_pair < len(ss):
                notes.append(f"{load} depth {d} pass {pas}: {len(ss)} steps, {n_pair} evaluations a card (the file's "
                             f"last evaluation is never written): the last step left out")
            for i, s in enumerate(ss[:n_pair]):
                if "drf" not in (s.get("n") or {}):
                    continue  # a prompt step: generation steps carry their drafted count (0 when no draft was found)
                w = int((s.get("n") or {}).get("tok", 0))
                t0 = s["t"][0][1]
                t1 = ss[i + 1]["t"][0][1] if i + 1 < len(ss) else s["t"][-1][1]
                cards = [by[c][i] for c in range(9)]
                g = sum(cards)
                agg.setdefault((load, d, pas, w), []).append((t1 - t0, g, cards))
                per_step.append((load, d, pas, i, w, t1 - t0, g) + tuple(cards))
    md = ["## What a pass of k tokens costs (the speculative arms, step by step, host profile)\n",
          "Per load, depth, request and width k (the step's tokens: 1 + the drafted tokens, every one accepted): the "
          "steps, the step's time, the nine cards' GPU time and the rest (step less GPU), medians over the steps. A "
          "step of width 1 is a one-token pass with no draft found and no pipelining (a slot that can speculate never "
          "pipelines).\n",
          "| Load | Depth | Request | k | Steps | Step ms | GPU ms | Rest ms | Step ms a token |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (load, d, pas, w), v in agg.items():
        sm, gm = med([x[0] for x in v]), med([x[1] for x in v])
        rm = med([x[0] - x[1] for x in v])
        md.append(f"| {load} | {d} | {pas} | {w} | {len(v)} | {sm:.2f} | {gm:.2f} | {rm:.2f} | {sm / w:.2f} |")
        for k, val in (("steps", len(v)), ("step_ms", sm), ("gpu_ms", gm), ("rest_ms", rm)):
            rows_out.append((load, d, pas, w, k, val))
        for c in range(9):
            rows_out.append((load, d, pas, w, f"card{c}_gpu", med([x[2][c] for x in v])))
    md.append("\n### Each card's GPU ms a pass by k (the repeated request; the load with the most steps of that k)\n")
    best = {}
    for (load, d, pas, w), v in agg.items():
        if pas != "2":
            continue
        if (d, w) not in best or len(v) > len(agg[best[(d, w)]]):
            best[(d, w)] = (load, d, pas, w)
    keys = sorted(best, key=lambda x: (int(x[0]), x[1]))
    md.append("| Card | " + " | ".join(f"{d}, k={w} ({best[(d, w)][0]})" for d, w in keys) + " |")
    md.append("|---:|" + "---:|" * len(keys))
    for c in range(9):
        md.append(f"| {c} | " + " | ".join(f"{med([x[2][c] for x in agg[best[k]]]):.2f}" for k in keys) + " |")
    md.append("\n## Notes\n")
    md.extend(f"- {n}" for n in notes)
    with open(a.md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(a.tsv, "w") as f:
        f.write("load\tdepth\tpass\tk\tname\tvalue\n")
        for r in rows_out:
            v = r[5]
            f.write("\t".join(map(str, r[:5])) + "\t" + (f"{v:.4f}" if isinstance(v, float) else str(v)) + "\n")
    with open(os.path.splitext(a.tsv)[0] + "_steps.tsv", "w") as f:
        f.write("load\tdepth\tpass\tstep\tk\tstep_ms\tgpu_ms\t" + "\t".join(f"card{c}" for c in range(9)) + "\n")
        for r in per_step:
            f.write("\t".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
