#!/usr/bin/env python3
"""graph_passes.py: how every graph evaluation of a deep-lane load ran (kernel by kernel, captured or replayed), pass
by pass, the first request at a depth against its repeat (R192, Q1561).

  python3 -I graph_passes.py GP.csv [--names d4096p1,d4096p2] [--gap-ms 3000] [--min-main 45] [--main-device 0]
      [--last-device 8] [--tsv passes.tsv] > passes.md

GP.csv is a GGML_CUDA_PROFILE file in graph mode (GGML_CUDA_PROFILE_MODE=graph): one line per evaluation and card with
the columns device, eval, host_ms, group, label, us, mode, graph, phase, how, groups, eval_groups, eval_us. "how" is
the CUDA backend's own decision for that evaluation (ggml_backend_cuda_graph_compute): "eager" (its kernels launched
one by one), "capture" (captured into a CUDA graph, then launched) or "replay" (a launch of graphs captured before).
eval_us is the evaluation's GPU time on its card, from an event before its first launch to one after its last.

The deep lane pauses before every request (PAUSE), so a gap of more than --gap-ms in host_ms cuts the file into
pieces; a piece with at least --min-main evaluations on --main-device is a generation request, named in order by
--names (as split_trace.py does in R121). Within a request, a main pass is one evaluation of the main model on every
card: the j-th evaluation on each of cards 0 to --last-device minus 1 (the MTP draft runs on the last card only) and,
on the last card, the first evaluation enqueued after the pass's evaluation on the card before it; the last card's
other evaluations are the draft's. Pass 0 reads the prompt's last 4 tokens in both requests (with
--ctx-checkpoints-incall the first request reads its 64 new tokens as 60 and then 4; the 60-token micro-batch is a
prompt graph, written only if that graph is evaluated again), pass 1 is the first verify pass after the prompt, and the
request's last pass verifies the reply's last tokens.

Prints: per request, the evaluations by how (main passes and draft apart); every main pass of both requests side by
side (how on each card as one letter, E eager, C capture, R replay; GPU time summed over the cards; the difference);
the passes whose how differs from a replay on any card, and the draft's evaluations that ran eagerly or were captured,
with their order in the request. The last evaluation of each graph in the file is never written by the profiler (its
events are read at the graph's next evaluation), so a request's very last pass can miss a card: such a pass is shown
with "-" for that card.
Standard library only.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict

LETTER = {"eager": "E", "capture": "C", "replay": "R", "patch": "P"}


def read_pieces(path, gap_ms):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"d": int(r["device"]), "eval": int(r["eval"]), "t": float(r["host_ms"]),
                             "graph": int(r["graph"]), "how": r["how"], "us": float(r["eval_us"]), "label": r["label"]})
            except (KeyError, TypeError, ValueError):
                continue  # a line cut at the server's exit
    # an evaluation is written when its graph is next evaluated, so the file is not in time order
    rows.sort(key=lambda e: e["t"])
    pieces, cur, tmax = [], [], None
    for e in rows:
        if cur and e["t"] > tmax + gap_ms:
            pieces.append(cur)
            cur = []
            tmax = None
        cur.append(e)
        tmax = e["t"] if tmax is None else max(tmax, e["t"])
    if cur:
        pieces.append(cur)
    return pieces


def passes_of(piece, main_dev, last_dev):
    by_dev = defaultdict(list)
    for e in piece:
        if e["label"] != "graph":  # "(not ready)" or "(unmatched)": no time
            e["us"] = float("nan")
        by_dev[e["d"]].append(e)
    for v in by_dev.values():
        v.sort(key=lambda e: e["eval"])
    n = min(len(by_dev[d]) for d in range(last_dev))
    passes = [[by_dev[d][j] for d in range(last_dev)] for j in range(n)]
    last = by_dev.get(last_dev, [])
    used = set()
    k = 0
    for p in passes:
        t_prev = p[-1]["t"]
        while k < len(last) and last[k]["t"] < t_prev:
            k += 1
        if k < len(last):
            p.append(last[k])
            used.add(k)
            k += 1
        else:
            p.append(None)
    draft = [e for i, e in enumerate(last) if i not in used]
    return passes, draft


def pass_how(p):
    return "".join(LETTER.get(e["how"], "?") if e else "-" for e in p)


def pass_ms(p):
    vals = [e["us"] for e in p if e and e["us"] == e["us"]]
    return sum(vals) / 1000.0, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gp")
    ap.add_argument("--names", default="d4096p1,d4096p2")
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--min-main", type=int, default=45)
    ap.add_argument("--main-device", type=int, default=0)
    ap.add_argument("--last-device", type=int, default=8)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    names = a.names.split(",")
    pieces = read_pieces(a.gp, a.gap_ms)
    reqs = []
    print(f"# How every graph evaluation ran, pass by pass ({a.gp})\n")
    print("| piece | rows | evaluations on card 0 | host_ms span | request |")
    print("|---|---|---|---|---|")
    for i, pc in enumerate(pieces):
        n0 = sum(1 for e in pc if e["d"] == a.main_device)
        name = ""
        if n0 >= a.min_main and len(reqs) < len(names):
            name = names[len(reqs)]
            reqs.append((name, pc))
        print(f"| {i} | {len(pc)} | {n0} | {min(e['t'] for e in pc):.1f} to {max(e['t'] for e in pc):.1f} | {name or '-'} |")
    if len(reqs) < len(names):
        print(f"\nonly {len(reqs)} generation requests of {len(names)} names")
        sys.exit(1)

    res = {}
    print("\n## Evaluations by how\n")
    print("| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, pc in reqs:
        passes, draft = passes_of(pc, a.main_device, a.last_device)
        res[name] = (passes, draft)
        mc = Counter(e["how"] for p in passes for e in p if e)
        dc = Counter(e["how"] for e in draft)
        other_m = sum(v for k, v in mc.items() if k not in ("eager", "capture", "replay"))
        other_d = sum(v for k, v in dc.items() if k not in ("eager", "capture", "replay"))
        print(f"| {name} | {len(passes)} | {mc['eager']} | {mc['capture']} | {mc['replay']} | {other_m} | {len(draft)} | "
              f"{dc['eager']} | {dc['capture']} | {dc['replay']} | {other_d} |")

    n1, n2 = names[0], names[1]
    p1, p2 = res[n1][0], res[n2][0]
    n = max(len(p1), len(p2))
    rows = []
    print(f"\n## Main passes, {n1} against {n2}\n")
    print("How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.\n")
    print(f"| pass | {n1} how | {n1} GPU ms | {n2} how | {n2} GPU ms | difference ms |")
    print("|---|---|---|---|---|---|")
    for j in range(n):
        a1 = p1[j] if j < len(p1) else None
        a2 = p2[j] if j < len(p2) else None
        h1, (m1, c1) = (pass_how(a1), pass_ms(a1)) if a1 else ("", (float("nan"), 0))
        h2, (m2, c2) = (pass_how(a2), pass_ms(a2)) if a2 else ("", (float("nan"), 0))
        tag = "prompt" if j == 0 else "first verify" if j == 1 else ""
        if j == len(p1) - 1:
            tag = (tag + f", {n1} last").strip(", ")
        if j == len(p2) - 1:
            tag = (tag + f", {n2} last").strip(", ")
        rows.append((j, tag, h1, m1, c1, h2, m2, c2))
        print(f"| {j}{' (' + tag + ')' if tag else ''} | {h1} | {m1:.2f} | {h2} | {m2:.2f} | {m1 - m2:+.2f} |")

    print("\n## The first verify pass and the last pass, card by card\n")
    print("GPU ms of each card's evaluation, how in parentheses.\n")
    print("| pass | request | " + " | ".join(f"card {d}" for d in range(a.last_device + 1)) + " |")
    print("|---|---|" + "---|" * (a.last_device + 1))
    for label, j1, j2 in (("first verify", 1, 1), ("last", len(p1) - 1, len(p2) - 1), ("second verify", 2, 2)):
        for name, ps, j in ((n1, p1, j1), (n2, p2, j2)):
            if j >= len(ps):
                continue
            cells = [f"{e['us'] / 1000.0:.2f} ({LETTER.get(e['how'], '?')})" if e else "-" for e in ps[j]]
            print(f"| {label} ({j}) | {name} | " + " | ".join(cells) + " |")

    print("\n## Passes not replayed on every card\n")
    print("| request | pass | how | GPU ms | cards eager | cards captured |")
    print("|---|---|---|---|---|---|")
    for name in (n1, n2):
        ps = res[name][0]
        for j, p in enumerate(ps):
            h = pass_how(p)
            if set(h) - {"R"}:
                print(f"| {name} | {j}{' (last)' if j == len(ps) - 1 else ''} | {h} | {pass_ms(p)[0]:.2f} | {h.count('E')} | {h.count('C')} |")

    print("\n## The draft's evaluations that ran eagerly or were captured\n")
    print("| request | draft evaluation (order) | how | graph | GPU ms | after main pass |")
    print("|---|---|---|---|---|---|")
    for name in (n1, n2):
        ps, dr = res[name]
        for i, e in enumerate(dr):
            if e["how"] == "replay":
                continue
            after = sum(1 for p in ps if p[-1] is not None and p[-1]["t"] <= e["t"]) - 1
            print(f"| {name} | {i} | {e['how']} | {e['graph']} | {e['us'] / 1000.0:.2f} | {after} |")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write(f"pass\ttag\thow_{n1}\tgpu_ms_{n1}\tcards_{n1}\thow_{n2}\tgpu_ms_{n2}\tcards_{n2}\n")
            for j, tag, h1, m1, c1, h2, m2, c2 in rows:
                f.write(f"{j}\t{tag}\t{h1}\t{m1:.3f}\t{c1}\t{h2}\t{m2:.3f}\t{c2}\n")


if __name__ == "__main__":
    main()
