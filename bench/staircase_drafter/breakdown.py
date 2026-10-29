"""Where the head's acceptance comes from: score.py's per-position scores split by segment (reasoning, answer, tool
call), by what llama-server's lookups would guess there (source_split.py's sources), by Flint's top probability, by the
position in the session (context length and request number), by trace for agent_tokens.py's rows and by source x
Flint's top probability. Each cell has the
positions, their share, the head's acceptance (--metric) with a 95 % interval from resampling whole sessions, the
expected overlap of the two distributions (score.py's overlap, sum of min(p, q)) and, per source, the lookup's own
accuracy. "head positions" is what the head takes in the server's order with --pipe-guess staircase (copy, then
2-token continuations, then the head): the cont1 and none positions.

usage: python breakdown.py ROWS.jsonl SCORES.npz [--windows g9,g9miss] [--metric uniform]
ROWS.jsonl and SCORES.npz as for source_split.py; score.py must be the version that saves pmax and overlap.
Needs numpy.
"""
import argparse
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np

_spec = importlib.util.spec_from_file_location("source_split", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "source_split.py"))
source_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source_split)

SOURCES = ("copy", "cont2", "cont1", "none")
PMAX = (0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01)
CTX = (0, 4096, 8192, 16384, 32768, 65536, 1 << 30)
REQ = (0, 1, 2, 5, 10, 20, 1 << 30)


def ci(v, groups, n_boot=1000, seed=0):
    """mean and a 95 % interval from resampling whole sessions"""
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug))
    idx = np.random.default_rng(seed).integers(0, len(ug), (n_boot, len(ug)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def table(title, items, d, lookup=False):
    """items: (label, mask over d's positions)"""
    print(f"\n{title}\n")
    print("| | positions | share | head | 95 % | overlap" + (" | lookup right |" if lookup else " |"))
    print("|---|---|---|---|---|---" + ("|---|" if lookup else "|"))
    n_all = len(d["v"])
    for k, s in items:
        n = int(s.sum())
        if not n:
            continue
        m, lo, hi = ci(d["v"][s], d["seq"][s])
        row = f"| {k} | {n} | {n / n_all:.3f} | {m:.3f} | {lo:.3f}–{hi:.3f} | {d['ov'][s].mean():.3f}"
        print(row + (f" | {d['lk'][s].mean():.3f} |" if lookup else " |"))


def bins(x, edges, fmt, right=False):
    """(label, mask) per bin [a, b) of x, or (a, b] with right"""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append((fmt(a, b), (x > a) & (x <= b) if right else (x >= a) & (x < b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("scores")
    ap.add_argument("--windows", default="g9,g9miss")
    ap.add_argument("--metric", default="uniform", choices=("uniform", "coupled", "top"))
    args = ap.parse_args()
    z = np.load(args.scores)
    seq, j, seg, pmax = z["seq"], z["j"], z["seg"], z["pmax"]
    by_seq = defaultdict(list)
    for i in range(len(seq)):
        by_seq[int(seq[i])].append(i)
    n = len(seq)
    src = np.full(n, "", object)
    lk = np.zeros(n)
    pos = np.zeros(n, np.int64)
    req = np.zeros(n, np.int64)
    tag = np.full(n, "", object)   # the trace a row came from (agent_tokens.py's tag), if any
    for line in open(args.rows):
        r = json.loads(line)
        sid = int(r["id"].split("_")[-1])
        if sid not in by_seq:
            continue
        lab = source_split.labels(r)
        n_prompt = len(r["prompt_tokens"])
        ranges = r.get("gen_ranges") or [(n_prompt, n_prompt + len(r["gen_tokens"]))]
        starts = np.array([s for s, _ in ranges])
        for i in by_seq[sid]:
            x = lab.get(int(j[i]))
            if x is not None:
                src[i], lk[i] = x
            pos[i] = n_prompt + int(j[i])
            req[i] = int(np.searchsorted(starts, pos[i], side="right"))
            tag[i] = (r.get("tag") or "").rsplit("/", 1)[0]
    keep = src != ""
    print(f"{int(keep.sum())} of {n} scored positions have a guess source; metric {args.metric}; {args.scores}")
    for w in args.windows.split(","):
        if f"{w}/{args.metric}" not in z.files:
            print(f"\n(window {w} not in the file)")
            continue
        d = {"v": z[f"{w}/{args.metric}"][keep].astype(float), "ov": z[f"{w}/overlap"][keep].astype(float),
             "seq": seq[keep], "lk": lk[keep], "src": src[keep], "seg": seg[keep], "pmax": pmax[keep],
             "pos": pos[keep], "req": req[keep], "tag": tag[keep]}
        hp = np.isin(d["src"], ("cont1", "none"))
        dh = {k: v[hp] for k, v in d.items()}
        pfmt = lambda a, b: f"{a:g}–{min(b, 1):g}"   # noqa: E731
        print(f"\n## window {w}")
        table("All positions and the head's positions in the server's order",
              [("all", np.ones(len(hp), bool)), ("head positions", hp)], d)
        table("By guess source", [(s, d["src"] == s) for s in SOURCES], d, lookup=True)
        table("By segment, all positions", [(s, d["seg"] == s) for s in sorted(set(d["seg"].tolist()))], d)
        table("By segment, the head's positions", [(s, dh["seg"] == s) for s in sorted(set(dh["seg"].tolist()))], dh)
        table("By Flint's top probability, the head's positions", bins(dh["pmax"], PMAX, pfmt), dh)
        table("By position in the session (tokens of context), the head's positions",
              bins(dh["pos"], CTX, lambda a, b: f"{a // 1024}k–{b // 1024}k" if b < 1 << 30 else f"≥ {a // 1024}k"), dh)
        table("By request number in the session, the head's positions",
              bins(dh["req"], REQ, lambda a, b: f"{b}" if b == a + 1 else f"{a + 1}–{b}" if b < 1 << 30 else f"> {a}",
                   right=True), dh)
        if any(dh["tag"]):
            table("By trace, the head's positions", [(t, dh["tag"] == t) for t in sorted(set(dh["tag"].tolist()))], dh)
        pb = bins(d["pmax"], PMAX, pfmt)
        print("\nBy guess source x Flint's top probability: head / lookup right (positions)\n")
        print("| source | " + " | ".join(k for k, _ in pb) + " |")
        print("|---|" + "---|" * len(pb))
        for s in SOURCES:
            cells = []
            for _, m in pb:
                sel = (d["src"] == s) & m
                cells.append(f"{d['v'][sel].mean():.3f} / {d['lk'][sel].mean():.3f} ({int(sel.sum())})"
                             if sel.any() else "")
            print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
