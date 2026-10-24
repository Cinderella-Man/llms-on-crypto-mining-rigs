"""Speed per workload of pipelined speculation with trained staircase heads (Q20/D1), through Q87's discrete-event
simulator (bench/ngram_replay/pipeline_sim.py) on the held-out transcripts.

The guesser is the copy lookup (match >= 4) first, then a head. Each head guess is right with the per-position
probability score.py measured for it (Gumbel-coupled, or the top guess under ordinary sampling), with fixed draws per
position: the staircase window's probability while other guesses are in flight, the after-miss window's when none is
(the first guess after a miss or after the copy lookup's run ends). Drafter time is not charged (the card-1 and
spare-card placements of pipeline_drafter.md section 3.3 keep it off the cycle on six cards).

usage: python sim.py --gen TEST_GEN.jsonl --scores DIR [--out RESULTS.json] [--jobs N]
DIR holds score.py's scores-<head>.npz files.
"""
import argparse
import importlib.util
import json
import os
import random
import zlib
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pipeline_sim", os.path.join(HERE, "..", "ngram_replay", "pipeline_sim.py"))
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

# (setup index into ps.SETUPS, head, staircase window, after-miss window, metric)
RUNS = [
    (0, "A", "stair", "miss", "coupled"),
    (0, "A", "stair", "miss", "top"),
    (0, "A", "stair", "stair", "coupled"),
    (0, "B", "pre", "miss", "coupled"),
    (0, "B", "late", "miss", "coupled"),
    (0, "C", "plain", "plain", "coupled"),
    (0, "U", "miss", "miss", "coupled"),
    (0, "T", "plain", "plain", "coupled"),
    (1, "A", "nine", "miss_emb", "coupled"),
    (1, "B", "nine", "miss", "coupled"),
    (1, "C", "plain", "plain", "coupled"),
    (1, "T", "plain", "plain", "coupled"),
    (2, "A", "nine_light", "miss_emb", "coupled"),
    (2, "B", "nine_light", "miss", "coupled"),
    (2, "C", "plain", "plain", "coupled"),
    (2, "T", "plain", "plain", "coupled"),
]


class StairGuesser:
    """Right with probability a_stair[i] when other guesses are in flight, a_miss[i] when none is; fixed draws."""

    def __init__(self, a_stair, a_miss, seed=1):
        self.a_stair, self.a_miss, self.seed = a_stair, a_miss, seed

    def begin(self, full, index):
        rng = random.Random(self.seed ^ zlib.crc32(bytes(str(len(full)) + str(full[:16]), "ascii")))
        self.u = [rng.random() for _ in full]

    def next(self, full, n_v, overlay):
        i = n_v + len(overlay)
        a = (self.a_stair if overlay else self.a_miss)[i]
        if a is None:
            return None
        return full[i] if self.u[i] < a else -1


def per_position(scores, rows):
    """{(head, window, metric): {seq_id: full-length list of hit probabilities (None where no guess)}}."""
    out = {}
    for head, z in scores.items():
        seq, j = z["seq"], z["j"]
        names = {k.split("/")[0] for k in z.files if "/" in k}
        for name in names:
            for metric in ("coupled", "top"):
                v = z[f"{name}/{metric}"]
                d = {}
                for sid, r in enumerate(rows):
                    a = [None] * (len(r["prompt_tokens"]) + len(r["gen_tokens"]))
                    sel = seq == sid
                    for jj, x in zip(j[sel], v[sel]):
                        a[len(r["prompt_tokens"]) + int(jj)] = float(x)
                    d[sid] = a
                out[(head, name, metric)] = d
    return out


_SHARED = None


def run_rows(idx):
    rows, probs, configs, table = _SHARED
    runs = ps.Runs()
    for ri in idx:
        row = rows[ri]
        full = row["prompt_tokens"] + row["gen_tokens"]
        index = ps.SeqIndex(full)
        view = table.held_out(ri)
        for cfg, si, make, depth in configs:
            _, total, layers = ps.SETUPS[si]
            st = ps.stage_times(total, layers, ps.HOP_MS, ps.GAP_MS)
            times, stats = ps.simulate(full, len(row["prompt_tokens"]), make(ri, view), st, ps.HOP_MS, ps.GAP_MS,
                                       depth, index)
            runs.add(cfg, row, times, stats)
    return {"t": {c: dict(v) for c, v in runs.t.items()}, "n": {c: dict(v) for c, v in runs.n.items()},
            "stats": {c: dict(v) for c, v in runs.stats.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    rows = ps.replay.load(args.gen)
    scores = {}
    for f in sorted(os.listdir(args.scores)):
        if f.startswith("scores-") and f.endswith(".npz"):
            scores[f[7:-4]] = np.load(os.path.join(args.scores, f))
    probs = per_position(scores, rows)
    configs = []
    for si, (sname, total, layers) in enumerate(ps.SETUPS):
        depth = len(layers) - 1
        configs.append((f"{sname} | copy >= 4, else ngram-cache", si,
                        lambda ri, v: ps.ChainGuesser(ps.CopyGuesser(4), ps.CacheGuesser()), depth))
        configs.append((f"{sname} | copy >= 4, else table + request counts", si,
                        lambda ri, v: ps.ChainGuesser(ps.CopyGuesser(4), ps.TableGuesser(v, request=True)), depth))
        for p in (0.5, 0.6, 0.7):
            configs.append((f"{sname} | copy >= 4, else oracle p {p}", si,
                            lambda ri, v, p=p: ps.ChainGuesser(ps.CopyGuesser(4), ps.OracleGuesser(p)), depth))
    for si, head, win, miss, metric in RUNS:
        if (head, win, metric) not in probs or (head, miss, metric) not in probs:
            continue
        sname, total, layers = ps.SETUPS[si]
        a_s, a_m = probs[(head, win, metric)], probs[(head, miss, metric)]
        configs.append((f"{sname} | copy >= 4, else head {head} ({win}; after a miss {miss}; {metric})", si,
                        lambda ri, v, a_s=a_s, a_m=a_m: ps.ChainGuesser(ps.CopyGuesser(4),
                                                                        StairGuesser(a_s[ri], a_m[ri])),
                        len(layers) - 1))
    plain = {}
    for cfg, si, make, depth in configs:
        st = ps.stage_times(ps.SETUPS[si][1], ps.SETUPS[si][2], ps.HOP_MS, ps.GAP_MS)
        plain[cfg] = ps.GAP_MS + sum(st) + (len(st) - 1) * ps.HOP_MS
    table = ps.ContinuationTable(rows)
    global _SHARED
    _SHARED = (rows, probs, configs, table)
    tasks = [sl for sl in (list(range(len(rows)))[k::args.jobs] for k in range(args.jobs)) if sl]
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        parts = pool.map(run_rows, tasks)
    runs = ps.Runs()
    for part in parts:
        for c, d in part["t"].items():
            for k, x in d.items():
                runs.t[c][k] += x
        for c, d in part["n"].items():
            for k, x in d.items():
                runs.n[c][k] += x
        for c, d in part["stats"].items():
            runs.stats[c] += Counter(d)
    res = {}
    print("| Config | " + " | ".join(ps.KEY_NAMES[k] for k in ps.KEYS) + " | Right guesses | Wrong guesses | tok/s, all |")
    print("|---|" + "---|" * (len(ps.KEYS) + 3))
    for cfg, si, make, depth in configs:
        s = runs.stats[cfg]
        n_all = runs.n[cfg]["all"]
        sp = {k: round(runs.speedup(cfg, k, plain[cfg]), 4) for k in ps.KEYS}
        res[cfg] = {"speedup": sp, "hit_share": s["hit"] / n_all, "miss_share": s["miss"] / n_all,
                    "tok_s_all": round(sp["all"] * 1000 / plain[cfg], 2)}
        print(f"| {cfg} | " + " | ".join(f"{sp[k]:.2f}" for k in ps.KEYS) +
              f" | {100 * res[cfg]['hit_share']:.1f} % | {100 * res[cfg]['miss_share']:.1f} % | {res[cfg]['tok_s_all']} |")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
