#!/usr/bin/env python3
"""tk_table.py: Q873's tables from tk_depth.sh's llama-bench output (standard library only).

    python3 -I tk_table.py <dest dir> <out dir> [<out dir> ...]

Reads every <out dir>/bench-<arm>.jsonl (one llama-bench test a line: n_prompt = k, n_depth, samples_ns per
repetition; a later out dir's test replaces an earlier one's for the same arm, depth and k) and writes into
<dest dir>:
  tk.csv      per arm, depth and k: the cells the cache view holds, every repetition's ms per pass, llama-bench's own
              mean and standard deviation, and the settled value (the median of repetitions SETTLE+1 to the last)
              with the settled repetitions' lowest, highest and spread
  tables.md   T(k) per depth and arm; the sparse arm against the dense one per depth and k (the difference of the
              settled values, rep by rep, and the class); the attention's added ms per 1,000 cells of cache from the
              slope of the settled values between depths; what each depth adds to a k-token pass; which kernel
              mode 1 picks at each depth and k
Every test's first repetitions run on cards that are still ramping up after the idle seconds of making a context and
restoring the depth's state (repetition 1 is the slowest in every test of both arms), so the comparisons use the
repetitions after the first SETTLE (2). The class of sparse2 against off at one depth and k: "gain" when every settled
repetition of sparse2 is faster than every settled repetition of off, "loss" when every one is slower, "negligible"
when the two ranges overlap.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ARMS = ["off", "sparse2"]
LABEL = {"off": "off (=0)", "sparse2": "=2 (no threshold)"}
KEPT = 2051        # cells the indexer keeps per query token (the op's n_kv_max)
MIN_RATIO = 1.5    # GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO's default
PAD = 256          # the cache view grows in steps of 256 cells
SETTLE = 2         # repetitions left out of the settled value


def load(outs):
    rows = {}
    for out in outs:
        for arm in ARMS:
            f = out / f"bench-{arm}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                t = json.loads(line)
                if t.get("n_gen", 0) != 0:
                    continue
                rows[(arm, t["n_depth"], t["n_prompt"])] = {
                    "ms": [s / 1e6 for s in t["samples_ns"]], "commit": t.get("build_commit", ""),
                    "avg": t["avg_ns"] / 1e6, "sd": t["stddev_ns"] / 1e6, "src": out.name}
    return rows


def settled(ms):
    s = ms[SETTLE:]
    return {"value": statistics.median(s), "lo": min(s), "hi": max(s), "spread": max(s) - min(s), "reps": s}


def cells(depth, k):
    n = depth + k
    return (n + PAD - 1) // PAD * PAD


def main():
    dest, outs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    dest.mkdir(parents=True, exist_ok=True)
    rows = load(outs)
    depths = sorted({d for (_, d, _) in rows})
    ks = sorted({k for (_, _, k) in rows})
    st = {key: settled(v["ms"]) for key, v in rows.items()}

    with open(dest / "tk.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "depth", "k", "cells", "reps_ms", "bench_mean_ms", "bench_sd_ms", "settled_ms",
                    "settled_lowest_ms", "settled_highest_ms", "settled_spread_ms", "run"])
        for (arm, d, k), v in sorted(rows.items(), key=lambda x: (ARMS.index(x[0][0]), x[0][1], x[0][2])):
            s = st[(arm, d, k)]
            w.writerow([arm, d, k, cells(d, k), " ".join(f"{x:.2f}" for x in v["ms"]), f"{v['avg']:.2f}",
                        f"{v['sd']:.2f}", f"{s['value']:.2f}", f"{s['lo']:.2f}", f"{s['hi']:.2f}",
                        f"{s['spread']:.2f}", v["src"]])

    md = []
    commits = sorted({v["commit"] for v in rows.values()})
    md.append(f"# T(k) at depth (Q873), llama-bench build {', '.join(commits)}\n")
    md.append(f"ms per k-token pass. Each arm's cell lists the {len(next(iter(rows.values()))['ms'])} repetitions in "
              f"order, then the settled value (the median of repetitions {SETTLE + 1} on). The difference is "
              f"sparse2's settled value minus off's, then rep by rep for the settled repetitions.\n")
    for d in depths:
        md.append(f"\n## Depth {d:,} (the cache view holds {cells(d, 1):,} cells at k = 1)\n")
        here = [a for a in ARMS if any((a, d, k) in rows for k in ks)]
        md.append("| k | " + " | ".join(LABEL[a] for a in here) + " | =2 − off, settled | rep by rep | class |")
        md.append("|---:|" + "---|" * len(here) + "---:|---|---|")
        for k in ks:
            cols = []
            for a in here:
                if (a, d, k) not in rows:
                    cols.append("—")
                    continue
                ms = rows[(a, d, k)]["ms"]
                cols.append(", ".join(f"{x:.2f}" for x in ms) + f" (**{st[(a, d, k)]['value']:.2f}**)")
            if ("off", d, k) in st and ("sparse2", d, k) in st:
                a, b = st[("off", d, k)], st[("sparse2", d, k)]
                diff = b["value"] - a["value"]
                reps = ", ".join(f"{y - x:+.2f}" for x, y in zip(a["reps"], b["reps"]))
                cls = "gain" if b["hi"] < a["lo"] else ("loss" if b["lo"] > a["hi"] else "negligible")
                tail = f"{diff:+.2f} ({100 * diff / a['value']:+.1f} %) | {reps} | {cls}"
            else:
                tail = "— | — | —"
            md.append(f"| {k} | " + " | ".join(cols) + f" | {tail} |")

    md.append("\n## The attention's added cost per 1,000 cells (slope of the settled values between two depths)\n")
    pairs = [(depths[i], depths[j]) for i in range(len(depths)) for j in range(i + 1, len(depths))]
    md.append("| arm | k | " + " | ".join(f"{a:,} → {b:,}" for a, b in pairs) + " |")
    md.append("|---|---:|" + "---:|" * len(pairs))
    for arm in ARMS:
        for k in ks:
            vals = []
            for a, b in pairs:
                if (arm, a, k) in st and (arm, b, k) in st:
                    slope = (st[(arm, b, k)]["value"] - st[(arm, a, k)]["value"]) / ((cells(b, k) - cells(a, k)) / 1000)
                    vals.append(f"{slope:.3f}")
                else:
                    vals.append("—")
            md.append(f"| {LABEL[arm]} | {k} | " + " | ".join(vals) + " |")

    if 0 in depths:
        md.append("\n## What the depth adds to a k-token pass (settled T(k) at the depth minus settled T(k) at 0)\n")
        deep = [d for d in depths if d > 0]
        md.append("| k | " + " | ".join(f"{LABEL[a]} at {d:,}" for d in deep for a in ARMS) + " |")
        md.append("|---:|" + "---:|" * (len(deep) * len(ARMS)))
        for k in ks:
            vals = []
            for d in deep:
                for a in ARMS:
                    if (a, d, k) in st and (a, 0, k) in st:
                        vals.append(f"{st[(a, d, k)]['value'] - st[(a, 0, k)]['value']:.2f}")
                    else:
                        vals.append("—")
            md.append(f"| {k} | " + " | ".join(vals) + " |")

    md.append("\n## Which kernel mode 1 picks (by the source: sparse when cells ≥ 1.5 × 2,051 × k)\n")
    md.append("| depth | " + " | ".join(f"k = {k}" for k in ks) + " |")
    md.append("|---:|" + "---|" * len(ks))
    for d in depths:
        md.append(f"| {d:,} | " + " | ".join(
            "sparse" if cells(d, k) >= MIN_RATIO * KEPT * k else "dense" for k in ks) + " |")
    (dest / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
