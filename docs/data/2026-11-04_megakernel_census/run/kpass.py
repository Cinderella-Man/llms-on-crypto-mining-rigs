#!/usr/bin/env python3
# kpass.py PROF_BENCH.csv BENCH_PROF.json OUTDIR: the launch census of k-token passes from Q583's llama-bench trace
# (GGML_CUDA_PROFILE over `llama-bench -p 1,2,3,4,5 -n 0`, one warm-up and the repetitions per test, test after test),
# cut into passes as Q583's verify_split.py does: a pass ends when the next evaluation runs on a lower-numbered card.
# Per k: launch groups per pass (all, without weights), GPU ms per pass (all, without weights), the duration
# percentiles of the groups without weights, and the time below an assumed fixed cost per group; then a least-squares
# line through the k values (ms = a + b k) for the groups without weights and for the whole pass. Writes
# OUTDIR/kpass.csv and OUTDIR/kpass_fit.csv and prints both. Standard library only.
import collections, csv, json, os, statistics, sys

FIXED = [3.0, 5.0, 7.0]


def has_weights(label):
    _, _, w = label.partition(" ")
    return any(x.count(":") == 2 for x in w.split())


def passes_of(path):
    evals = collections.OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["device"]), int(row["eval"]))
            e = evals.setdefault(key, {"t": float(row["host_ms"]), "groups": []})
            e["groups"].append((row["label"], float(row["us"])))
    out, cur, last = [], None, None
    for (dev, _), e in sorted(evals.items(), key=lambda kv: kv[1]["t"]):
        if cur is None or dev < last:
            cur = {"devs": [], "groups": [], "t": []}
            out.append(cur)
        if not cur["devs"] or cur["devs"][-1] != dev:
            cur["devs"].append(dev)
        cur["groups"] += e["groups"]
        cur["t"].append(e["t"])
        last = dev
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else 0.0


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    trace, js, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    bench = json.load(open(js))
    ks = [t["n_prompt"] for t in bench]
    reps = len(bench[0]["samples_ns"])
    bp = [p for p in passes_of(trace) if len(p["devs"]) >= 5]
    rows = []
    for i, k in enumerate(ks):
        for p in bp[i * (1 + reps):(i + 1) * (1 + reps)][1:]:
            wl = [us for l, us in p["groups"] if not has_weights(l)]
            r = {"k": k, "groups": len(p["groups"]), "groups_weightless": len(wl),
                 "gpu_ms": sum(us for _, us in p["groups"]) / 1000, "weightless_ms": sum(wl) / 1000,
                 "weightless_p10_us": pct(wl, 0.1), "weightless_median_us": pct(wl, 0.5),
                 "weightless_p90_us": pct(wl, 0.9)}
            for fx in FIXED:
                r[f"weightless_below_{fx:g}us_ms"] = sum(min(us, fx) for us in wl) / 1000
            rows.append(r)
    keys = list(rows[0].keys())
    with open(f"{outdir}/kpass.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for k in ks:
            rk = [r for r in rows if r["k"] == k]
            w.writerow([k] + [f"{statistics.median(r[c] for r in rk):.2f}" for c in keys[1:]])
    print(open(f"{outdir}/kpass.csv").read())
    with open(f"{outdir}/kpass_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "per_pass_ms", "per_token_ms", "k_range"])
        for q in ("weightless_ms", "gpu_ms"):
            for lo in (1, 2):
                sel = [r for r in rows if r["k"] >= lo]
                a, b = fit([r["k"] for r in sel], [r[q] for r in sel])
                w.writerow([q, f"{a:.2f}", f"{b:.2f}", f"{lo}-{max(ks)}"])
    print(open(f"{outdir}/kpass_fit.csv").read())


if __name__ == "__main__":
    main()
