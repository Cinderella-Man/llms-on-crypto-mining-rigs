#!/usr/bin/env python3
"""kpass_parts.py: where a whole-model k-token pass's time goes, from a llama-bench trace of the fork's light profiler
(kpass_prof.sh: llama-bench -p 1,5 -n 0 under GGML_CUDA_PROFILE, any GGML_CUDA_PROFILE_MODE).

  python3 -I kpass_parts.py --prof-summary <scripts/prof_summary.py> --event-us C --ks 1,5 --wall 1=82.4,5=217.1 \
      --out <dir> TAG=<prof-bench.csv> ...

Per trace (TAG, e.g. on_group):
  1. Each card's evaluations, in host-time order, are cut into two runs of llama-bench's tests (k = KS[0] first, then
     KS[1]) at the point that best separates their GPU times (two contiguous segments, least squares on log eval_us):
     a k-token pass evaluates each card's graph once, so the split needs no knowledge of warm-ups or deferred reads.
  2. Each k's rows go to <out>/<TAG>-k<k>.csv, and prof_summary.py summarizes them (--event-us C, --wall-ms the
     unprofiled T(k), --tol-pct 5) into <out>/<TAG>-k<k>.txt.
  3. For the modes that time launch groups (group, sample), the time per pass of each part of the model (S21's
     verify_split.py categories), C subtracted per span, with prof_summary.py's estimator (group: every span; sample:
     each group's mean over the evaluations that timed it): <out>/parts.tsv, one row per TAG, k and part.
  4. For graph mode, the cards' whole evaluations per pass (C subtracted once per evaluation) against the unprofiled
     wall: what the cards' timelines do not hold (the hops between cards, host work, gaps): rows "cards' GPU time" and
     "hops, host work and gaps" in parts.tsv.
Standard library only."""
import argparse, collections, csv, importlib.util, math, os, re, statistics, sys

PARTS = [  # verify_split.py's (S21), first match wins
    ("routed experts", r"^MUL_MAT_ID|_exps:"),
    ("shared expert", r"_shexp:"),
    ("hyper-connections (BF16)", r"\bhc_|DSV4_HC"),
    ("router", r"ffn_gate_inp:"),
    ("attention and recurrent projections", r" (attn_[a-z_]+|ssm_out|ssm_ba|ssm_alpha|ssm_beta|ssm_in|ssm_dt|eh_proj):"),
    ("Gated DeltaNet state and conv", r"GATED_DELTA_NET|SSM_CONV|ssm_conv|conv_states|ssm_states|_state"),
    ("attention (FLASH_ATTN_EXT)", r"FLASH_ATTN_EXT"),
    ("output layer", r" output:"),
    ("token embeddings", r"^GET_ROWS.* (token_embd|inp_embd)"),
]
NAMES = [p for p, _ in PARTS] + ["other ops without weights", "other weights"]


def part_of(label):
    for name, rx in PARTS:
        if re.search(rx, label):
            return name
    _, _, weights = label.partition(" ")
    return "other weights" if re.search(r":[a-z0-9_]+:\d+x\d+", weights) else "other ops without weights"


def split_rows(path):
    """-> header, {k_index: [rows]}: each card's evaluations cut into two host-time segments."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        header, rows = rd.fieldnames, list(rd)
    light = "eval_us" in header
    ev = collections.OrderedDict()
    for r in rows:
        key = (int(r["device"]), int(r["eval"]))
        e = ev.setdefault(key, {"t": float(r["host_ms"]), "us": 0.0, "rows": []})
        e["rows"].append(r)
        if light:
            if int(r["group"]) >= 0 and r["eval_us"] not in ("", "nan"):
                e["us"] = float(r["eval_us"])
        else:
            e["us"] += float(r["us"])
    out = {0: [], 1: []}
    for d in sorted({k[0] for k in ev}):
        es = sorted((e for k, e in ev.items() if k[0] == d), key=lambda e: e["t"])
        timed = [e for e in es if e["us"] > 1.0]  # an evaluation the profiler could not read has no time of its own
        logs = [math.log(e["us"]) for e in timed]
        best, t_split = None, float("inf")
        for s in range(1, len(timed)):
            a, b = logs[:s], logs[s:]
            sse = sum((x - statistics.fmean(a)) ** 2 for x in a) + sum((x - statistics.fmean(b)) ** 2 for x in b)
            if best is None or sse < best:
                best, t_split = sse, timed[s]["t"]
        for e in es:
            out[0 if e["t"] < t_split else 1].extend(e["rows"])
    return header, out


def parts_per_pass(ps, path, event_us):
    """prof_summary.py's estimator, summed by part: {part: ms per pass} for group and sample traces."""
    mode, evals, _ = ps.read_trace(path)
    devs = sorted({d for d, _ in evals})
    n_all = collections.Counter((e.device, e.graph) for e in evals.values())
    n_steps = max(1, len(evals) // max(1, len(devs)))
    hows = collections.Counter(e.how for e in evals.values())
    sel = [e for e in evals.values() if hows.get("replay", 0) == 0 or e.how == "replay"]
    by_dg = collections.defaultdict(list)
    for e in sel:
        by_dg[(e.device, e.graph)].append(e)
    acc = collections.Counter()
    for (d, g), es in by_dg.items():
        if mode == "sample":
            samples, labels = collections.defaultdict(list), {}
            for e in es:
                for group, label, us, _ in e.spans:
                    samples[group].append(us - event_us)
                    labels[group] = label
            for group, v in samples.items():
                acc[part_of(labels[group])] += statistics.fmean(v) * n_all[(d, g)] / n_steps
        else:
            w = n_all[(d, g)] / n_steps / len(es)
            for e in es:
                for _, label, us, _ in e.spans:
                    acc[part_of(label)] += (us - event_us) * w
    return mode, {p: acc[p] / 1000 for p in NAMES}


def graph_per_pass(ps, path, event_us):
    mode, evals, _ = ps.read_trace(path)
    devs = sorted({d for d, _ in evals})
    hows = collections.Counter(e.how for e in evals.values())
    sel = [e for e in evals.values() if hows.get("replay", 0) == 0 or e.how == "replay"]
    per_dev = collections.defaultdict(list)
    for e in sel:
        per_dev[e.device].append(e.eval_us - event_us * len(e.spans))
    return sum(statistics.median(v) for v in per_dev.values()) / 1000, len(devs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof-summary", required=True)
    ap.add_argument("--event-us", type=float, default=0.0)
    ap.add_argument("--ks", default="1,5")
    ap.add_argument("--wall", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("prof_summary", a.prof_summary)
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    ks = [int(x) for x in a.ks.split(",")]
    wall = {int(k): float(v) for k, v in (kv.split("=") for kv in a.wall.split(",") if kv)}
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for t in a.traces:
        tag, path = t.split("=", 1)
        header, seg = split_rows(path)
        for i, k in enumerate(ks):
            sub = os.path.join(a.out, f"{tag}-k{k}.csv")
            with open(sub, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(seg[i])
            with open(os.path.join(a.out, f"{tag}-k{k}.txt"), "w") as f:
                ps.summarize(sub, max_ms=1e9, event_us=a.event_us, wall_ms=wall.get(k), tol_pct=5, out=f)
            mode, _, _ = ps.read_trace(sub)
            if mode in ("group", "sample"):
                _, parts = parts_per_pass(ps, sub, a.event_us)
                for p in NAMES:
                    rows.append((tag, k, p, parts[p]))
                rows.append((tag, k, "sum of the parts", sum(parts.values())))
            elif mode == "graph":
                g_ms, _ = graph_per_pass(ps, sub, a.event_us)
                rows.append((tag, k, "cards' GPU time", g_ms))
                if k in wall:
                    rows.append((tag, k, "hops, host work and gaps", wall[k] - g_ms))
    with open(os.path.join(a.out, "parts.tsv"), "w") as f:
        f.write("trace\tk\tpart\tms_per_pass\n")
        for tag, k, p, v in rows:
            f.write(f"{tag}\t{k}\t{p}\t{v:.2f}\n")
    # the k = KS[1] / KS[0] ratio per trace and part
    by = {(tag, k, p): v for tag, k, p, v in rows}
    print("trace\tpart\t" + "\t".join(f"k{k}_ms" for k in ks) + f"\tratio_k{ks[1]}_k{ks[0]}\tgrowth_ms")
    for tag, k, p, v in rows:
        if k != ks[0]:
            continue
        hi = by.get((tag, ks[1], p))
        if hi is None:
            continue
        r = hi / v if v > 0.005 else float("nan")
        print(f"{tag}\t{p}\t{v:.2f}\t{hi:.2f}\t{r:.2f}\t{hi - v:.2f}")


if __name__ == "__main__":
    main()
