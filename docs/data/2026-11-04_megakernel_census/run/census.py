#!/usr/bin/env python3
# census.py TRACE.csv NAME OUTDIR [MAX_EVAL_MS]: a GGML_CUDA_PROFILE trace (device,eval,host_ms,group,label,us) of a
# layer-split model decoding one token at a time, read as a launch census for a persistent per-card kernel (Q670).
#
# The profiler records a CUDA event between every two launch groups (one node or one group of fused nodes), so the
# groups of one evaluation tile that card's GPU timeline: a group's duration is the time from the end of the group
# before it to its own end, i.e. its launch gap, the event's own cost and its kernels' run time together. host_ms is
# the host clock when the evaluation's events were read (after a stream synchronize), so consecutive evaluations give
# the window each card had, and the window minus the card's GPU time is what happened off that card's GPU (the hop
# from the previous card through the host, the graph launch, the profiler's own synchronize and CSV write).
#
# A decoded token is one decode-sized evaluation (summed groups < MAX_EVAL_MS, default 60) on every card in device
# order. Writes to OUTDIR: tokens.csv (per token: wall, GPU, hops, host time before card 0), cards.csv (per card),
# ops.csv (per op class: groups and est. kernels per token, ms per token, duration percentiles, the size buckets),
# fixed.csv (the time below an assumed fixed cost per group, for a range of costs), labels.csv (every normalized
# launch-group label), and prints summary.md. Standard library only.
import collections, csv, os, re, statistics, sys

QUANT = {"q2_0", "q3_K", "q4_0", "q4_1", "q4_K", "q5_0", "q5_K", "q6_K", "q8_0", "iq4_xs", "iq4_nl", "iq2_xxs",
         "iq3_xxs", "iq2_s", "iq3_s", "q2_K", "iq1_s", "iq1_m", "iq2_xs", "mxfp4"}
EDGES = [5.0, 10.0, 20.0, 50.0]  # µs; buckets < 5, 5-10, 10-20, 20-50, >= 50
BUCKETS = ["<5", "5-10", "10-20", "20-50", ">=50"]
FIXED = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # assumed fixed µs per launch group


def bucket(us):
    for i, e in enumerate(EDGES):
        if us < e:
            return i
    return len(EDGES)


def weights(label):
    _, _, w = label.partition(" ")
    return [x.split(":") for x in w.split() if x.count(":") == 2]


def op_class(label):
    """weight products by weight type and role; ops without weights by their first op"""
    ops = label.split(" ")[0]
    ws = weights(label)
    if not ws:
        return "-", ops.split("+")[0]
    name, typ = ws[0][0], ws[0][1]
    if "exps" in name:
        role = "experts"
    elif "shexp" in name:
        role = "shared expert"
    elif name.startswith("hc_"):
        role = "hyper-connections"
    elif name == "output":
        role = "output layer"
    elif name.startswith("ffn_gate_inp"):
        role = "router"
    else:
        role = "attention/recurrent projections"
    return typ, role


def kernels_est(label):
    """kernels a group launches at one token, from the fork's code: a product with quantized weights first quantizes
    its activations to Q8_1 (one more kernel, shared inside a fused group); everything else one kernel"""
    ws = weights(label)
    return 2 if any(w[1] in QUANT for w in ws) else 1


def norm_label(label):
    ops, _, rest = label.partition(" ")
    if weights(label):
        return label
    return ops + " " + re.sub(r"\d+", "N", rest)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else 0.0


def main():
    path, name, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_ms = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    os.makedirs(outdir, exist_ok=True)
    evals = collections.OrderedDict()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (int(r["device"]), int(r["eval"]))
            e = evals.setdefault(k, {"dev": k[0], "host": float(r["host_ms"]), "groups": []})
            e["groups"].append((r["label"], float(r["us"])))
    seq = sorted(evals.values(), key=lambda e: (e["host"], e["dev"]))
    for e in seq:
        e["gpu"] = sum(us for _, us in e["groups"]) / 1000.0
    devs = sorted({e["dev"] for e in seq})

    # tokens: one decode-sized evaluation per card, in device order, with nothing else in between
    tokens, cur = [], None
    for i, e in enumerate(seq):
        if e["gpu"] >= max_ms:
            cur = None
            continue
        if e["dev"] == devs[0]:
            cur = [i]
        elif cur is not None and e["dev"] == devs[len(cur)] and i == cur[-1] + 1:
            cur.append(i)
        else:
            cur = None
        if cur is not None and len(cur) == len(devs):
            tokens.append(cur)
            cur = None

    rows = []
    for t, idx in enumerate(tokens):
        ev = [seq[i] for i in idx]
        prev = tokens[t - 1] if t > 0 else None
        consecutive = prev is not None and idx[0] == prev[-1] + 1
        gpu = sum(e["gpu"] for e in ev)
        hops = sum(ev[c]["host"] - ev[c - 1]["host"] - ev[c]["gpu"] for c in range(1, len(ev)))
        row = {"token": t, "consecutive": int(consecutive), "gpu_ms": gpu, "hops_ms": hops,
               "groups": sum(len(e["groups"]) for e in ev)}
        if consecutive:
            last = seq[prev[-1]]["host"]
            row["wall_ms"] = ev[-1]["host"] - last
            row["pre_ms"] = ev[0]["host"] - last - ev[0]["gpu"]
        else:
            row["wall_ms"] = row["pre_ms"] = ""
        rows.append(row)
    with open(f"{outdir}/tokens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["token", "consecutive", "groups", "gpu_ms", "hops_ms", "pre_ms", "wall_ms"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    n_tok = len(tokens)
    cons = [r for r in rows if r["consecutive"]]

    # per card
    card = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx in tokens:
        ev = [seq[i] for i in idx]
        for c, e in enumerate(ev):
            d = card[e["dev"]]
            d["gpu"].append(e["gpu"])
            d["groups"].append(len(e["groups"]))
            d["weightless"].append(sum(1 for l, _ in e["groups"] if not weights(l)))
            d["kernels"].append(sum(kernels_est(l) for l, _ in e["groups"]))
            d["small10"].append(sum(1 for _, us in e["groups"] if us < 10.0))
            d["wl_ms"].append(sum(us for l, us in e["groups"] if not weights(l)) / 1000.0)
            if c > 0:
                d["hop"].append(e["host"] - ev[c - 1]["host"] - e["gpu"])
    with open(f"{outdir}/cards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "groups_per_token", "weightless_groups", "kernels_est", "groups_under_10us",
                    "gpu_ms_per_token", "weightless_ms_per_token", "hop_ms_median"])
        for dv in devs:
            d = card[dv]
            w.writerow([dv, round(statistics.median(d["groups"])), round(statistics.median(d["weightless"])),
                        round(statistics.median(d["kernels"])), round(statistics.median(d["small10"])),
                        f"{statistics.mean(d['gpu']):.3f}", f"{statistics.mean(d['wl_ms']):.3f}",
                        f"{statistics.median(d['hop']):.3f}" if d["hop"] else ""])

    # per op class, over every decoded token's groups
    cls = collections.defaultdict(lambda: {"us": [], "kern": 0})
    labels = collections.defaultdict(list)
    all_us, wl_us = [], []
    for idx in tokens:
        for i in idx:
            for l, us in seq[i]["groups"]:
                c = cls[op_class(l)]
                c["us"].append(us)
                c["kern"] += kernels_est(l)
                labels[norm_label(l)].append(us)
                all_us.append(us)
                if not weights(l):
                    wl_us.append(us)
    with open(f"{outdir}/ops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["weights", "class", "groups_per_token", "kernels_est_per_token", "ms_per_token", "p10_us",
                    "median_us", "p90_us"] + [f"n_{b}" for b in BUCKETS] + [f"ms_{b}" for b in BUCKETS])
        for k, c in sorted(cls.items(), key=lambda kv: -sum(kv[1]["us"])):
            n = [0] * len(BUCKETS)
            s = [0.0] * len(BUCKETS)
            for us in c["us"]:
                n[bucket(us)] += 1
                s[bucket(us)] += us
            w.writerow([k[0], k[1], f"{len(c['us']) / n_tok:.1f}", f"{c['kern'] / n_tok:.1f}",
                        f"{sum(c['us']) / n_tok / 1000:.3f}", f"{pct(c['us'], 0.1):.2f}",
                        f"{pct(c['us'], 0.5):.2f}", f"{pct(c['us'], 0.9):.2f}"]
                       + [f"{x / n_tok:.1f}" for x in n] + [f"{x / n_tok / 1000:.3f}" for x in s])
    with open(f"{outdir}/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "groups_per_token", "ms_per_token", "min_us", "median_us"])
        for l, xs in sorted(labels.items(), key=lambda kv: -sum(kv[1])):
            w.writerow([l, f"{len(xs) / n_tok:.1f}", f"{sum(xs) / n_tok / 1000:.3f}", f"{min(xs):.2f}",
                        f"{pct(xs, 0.5):.2f}"])

    # the time below an assumed fixed cost per group: what a schedule without launches could at most remove from
    # each group if that cost were all launch gap, before paying for its own barriers
    with open(f"{outdir}/fixed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fixed_us_per_group", "weightless_fixed_ms", "weightless_rest_ms", "all_fixed_ms", "all_rest_ms"])
        for fx in FIXED:
            wf = sum(min(us, fx) for us in wl_us) / n_tok / 1000
            af = sum(min(us, fx) for us in all_us) / n_tok / 1000
            w.writerow([fx, f"{wf:.3f}", f"{sum(wl_us) / n_tok / 1000 - wf:.3f}", f"{af:.3f}",
                        f"{sum(all_us) / n_tok / 1000 - af:.3f}"])

    # summary
    def med(key, rs):
        xs = [r[key] for r in rs if r[key] != ""]
        return statistics.median(xs) if xs else float("nan")
    print(f"# {name}\n")
    print(f"- trace: {len(seq)} evaluations on {len(devs)} cards; {n_tok} decoded tokens, {len(cons)} of them "
          f"directly after another (wall time measurable)")
    print(f"- per token (medians): GPU {med('gpu_ms', rows):.2f} ms, launch groups {med('groups', rows):.0f}, "
          f"hops between cards {med('hops_ms', rows):.2f} ms, host before card 0 {med('pre_ms', cons):.2f} ms, "
          f"wall {med('wall_ms', cons):.2f} ms (profiler on); GPU busy {100 * med('gpu_ms', cons) / med('wall_ms', cons):.1f} % of the wall")
    print(f"- groups per token: {len(all_us) / n_tok:.0f} (est. kernels "
          f"{sum(c['kern'] for c in cls.values()) / n_tok:.0f}); without weights {len(wl_us) / n_tok:.0f}, "
          f"{sum(wl_us) / n_tok / 1000:.2f} ms")
    print("- group durations, µs: min {:.2f}, p1 {:.2f}, p10 {:.2f}, median {:.2f}; without weights: p10 {:.2f}, "
          "median {:.2f}, p90 {:.2f}".format(min(all_us), pct(all_us, 0.01), pct(all_us, 0.1), pct(all_us, 0.5),
                                             pct(wl_us, 0.1), pct(wl_us, 0.5), pct(wl_us, 0.9)))
    nb = [0] * len(BUCKETS)
    for us in all_us:
        nb[bucket(us)] += 1
    print("- groups per token by duration: " + ", ".join(f"{b} µs {x / n_tok:.0f}" for b, x in zip(BUCKETS, nb)))


if __name__ == "__main__":
    main()
