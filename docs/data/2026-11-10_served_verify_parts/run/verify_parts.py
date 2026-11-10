#!/usr/bin/env python3
"""verify_parts.py: the verify pass of a served speculative step by part and card group, from verify_prof.sh's
group-mode GGML_CUDA_PROFILE trace, beside a plain pass of the same size (R49, Q1211).

  python3 -I verify_parts.py --kpass-parts <2026-11-04_kpass_profile/run/kpass_parts.py> \
      --prof-summary <scripts/prof_summary.py> --event-us 7.6 --plain <release-k4.csv> --out <dir> <prof-verify.csv> \
      [--other-regex mtp_] [--head-regex ' output:'] [--gap-ms 3000] [--skip 3] [--tol 0.12] > report.md
  python3 -I verify_parts.py --kpass-parts ... --prof-summary ... --self-test

The trace has one line per launch group of every graph evaluation (device, eval, host_ms, group, label, us); host_ms is
taken when the evaluation's stream has been synchronized, so under this profiler the cards of a pass run one after
another. The cut follows 2026-11-10_served_step_split/run/step_split.py, on this file's clock:
  - Evaluations with a launch group whose label matches --other-regex are the MTP draft's own graphs. A draft pass
    ends with an evaluation on the draft's last card (the highest card that runs one).
  - A main pass is the j-th evaluation of the model's graph on every card. A pause of more than --gap-ms without an
    evaluation cuts the trace; the pieces with at least ten main passes are the requests (r1, r2, ...).
  - A step is a main pass and the draft passes before the next main pass (a request's last pass is no step).
    Generation steps are those after the large passes a request starts with (a main pass with more than 1.5 times the
    request's median GPU time is a prompt micro-batch). A typical step has the commonest number of draft passes, no
    stray draft evaluation and a main pass within --tol of their median GPU time; the first --skip steps of a request
    are left out.
The verify passes of each request's typical steps are the kept passes (trace verify-r<i>). --plain is a file of kept
plain passes in the same format (release_parts.py's release-k4.csv: prompt passes of four tokens, no draft loaded,
no output row; trace plain). For each trace and card group (every card but the last, the last card, all cards):
  - the parts of a pass by kpass_parts.py's estimator (parts_per_pass: every span of every kept evaluation,
    --event-us subtracted per span, by part of the model), called on the group's rows;
  - the same time by the launch groups' first op and by label;
  - on the last card, by position in the graph: before the output head's product (the first launch group whose label
    matches --head-regex), the product, and what follows it (the sampling chain);
  - where the time goes (BUCKETS, the first rule that matches a launch group wins): the head's product, the sampling
    chain after it, the output's hyper-connection products, the GATED_DELTA_NET and GDN_CONV groups, the copies of the
    PLE conv history, the routed experts, the shared expert with the router and the hyper-connections, everything else.
Writes to --out: <trace>.csv (the kept rows), <trace>.txt and <trace>-last.txt (prof_summary.py's report on all cards
and on the last card), parts.tsv, accounting.tsv (the buckets), by_op.tsv, labels.tsv, per_card.tsv,
last_card_groups.tsv (the last card's launch groups in graph order), passes.tsv (every main pass of the trace) and
cut.tsv; the Markdown report goes to stdout.
Standard library only."""
import argparse, collections, csv, importlib.util, os, re, statistics, sys, tempfile


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_evals(path, other_rx=None):
    """-> header, [evaluation]: {d, n, t, us, rows, other}, in the file's order."""
    ev = collections.OrderedDict()
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames
        for r in rd:
            try:
                key, t, us = (int(r["device"]), int(r["eval"])), float(r["host_ms"]), float(r["us"])
                int(r["group"])
            except (KeyError, TypeError, ValueError):
                continue  # a line cut at the server's exit
            if None in r or r["label"] is None:
                continue
            e = ev.setdefault(key, {"d": key[0], "n": key[1], "t": t, "us": 0.0, "rows": [], "other": False})
            e["rows"].append(r)
            e["us"] += us
            if other_rx is not None and other_rx.search(r["label"]):
                e["other"] = True
    return header, list(ev.values())


def cut_trace(evs, gap_ms, skip, tol):
    """-> passes (every main pass, annotated), requests ([[pass]]), info."""
    devs = sorted({e["d"] for e in evs})
    model = {d: sorted((e for e in evs if e["d"] == d and not e["other"]), key=lambda e: e["t"]) for d in devs}
    others = sorted((e for e in evs if e["other"]), key=lambda e: e["t"])
    counts = {d: len(model[d]) for d in devs}
    n = min(counts.values())
    info = {"devs": devs, "counts": counts, "others": len(others), "disorder": 0}
    draft_last = max((e["d"] for e in others), default=devs[-1])
    info["draft_last"] = draft_last
    info["other_by_card"] = dict(sorted(collections.Counter(e["d"] for e in others).items()))
    passes = []
    for j in range(n):
        es = [model[d][j] for d in devs]
        if any(b["t"] < a["t"] for a, b in zip(es, es[1:])) or (passes and es[0]["t"] < passes[-1]["t1"]):
            info["disorder"] += 1
        passes.append({"j": j, "evs": es, "t0": es[0]["t"], "t1": es[-1]["t"], "us": sum(e["us"] for e in es),
                       "seg": 0, "request": "", "i": -1, "kind": "", "typical": 0, "n_dp": 0, "stray": 0})
    # the pieces: a pause of more than gap_ms between two consecutive evaluations of any graph
    ts = sorted(e["t"] for e in evs)
    cuts = [b for a, b in zip(ts, ts[1:]) if b - a > gap_ms]
    for p in passes:
        p["seg"] = sum(1 for c in cuts if c <= p["t0"])
    # the draft passes after each main pass, up to the next main pass of the same piece
    k = 0
    for p, q in zip(passes, passes[1:] + [None]):
        hi = q["t0"] if q is not None and q["seg"] == p["seg"] else float("inf")
        while k < len(others) and others[k]["t"] <= p["t1"]:
            k += 1
        stray = 0
        while k < len(others) and others[k]["t"] < hi and (q is None or q["seg"] == p["seg"] or
                                                          sum(1 for c in cuts if c <= others[k]["t"]) == p["seg"]):
            if others[k]["d"] == draft_last:
                p["n_dp"] += 1
                stray = 0
            else:
                stray += 1
            k += 1
        p["stray"] = stray
    segs = collections.OrderedDict()
    for p in passes:
        segs.setdefault(p["seg"], []).append(p)
    info["pieces"] = [len(v) for v in segs.values()]
    requests = [v for v in segs.values() if len(v) >= 10]
    info["requests"] = []
    for r, ps in enumerate(requests, 1):
        name = f"r{r}"
        steps = ps[:-1]  # the last main pass has no next one: no step
        med = statistics.median(p["us"] for p in steps)
        g0 = 0
        while g0 < len(steps) and steps[g0]["us"] > 1.5 * med:
            g0 += 1
        gen = steps[g0:]
        mode = collections.Counter(p["n_dp"] for p in gen).most_common(1)[0][0]
        mmed = statistics.median(p["us"] for p in gen if p["n_dp"] == mode)
        for p in ps:
            p["request"], p["kind"] = name, "last"
        for p in steps[:g0]:
            p["kind"] = "prompt"
        for i, p in enumerate(gen):
            p["i"], p["kind"] = i, "step"
            p["typical"] = int(i >= skip and p["n_dp"] == mode and p["stray"] == 0 and abs(p["us"] / mmed - 1.0) <= tol)
        typ = [p for p in gen if p["typical"]]
        gaps = [b["t0"] - a["t0"] for a, b in zip(typ, typ[1:]) if b["j"] == a["j"] + 1]
        info["requests"].append({"name": name, "passes": len(ps), "prompt": g0, "steps": len(gen), "mode": mode,
                                 "by_dp": dict(sorted(collections.Counter(p["n_dp"] for p in gen).items())),
                                 "typical": len(typ), "median_raw_ms": mmed / 1000,
                                 "step_ms": statistics.median(gaps) if gaps else float("nan"),
                                 "t0": ps[0]["t0"], "t1": ps[-1]["t1"]})
    return passes, requests, info


def first_op(label):
    return label.split(" ", 1)[0].split("+")[0]


# Where a pass's time goes, for the table "where the difference goes": the first rule that matches a launch group wins.
# A rule sees the label, the part (kpass_parts.py's), and on the last card whether the group is the head's product or
# follows it in the graph.
BUCKETS = [
    ("the head's product", lambda lab, part, head, after: head),
    ("the sampling chain after the product", lambda lab, part, head, after: after),
    ("the output's hyper-connection products", lambda lab, part, head, after: "output_hc_" in lab),
    ("GATED_DELTA_NET", lambda lab, part, head, after: first_op(lab) == "GATED_DELTA_NET"),
    ("GDN_CONV", lambda lab, part, head, after: first_op(lab) == "GDN_CONV"),
    ("copies of the PLE conv history (cache_ple_r)", lambda lab, part, head, after: "cache_ple_r" in lab),
    ("routed experts", lambda lab, part, head, after: part == "routed experts"),
    ("shared expert, router, hyper-connections", lambda lab, part, head, after:
        part in ("shared expert", "router", "hyper-connections (BF16)")),
    ("everything else", lambda lab, part, head, after: True),
]


def buckets(kp, items):
    """items: (label, ms, head, after) -> {bucket: ms}."""
    out = collections.OrderedDict((b, 0.0) for b, _ in BUCKETS)
    for lab, ms, head, after in items:
        part = kp.part_of(lab)
        out[next(b for b, f in BUCKETS if f(lab, part, head, after))] += ms
    return out


def norm_label(label):
    return re.sub(r"\bnode_\d+", "node", label)


def measure(name, header, kept, devs, kp, ps, event_us, head_rx, out):
    """The tables of one trace: kept is a list of passes, each a list of one evaluation per card."""
    n_pass = len(kept)
    groups = [("first", devs[:-1]), ("last", devs[-1:]), ("all", devs)]
    res = {"name": name, "passes": n_pass, "parts": {}, "ops": {}, "labels": {}, "per_card": {}, "raw_card": {}}
    path = os.path.join(out, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for d in devs:
            for es in kept:
                for e in es:
                    if e["d"] == d:
                        w.writerows(e["rows"])
    with open(os.path.join(out, name + ".txt"), "w") as f:
        ps.summarize(path, max_ms=1e9, event_us=event_us, top=40, out=f)
    for g, ds in groups:
        if g == "all":
            sub = path
        else:
            sub = os.path.join(out, f".{name}-{g}.csv")
            with open(sub, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for es in kept:
                    for e in es:
                        if e["d"] in ds:
                            w.writerows(e["rows"])
        _, parts = kp.parts_per_pass(ps, sub, event_us)
        res["parts"][g] = parts
        if g == "last":
            with open(os.path.join(out, name + "-last.txt"), "w") as f:
                ps.summarize(sub, max_ms=1e9, event_us=event_us, top=60, out=f)
        if g != "all":
            os.remove(sub)
        ops, labels = collections.Counter(), collections.Counter()
        n_ops, n_labels = collections.Counter(), collections.Counter()
        for es in kept:
            for e in es:
                if e["d"] not in ds:
                    continue
                for r in e["rows"]:
                    us = float(r["us"]) - event_us
                    ops[first_op(r["label"])] += us / n_pass / 1000
                    n_ops[first_op(r["label"])] += 1 / n_pass
                    labels[norm_label(r["label"])] += us / n_pass / 1000
                    n_labels[norm_label(r["label"])] += 1 / n_pass
        res["ops"][g], res["labels"][g] = (ops, n_ops), (labels, n_labels)
    for d in devs:
        es = [e for p in kept for e in p if e["d"] == d]
        res["per_card"][d] = statistics.fmean(e["us"] - event_us * len(e["rows"]) for e in es) / 1000
        res["raw_card"][d] = statistics.median(e["us"] for e in es) / 1000
    # the last card's launch groups in graph order, over the evaluations with the commonest graph
    last = [e for p in kept for e in p if e["d"] == devs[-1]]
    seqs = collections.Counter(tuple(r["label"] for r in e["rows"]) for e in last)
    seq, n_seq = seqs.most_common(1)[0]
    same = [e for e in last if tuple(r["label"] for r in e["rows"]) == seq]
    ms = [statistics.fmean(float(e["rows"][i]["us"]) - event_us for e in same) / 1000 for i in range(len(seq))]
    head = [i for i, lab in enumerate(seq) if head_rx.search(lab)]
    pos = {"before the head's product": sum(ms[:head[0]]) if head else sum(ms),
           "the head's product": sum(ms[i] for i in head),
           "after the head's product": sum(ms[head[-1] + 1:]) if head else 0.0}
    # the groups between and after the head's products count as after the first one
    if head:
        pos["after the head's product"] = sum(ms[head[0]:]) - pos["the head's product"]
    res.update(seq=list(zip(seq, ms)), seq_evals=(n_seq, len(last)), head=head, pos=pos)
    # where the time goes: the last card from its commonest graph in order, the other cards from their labels
    labels, _ = res["labels"]["first"]
    res["buckets"] = {"first": buckets(kp, [(lab, v, False, False) for lab, v in labels.items()]),
                      "last": buckets(kp, [(lab, v, i in head, bool(head) and i > head[0] and i not in head)
                                           for i, (lab, v) in enumerate(zip(seq, ms))])}
    return res


def table(cols, rows, file=sys.stdout):
    print("| " + " | ".join(cols) + " |", file=file)
    print("|---" + "|---:" * (len(cols) - 1) + "|", file=file)
    for r in rows:
        print("| " + " | ".join(r) + " |", file=file)
    print(file=file)


def run(a, kp, ps, file=sys.stdout):
    os.makedirs(a.out, exist_ok=True)
    other_rx, head_rx = re.compile(a.other_regex), re.compile(a.head_regex)
    header, evs = read_evals(a.trace, other_rx)
    passes, requests, info = cut_trace(evs, a.gap_ms, a.skip, a.tol)
    devs = info["devs"]
    P = lambda *x: print(*x, file=file)
    P(f"{len(evs)} evaluations on cards {devs[0]} to {devs[-1]}: the model's graph " +
      ", ".join(f"{d}: {c}" for d, c in info["counts"].items()) + f"; the draft's graphs {info['others']} (" +
      ", ".join(f"card {d}: {c}" for d, c in info["other_by_card"].items()) + f"), its last card {info['draft_last']}")
    P(f"main passes per piece (cut at pauses over {a.gap_ms:g} ms): " + ", ".join(str(x) for x in info["pieces"]) +
      f"; passes whose cards are out of order: {info['disorder']}")
    P()
    table(["request", "main passes", "prompt passes", "generation steps", "by draft passes a step", "typical steps",
           "median raw GPU time of a typical verify pass, ms", "from pass to pass under the profiler, ms"],
          [[q["name"], str(q["passes"]), str(q["prompt"]), str(q["steps"]),
            ", ".join(f"{k}: {v}" for k, v in q["by_dp"].items()), str(q["typical"]), f"{q['median_raw_ms']:.2f}",
            f"{q['step_ms']:.2f}"] for q in info["requests"]], file)
    results = []
    for q, ps_ in zip(info["requests"], requests):
        kept = [p["evs"] for p in ps_ if p["typical"]]
        results.append(measure(f"verify-{q['name']}", header, kept, devs, kp, ps, a.event_us, head_rx, a.out))
    plain = None
    if a.plain:
        ph, pevs = read_evals(a.plain)
        pdevs = sorted({e["d"] for e in pevs})
        if pdevs != devs:
            sys.exit(f"--plain holds cards {pdevs}, the trace {devs}")
        per = {d: sorted((e for e in pevs if e["d"] == d), key=lambda e: e["t"]) for d in pdevs}
        n = min(len(v) for v in per.values())
        plain = measure("plain", ph, [[per[d][j] for d in pdevs] for j in range(n)], devs, kp, ps, a.event_us, head_rx,
                        a.out)
        os.remove(os.path.join(a.out, "plain.csv"))  # a copy of --plain
    traces = ([plain] if plain else []) + results
    ref = plain or results[0]
    gname = {"first": f"cards {devs[0]} to {devs[-2]}" if len(devs) > 2 else f"card {devs[0]}", "last": f"card {devs[-1]}",
             "all": "all cards"}
    dcols = [f"{t['name']} − {ref['name']}" for t in traces if t is not ref]
    cols = [f"{t['name']} ({t['passes']} passes)" for t in traces] + dcols
    for g in ("first", "last", "all"):
        P(f"**GPU time of a pass by part, ms: {gname[g]}**")
        P()
        names = [p for p in kp.NAMES if any(abs(t["parts"][g][p]) > 0.004 for t in traces)]
        rows = [[p] + [f"{t['parts'][g][p]:.2f}" for t in traces] +
                [f"{t['parts'][g][p] - ref['parts'][g][p]:+.2f}" for t in traces if t is not ref] for p in names]
        tot = [sum(t["parts"][g].values()) for t in traces]
        rows.append(["**sum**"] + [f"**{x:.2f}**" for x in tot] +
                    [f"**{x - tot[0]:+.2f}**" for t, x in zip(traces, tot) if t is not ref])
        table(["Part"] + cols, rows, file)
    for g in ("first", "last"):
        P(f"**Where the difference goes, ms a pass: {gname[g]}**" +
          (" (its commonest graph, launch groups in order)" if g == "last" else ""))
        P()
        keys = [b for b, _ in BUCKETS if any(abs(t["buckets"][g][b]) > 0.004 for t in traces)]
        rows = [[b] + [f"{t['buckets'][g][b]:.2f}" for t in traces] +
                [f"{t['buckets'][g][b] - ref['buckets'][g][b]:+.2f}" for t in traces if t is not ref] for b in keys]
        tot = [sum(t["buckets"][g].values()) for t in traces]
        rows.append(["**sum**"] + [f"**{x:.2f}**" for x in tot] +
                    [f"**{x - tot[0]:+.2f}**" for t, x in zip(traces, tot) if t is not ref])
        table(["Where"] + cols, rows, file)
    P("**The last card by position in its graph, ms** (the evaluations with the commonest graph: " +
      ", ".join(f"{t['name']} {t['seq_evals'][0]} of {t['seq_evals'][1]}, {len(t['seq'])} launch groups" for t in traces) + ")")
    P()
    rows = [[k] + [f"{t['pos'][k]:.2f}" for t in traces] + [f"{t['pos'][k] - ref['pos'][k]:+.2f}" for t in traces if t is not ref]
            for k in ref["pos"]]
    table(["Position"] + cols, rows, file)
    for g, top in (("first", 14), ("last", 14)):
        P(f"**By the launch groups' first op, ms a pass (launch groups a pass): {gname[g]}**")
        P()
        keys = sorted({k for t in traces for k in t["ops"][g][0]},
                      key=lambda k: -abs(traces[-1]["ops"][g][0][k] - ref["ops"][g][0][k]))
        rows = [[k] + [f"{t['ops'][g][0][k]:.2f} ({t['ops'][g][1][k]:.0f})" for t in traces] +
                [f"{t['ops'][g][0][k] - ref['ops'][g][0][k]:+.2f}" for t in traces if t is not ref] for k in keys[:top]]
        rest = keys[top:]
        if rest:
            rows.append([f"the other {len(rest)} ops"] + [f"{sum(t['ops'][g][0][k] for k in rest):.2f}" for t in traces] +
                        [f"{sum(t['ops'][g][0][k] - ref['ops'][g][0][k] for k in rest):+.2f}" for t in traces if t is not ref])
        table(["First op"] + cols, rows, file)
    for t in results[-1:]:
        if t["head"]:
            i0 = max(0, t["head"][0] - 3)
            P(f"**The last card's launch groups from three before the head's product on, {t['name']}, ms**")
            P()
            table(["Group", "Label", "ms"], [[str(i), "`" + lab[:110] + "`", f"{ms:.3f}"]
                                              for i, (lab, ms) in list(enumerate(t["seq"]))[i0:]], file)
    P("**Per card, ms a pass (the profiler's cost subtracted per span)**")
    P()
    table(["Card"] + cols, [[str(d)] + [f"{t['per_card'][d]:.2f}" for t in traces] +
                            [f"{t['per_card'][d] - ref['per_card'][d]:+.2f}" for t in traces if t is not ref] for d in devs] +
          [["**all**"] + [f"**{sum(t['per_card'].values()):.2f}**" for t in traces] +
           [f"**{sum(t['per_card'].values()) - sum(ref['per_card'].values()):+.2f}**" for t in traces if t is not ref]], file)

    w = lambda name: open(os.path.join(a.out, name), "w")
    with w("parts.tsv") as f:
        f.write("trace\tcards\tpasses\tpart\tms_per_pass\n")
        for t in traces:
            for g in ("first", "last", "all"):
                for p in kp.NAMES:
                    f.write(f"{t['name']}\t{gname[g]}\t{t['passes']}\t{p}\t{t['parts'][g][p]:.3f}\n")
                f.write(f"{t['name']}\t{gname[g]}\t{t['passes']}\tsum of the parts\t{sum(t['parts'][g].values()):.3f}\n")
            for k, v in t["pos"].items():
                f.write(f"{t['name']}\t{gname['last']}\t{t['passes']}\tby position: {k}\t{v:.3f}\n")
    for fname, key, col in (("by_op.tsv", "ops", "first_op"), ("labels.tsv", "labels", "label")):
        with w(fname) as f:
            f.write(f"trace\tcards\t{col}\tgroups_per_pass\tms_per_pass\n")
            for t in traces:
                for g in ("first", "last"):
                    ms, n = t[key][g]
                    for k, v in sorted(ms.items(), key=lambda kv: -kv[1]):
                        f.write(f"{t['name']}\t{gname[g]}\t{k}\t{n[k]:.2f}\t{v:.4f}\n")
    with w("accounting.tsv") as f:
        f.write("trace\tcards\tpasses\twhere\tms_per_pass\n")
        for t in traces:
            for g in ("first", "last"):
                for b, v in t["buckets"][g].items():
                    f.write(f"{t['name']}\t{gname[g]}\t{t['passes']}\t{b}\t{v:.3f}\n")
    with w("per_card.tsv") as f:
        f.write("trace\tcard\tpasses\tms_per_pass\tmedian_raw_ms\n")
        for t in traces:
            for d in devs:
                f.write(f"{t['name']}\t{d}\t{t['passes']}\t{t['per_card'][d]:.3f}\t{t['raw_card'][d]:.3f}\n")
    with w("last_card_groups.tsv") as f:
        f.write("trace\tgroup\tposition\tlabel\tms\n")
        for t in traces:
            h = t["head"]
            for i, (lab, ms) in enumerate(t["seq"]):
                where = "before" if not h or i < h[0] else "head" if i in h else "after"
                f.write(f"{t['name']}\t{i}\t{where}\t{lab}\t{ms:.4f}\n")
    with w("passes.tsv") as f:
        f.write("pass\tpiece\trequest\tkind\tstep\ttypical\tdraft_passes_after\tstray_draft_evals\tfirst_card_host_ms\t"
                "last_card_host_ms\traw_ms_all_cards\t" + "\t".join(f"raw_ms_card{d}" for d in devs) + "\n")
        for p in passes:
            f.write("%d\t%d\t%s\t%s\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.2f\t" % (
                p["j"], p["seg"], p["request"], p["kind"], p["i"], p["typical"], p["n_dp"], p["stray"], p["t0"], p["t1"],
                p["us"] / 1000) + "\t".join(f"{e['us'] / 1000:.2f}" for e in p["evs"]) + "\n")
    with w("cut.tsv") as f:
        f.write("request\tmain_passes\tprompt_passes\tgeneration_steps\tdraft_passes_a_step\ttypical_steps\t"
                "median_raw_ms_typical_verify\tprofiled_ms_pass_to_pass\tfirst_host_ms\tlast_host_ms\n")
        for q in info["requests"]:
            f.write("%s\t%d\t%d\t%d\t%s\t%d\t%.2f\t%.2f\t%.1f\t%.1f\n" % (
                q["name"], q["passes"], q["prompt"], q["steps"], ",".join(f"{k}:{v}" for k, v in q["by_dp"].items()),
                q["typical"], q["median_raw_ms"], q["step_ms"], q["t0"], q["t1"]))
    return info, traces


def self_test(a, kp, ps):
    """A synthetic trace of three cards: one load evaluation, two requests (a prompt pass, a small pass, steps of a
    verify pass and four draft passes; one step with two draft passes, one slow verify pass), and a plain file."""
    C = 8.0
    layer = [("RMS_NORM+MUL node_%d", 20.0), ("MUL_MAT attn_q:q4_0:2560x4096", 300.0), ("GATED_DELTA_NET node_%d", 50.0),
             ("MUL_MAT_ID ffn_up_exps:q2_0:2560x512", 200.0), ("ADD node_%d", 10.0)]
    tail = [("RMS_NORM+MUL result_norm", 15.0), ("MUL_MAT output:q6_k:2560x151936", 2400.0), ("TOP_K node_%d", 700.0),
            ("SOFT_MAX node_%d", 90.0)]
    extra = {"GATED_DELTA_NET": 40.0}  # what the verify pass adds per launch group on every card
    rows, n_eval, t = [], collections.Counter(), [0.0]

    def ev(d, groups, scale=1.0):
        t[0] += sum(us for _, us in groups) * scale / 1000 + 0.2
        for k, (lab, us) in enumerate(groups):
            rows.append([d, n_eval[d], f"{t[0]:.3f}", k, lab % (100 * d + k) if "%d" in lab else lab, f"{us * scale + C:.2f}"])
        n_eval[d] += 1

    def main_pass(verify, scale=1.0):
        for d in range(3):
            g = [(lab, us + (extra.get(lab.split(" ")[0], 0.0) if verify else 0.0)) for lab, us in layer]
            ev(d, g + (tail if verify and d == 2 else []), scale)

    def draft(n):
        for _ in range(n):
            ev(1, [("GET_ROWS mtp_tok_embd", 30.0)])
            ev(2, [("RMS_NORM+MUL norm", 10.0), ("REPEAT mtp_enorm", 5.0), ("MUL_MAT mtp_head:q8_0:2560x8192", 900.0)])

    with tempfile.TemporaryDirectory() as tmp:
        main_pass(False, 5.0)                      # the load
        for _ in range(2):
            t[0] += 5000.0
            main_pass(False, 12.0)                 # the prompt
            main_pass(True); draft(4)              # a small pass that looks like a step
            for i in range(16):
                main_pass(True, 1.3 if i == 9 else 1.0)
                draft(2 if i == 5 else 4)
            main_pass(True, 6.0)                   # the last pass: one-off launches, no step
        trace = os.path.join(tmp, "prof.csv")
        with open(trace, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["device", "eval", "host_ms", "group", "label", "us"])
            w.writerows(rows)
            f.write("2,999,1.0,3,MUL_MAT cut")     # a line cut at the server's exit
        rows.clear(); n_eval.clear(); t[0] = 0.0
        for _ in range(7):
            main_pass(False)
        plain = os.path.join(tmp, "plain.csv")
        with open(plain, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["device", "eval", "host_ms", "group", "label", "us"])
            w.writerows(rows)
        b = argparse.Namespace(trace=trace, plain=plain, out=os.path.join(tmp, "out"), other_regex="mtp_",
                               head_regex=" output:", gap_ms=3000.0, skip=3, tol=0.12, event_us=C)
        with open(os.path.join(tmp, "report.md"), "w") as f:
            info, traces = run(b, kp, ps, f)
        checks, fails = 0, []

        def check(cond, what):
            nonlocal checks
            checks += 1
            if not cond:
                fails.append(what)

        near = lambda x, y: abs(x - y) < 1e-6
        check(info["pieces"] == [1, 19, 19], f"pieces {info['pieces']}")
        check(len(info["requests"]) == 2, "two requests")
        for q in info["requests"]:
            check(q["prompt"] == 1 and q["steps"] == 17, f"{q['name']}: {q['prompt']} prompt passes, {q['steps']} steps")
            # 17 steps: the first three skipped, one with two draft passes, one slow
            check(q["typical"] == 12 and q["mode"] == 4, f"{q['name']}: {q['typical']} typical steps")
        pl, v1, v2 = traces
        check(pl["passes"] == 7 and v1["passes"] == 12 and v2["passes"] == 12, "passes per trace")
        lay = sum(us for _, us in layer) / 1000
        check(near(sum(pl["parts"]["all"].values()), 3 * lay), "plain: all cards")
        check(near(sum(pl["parts"]["last"].values()), lay), "plain: the last card")
        for v in (v1, v2):
            check(near(sum(v["parts"]["first"].values()) - sum(pl["parts"]["first"].values()), 2 * 0.040), "cards 0 to 1: +0.08")
            check(near(v["parts"]["first"]["Gated DeltaNet state and conv"] - pl["parts"]["first"]["Gated DeltaNet state and conv"],
                       0.080), "the difference is in the Gated DeltaNet part")
            check(near(sum(v["parts"]["last"].values()) - sum(pl["parts"]["last"].values()), 0.040 + 3.205), "the last card: +3.245")
            check(near(v["parts"]["last"]["output layer"], 2.400), "the output layer's part")
            check(near(v["pos"]["the head's product"], 2.400) and near(v["pos"]["after the head's product"], 0.790)
                  and near(v["pos"]["before the head's product"], lay + 0.040 + 0.015), "the last card by position")
            check(near(v["ops"]["last"][0]["TOP_K"], 0.700), "by first op")
            check(near(sum(v["per_card"].values()), sum(v["parts"]["all"].values())), "per card adds up")
            check(v["seq_evals"] == (12, 12), "one graph on the last card")
            bl, bf, pb = v["buckets"]["last"], v["buckets"]["first"], pl["buckets"]
            check(near(bl["the head's product"], 2.400) and near(bl["the sampling chain after the product"], 0.790),
                  "where: the head and the sampling chain")
            check(near(bl["GATED_DELTA_NET"] - pb["last"]["GATED_DELTA_NET"], 0.040)
                  and near(bf["GATED_DELTA_NET"] - pb["first"]["GATED_DELTA_NET"], 0.080), "where: GATED_DELTA_NET")
            check(near(sum(bl.values()), sum(v["parts"]["last"].values()))
                  and near(sum(bf.values()), sum(v["parts"]["first"].values())), "where: adds up to the parts")
            check(near(bl["everything else"] - pb["last"]["everything else"], 0.015), "where: the rest of the last card")
        check(near(pl["pos"]["after the head's product"], 0.0) and not pl["head"], "plain: no head")
        for name in ("parts.tsv", "by_op.tsv", "labels.tsv", "per_card.tsv", "last_card_groups.tsv", "passes.tsv",
                     "accounting.tsv",
                     "cut.tsv", "verify-r1.csv", "verify-r2.txt", "verify-r2-last.txt"):
            check(os.path.getsize(os.path.join(b.out, name)) > 0, f"{name} written")
        check(sum(1 for _ in open(os.path.join(b.out, "passes.tsv"))) == 1 + 39, "passes.tsv: every main pass")
    print(f"verify_parts self-test: {checks} checks, {len(fails)} failed")
    for x in fails:
        print("  FAILED:", x)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--kpass-parts", required=True); ap.add_argument("--prof-summary", required=True)
    ap.add_argument("--event-us", type=float, default=0.0); ap.add_argument("--plain", default="")
    ap.add_argument("--other-regex", default="mtp_"); ap.add_argument("--head-regex", default=" output:")
    ap.add_argument("--gap-ms", type=float, default=3000.0); ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.12); ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    kp, ps = load_mod(a.kpass_parts, "kpass_parts"), load_mod(a.prof_summary, "prof_summary")
    if a.self_test:
        sys.exit(0 if self_test(a, kp, ps) else 1)
    if not a.trace or not a.out:
        ap.error("the trace and --out are required")
    run(a, kp, ps)


if __name__ == "__main__":
    main()
