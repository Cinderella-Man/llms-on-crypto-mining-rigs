#!/usr/bin/env python3
"""depth_parts.py: a served speculative step at context depth by part, the verify pass and the MTP draft's passes
together, from a group-mode GGML_CUDA_PROFILE trace of a deep-lane load (bench/deep_lane with PAUSE, so that the trace
falls into one piece per request), beside a shallow trace of the same line (R51, Q1213).

  python3 -I depth_parts.py --verify-parts <2026-11-10_served_verify_parts/run/verify_parts.py> \
      --kpass-parts <2026-11-04_kpass_profile/run/kpass_parts.py> --event-us 7.6 \
      --deep <prof-depth.csv> --requests <requests.csv> --load prof [--ref-load warm] \
      [--shallow <prof-verify.csv> --shallow-label 'R49'] [--grow 16384:49152,32768:49152] \
      [--other-regex mtp_] [--gap-ms 3000] [--skip 3] [--tol 0.12] [--top 30] --out <dir> > report.md
  python3 -I depth_parts.py --verify-parts ... --kpass-parts ... --self-test

The trace has one line per launch group of every graph evaluation (device, eval, host_ms, group, label, us); host_ms is
taken once the evaluation's stream is synchronized, so under this profiler the cards run one after another.
  - Evaluations with a launch group whose label matches --other-regex are the MTP draft's own graphs; the others are
    the model's. The j-th model evaluation on every card is main pass j.
  - A pause of more than --gap-ms without an evaluation cuts the trace into pieces. In the deep trace the pieces after
    the load's own are the load's requests in --requests order (the client's warm-up, then per depth a prime and two
    decodes); the trace must hold one piece per request, else the script stops. In the shallow trace (verify_prof.sh's
    load: two requests of 256 tokens, R49) every piece of at least ten main passes is a decode, r1, r2, ...
  - The draft's evaluations after a main pass and before the next belong to that main pass. A step is a main pass with
    them; a draft pass is one evaluation on the draft's last card (the highest card the draft runs on).
  - In a decode, the steps are its main passes but the last; its first passes above 1.5 times the median GPU time are
    prompt micro-batches. A typical step has the commonest number of draft passes, no evaluation after its last draft
    pass, a main pass within --tol of the median of those, and is not among the first --skip steps.
  - The wall time of a step is from the end of the previous step's last evaluation to the end of its own last one;
    outside the kernels is that wall time less the step's raw GPU time (the host, the hops between cards, the uploads
    of the graph inputs, and this profiler's synchronization and writing).
Per decode, the mean over its typical steps, in ms a step, the profiler's --event-us subtracted per launch group:
  - by part: kpass_parts.py's (R41's) parts, for the main pass on every card but the last, the main pass on the last
    card, and the draft's passes;
  - where the growth goes (BUCKETS): the gather of every pooled indexer key, the indexer's scores, its top-k and mask,
    its other launch groups, the attention, the output heads, everything else;
  - by launch group (first op and tensor name, node numbers dropped), with launch groups a step;
  - per card; the totals and the wall time.
--ref-load names an unprofiled load in --requests with the same requests: the server's own ms a step of each decode
(predicted_ms / (predicted_n - draft_n_accepted)) is put beside the profiled one, and that step less the profiled GPU
time is the estimate of what lies outside the kernels without the profiler. --grow pairs depths: each column is the
deeper depth's request less the shallower's, pass 1 against pass 1 and pass 2 against pass 2.
Writes to --out: requests.tsv (the cut per request), parts.tsv, buckets.tsv, groups.tsv, per_card.tsv, totals.tsv,
steps.tsv (every pass of every decode); the Markdown report goes to stdout. Standard library only."""
import argparse, bisect, collections, csv, importlib.util, os, re, statistics, sys, tempfile


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_label(label):
    return re.sub(r"\bnode_\d+", "node", label)


def cut(evs, gap_ms):
    """-> info, [piece: [main pass]]; every main pass carries its draft evaluations ("dr")."""
    devs = sorted({e["d"] for e in evs if not e["other"]})
    model = {d: sorted((e for e in evs if e["d"] == d and not e["other"]), key=lambda e: e["t"]) for d in devs}
    others = sorted((e for e in evs if e["other"]), key=lambda e: e["t"])
    n = min(len(v) for v in model.values())
    draft_last = max((e["d"] for e in others), default=devs[-1])
    ts = sorted(e["t"] for e in evs)
    cuts = [b for a, b in zip(ts, ts[1:]) if b - a > gap_ms]
    piece = lambda t: bisect.bisect_right(cuts, t)
    info = {"devs": devs, "counts": {d: len(v) for d, v in model.items()}, "others": len(others),
            "draft_last": draft_last, "other_by_card": dict(sorted(collections.Counter(e["d"] for e in others).items())),
            "disorder": 0, "unassigned": 0}
    passes = []
    for j in range(n):
        es = [model[d][j] for d in devs]
        if any(b["t"] < a["t"] for a, b in zip(es, es[1:])) or (passes and es[0]["t"] < passes[-1]["t1"]):
            info["disorder"] += 1
        passes.append({"j": j, "evs": es, "t0": es[0]["t"], "t1": es[-1]["t"], "us": sum(e["us"] for e in es),
                       "piece": piece(es[0]["t"]), "dr": [], "n_dp": 0, "stray": 0, "request": "", "kind": "",
                       "i": -1, "typical": 0, "wall": float("nan")})
    t1s = [p["t1"] for p in passes]
    for o in others:
        i = bisect.bisect_right(t1s, o["t"]) - 1
        if i < 0 or passes[i]["piece"] != piece(o["t"]) or (i + 1 < n and o["t"] > passes[i + 1]["t0"]):
            info["unassigned"] += 1  # a draft evaluation before the piece's first main pass (a load's warm-up)
            continue
        passes[i]["dr"].append(o)
    for p in passes:
        last = max((k for k, o in enumerate(p["dr"]) if o["d"] == draft_last), default=-1)
        p["n_dp"] = sum(1 for o in p["dr"] if o["d"] == draft_last)
        p["stray"] = len(p["dr"]) - 1 - last
        p["end"] = max([p["t1"]] + [o["t"] for o in p["dr"]])
        p["gpu_raw"] = p["us"] + sum(o["us"] for o in p["dr"])
    pieces = collections.OrderedDict()
    for p in passes:
        pieces.setdefault(p["piece"], []).append(p)
    info["pieces"] = [len(v) for v in pieces.values()]
    return info, list(pieces.values())


def decode_steps(ps, name, skip, tol):
    """Marks a decode's passes (prompt, step, last) and its typical steps; -> the request's cut."""
    steps = ps[:-1]
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
    for k, p in enumerate(ps[1:], 1):
        p["wall"] = p["end"] - ps[k - 1]["end"]
    for i, p in enumerate(gen):
        p["i"], p["kind"] = i, "step"
        p["typical"] = int(i >= skip and p["n_dp"] == mode and p["stray"] == 0 and abs(p["us"] / mmed - 1.0) <= tol)
    typ = [p for p in gen if p["typical"]]
    if not typ:
        sys.exit(f"{name}: no typical step among {len(gen)}")
    return {"name": name, "passes": len(ps), "prompt": g0, "steps": len(gen), "mode": mode, "typical": len(typ),
            "by_dp": dict(sorted(collections.Counter(p["n_dp"] for p in gen).items())), "kept": typ, "all": ps}


# Where the step's time goes at depth, for the table "where the growth goes": the first rule that matches a launch group
# wins. A rule sees the label and whether the group is the gather of every pooled key, the GET_ROWS that feeds a
# LIGHTNING_INDEXER (the nearest GET_ROWS among the two groups before it in the same evaluation: qwen4exp's
# build_qsa_sel gathers all n_pool pooled indexer keys by cell for each QSA layer and pass).
BUCKETS = [
    ("the gather of every pooled key (GET_ROWS before the indexer)", lambda lab, gather: gather),
    ("the indexer's scores (LIGHTNING_INDEXER)", lambda lab, gather: lab.startswith("LIGHTNING_INDEXER")),
    ("the indexer's top-k and mask (TOP_K, TOP_K_MASK)", lambda lab, gather: "indexer_top_k" in lab or "indexer_sel" in lab),
    ("the indexer's other launch groups (projections, norms, RoPE)", lambda lab, gather: "indexer" in lab),
    ("attention (FLASH_ATTN_EXT: the list of kept cells and the kernel)", lambda lab, gather: lab.startswith("FLASH_ATTN_EXT")),
    ("the output heads (` output:`)", lambda lab, gather: " output:" in lab),
    ("everything else", lambda lab, gather: True),
]


def gathers(rows):
    """Indices of the rows that are the gather of every pooled key: the GET_ROWS just before a LIGHTNING_INDEXER."""
    out = set()
    for k, r in enumerate(rows):
        if r["label"].startswith("LIGHTNING_INDEXER"):
            for i in (k - 1, k - 2):
                if i >= 0 and rows[i]["label"].startswith("GET_ROWS "):
                    out.add(i)
                    break
    return out


def measure(q, devs, kp, event_us):
    """ms a step over the request's typical steps: by part, by bucket, by launch group, per card, the totals."""
    groups = {"main, cards %d to %d" % (devs[0], devs[-2]) if len(devs) > 2 else "main, card %d" % devs[0]: devs[:-1],
              "main, card %d" % devs[-1]: devs[-1:]}
    gnames = list(groups) + ["draft"]
    kept = q["kept"]
    n = len(kept)
    parts = {g: collections.Counter() for g in gnames}
    labels = {g: collections.Counter() for g in gnames}
    nlab = {g: collections.Counter() for g in gnames}
    buckets = {g: collections.Counter() for g in gnames}
    card = collections.Counter()
    draft_card = collections.Counter()
    for p in kept:
        items = [(g, e) for g, ds in groups.items() for e in p["evs"] if e["d"] in ds] + [("draft", o) for o in p["dr"]]
        for g, e in items:
            gat = gathers(e["rows"])
            for k, r in enumerate(e["rows"]):
                us = float(r["us"]) - event_us
                parts[g][kp.part_of(r["label"])] += us / n / 1000
                buckets[g][next(b for b, f in BUCKETS if f(r["label"], k in gat))] += us / n / 1000
                lab = norm_label(r["label"])
                labels[g][lab] += us / n / 1000
                nlab[g][lab] += 1 / n
                if g == "draft":
                    draft_card[e["d"]] += us / n / 1000
                else:
                    card[e["d"]] += us / n / 1000
    tot = collections.OrderedDict()
    for g in gnames:
        tot[g] = sum(parts[g].values())
    tot["GPU time of the step"] = sum(tot[g] for g in gnames)
    tot["draft passes a step"] = q["mode"]
    tot["wall time a step under the profiler"] = statistics.fmean(p["wall"] for p in kept)
    tot["outside the kernels"] = statistics.fmean(p["wall"] - p["gpu_raw"] / 1000 for p in kept)
    tot["the profiler's subtracted cost"] = statistics.fmean(
        (sum(len(e["rows"]) for e in p["evs"]) + sum(len(o["rows"]) for o in p["dr"])) * event_us / 1000 for p in kept)
    return {"gnames": gnames, "parts": parts, "buckets": buckets, "labels": labels, "nlab": nlab, "card": card,
            "draft_card": draft_card, "tot": tot}


def read_requests(path, load):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["load"] == load]


def server_step(r):
    try:
        steps = int(r["gen_n"]) - int(r["draft_accepted"])
        return float(r["gen_ms"]) / steps if steps > 0 else float("nan")
    except (KeyError, ValueError):
        return float("nan")


def table(cols, rows, file):
    print("| " + " | ".join(cols) + " |", file=file)
    print("|---" + "|---:" * (len(cols) - 1) + "|", file=file)
    for r in rows:
        print("| " + " | ".join(r) + " |", file=file)
    print(file=file)


def f2(x):
    return "" if x != x else f"{x:.2f}"


def run(a, vp, kp, file=sys.stdout):
    os.makedirs(a.out, exist_ok=True)
    other_rx = re.compile(a.other_regex)
    P = lambda *x: print(*x, file=file)
    reqs = []   # every decode measured: name, trace, depth, pass, cut, measure, server steps
    cuts = []   # (trace, request name, kind, depth, pass, piece size)
    infos = {}
    traces = ([("shallow", a.shallow)] if a.shallow else []) + ([("deep", a.deep)] if a.deep else [])
    devs = None
    for tname, path in traces:
        _, evs = vp.read_evals(path, other_rx)
        info, pieces = cut(evs, a.gap_ms)
        infos[tname] = info
        if devs is None:
            devs = info["devs"]
        elif info["devs"] != devs:
            sys.exit(f"the traces hold different cards: {devs} and {info['devs']}")
        if tname == "shallow":
            k = 0
            for ps in pieces:
                if len(ps) < 10:
                    cuts.append((tname, "", "load", "", "", len(ps)))
                    continue
                k += 1
                q = decode_steps(ps, f"{a.shallow_label} r{k}", a.skip, a.tol)
                cuts.append((tname, q["name"], "decode", a.shallow_depth, str(k), len(ps)))
                reqs.append({"name": q["name"], "trace": tname, "depth": a.shallow_depth, "pass": str(k), "q": q,
                             "m": measure(q, devs, kp, a.event_us), "srv": float("nan"), "ref": float("nan")})
            continue
        rows = read_requests(a.requests, a.load)
        ref = {(r["kind"], r["depth"], r["pass"]): r for r in read_requests(a.requests, a.ref_load)} if a.ref_load else {}
        extra = len(pieces) - len(rows)
        if extra > 0 and all(len(ps) <= 3 for ps in pieces[:extra]):  # the load's own evaluations
            for ps in pieces[:extra]:
                cuts.append((tname, "", "load", "", "", len(ps)))
            pieces = pieces[extra:]
        if len(pieces) != len(rows):
            sys.exit(f"{path}: {len(pieces)} pieces ({', '.join(map(str, infos[tname]['pieces']))} main passes) for "
                     f"{len(rows)} requests of load {a.load}: the trace does not cut into one piece per request")
        for ps, r in zip(pieces, rows):
            name = f"{r['depth']} p{r['pass']}" if r["kind"] == "decode" else f"{r['kind']} {r['depth']}"
            cuts.append((tname, name, r["kind"], r["depth"], r["pass"], len(ps)))
            for p in ps:
                p["request"], p["kind"] = name, r["kind"]
            if r["kind"] != "decode":
                continue
            q = decode_steps(ps, name, a.skip, a.tol)
            rr = ref.get(("decode", r["depth"], r["pass"]))
            reqs.append({"name": name, "trace": tname, "depth": r["depth"], "pass": r["pass"], "q": q,
                         "m": measure(q, devs, kp, a.event_us), "srv": server_step(r),
                         "ref": server_step(rr) if rr else float("nan"), "row": r, "ref_row": rr})
    by = {(q["trace"], q["depth"], q["pass"]): q for q in reqs if q["trace"] == "deep"}
    grows = []
    for pair in [x for x in a.grow.split(",") if x]:
        lo, hi = pair.split(":")
        for pas in ("1", "2"):
            if ("deep", lo, pas) in by and ("deep", hi, pas) in by:
                grows.append((f"{hi} − {lo}, p{pas}", by[("deep", lo, pas)], by[("deep", hi, pas)]))
    gnames = reqs[0]["m"]["gnames"]

    # the cut
    for tname, info in infos.items():
        P(f"**{tname} trace:** model evaluations per card " + ", ".join(f"{d}: {c}" for d, c in info["counts"].items()) +
          f"; the draft's {info['others']} (" + ", ".join(f"card {d}: {c}" for d, c in info["other_by_card"].items()) +
          f"), its last card {info['draft_last']}; main passes per piece (cut at pauses over {a.gap_ms:g} ms) " +
          ", ".join(map(str, info["pieces"])) + f"; passes out of order {info['disorder']}; draft evaluations outside "
          f"a step {info['unassigned']}")
        P()
    table(["Trace", "Request", "Kind", "Depth", "Pass", "Main passes"], [[c[0], c[1], c[2], c[3], c[4], str(c[5])] for c in cuts], file)
    cols = [q["name"] for q in reqs]
    table(["Decode"] + ["prompt passes", "steps", "draft passes a step (steps)", "typical steps",
                        "server's ms a step, profiled", "server's ms a step, unprofiled load"],
          [[q["name"], str(q["q"]["prompt"]), str(q["q"]["steps"]),
            ", ".join(f"{k}: {v}" for k, v in q["q"]["by_dp"].items()), str(q["q"]["typical"]), f2(q["srv"]), f2(q["ref"])]
           for q in reqs], file)
    gcols = [g[0] for g in grows]

    def rowvals(get):
        return [f2(get(q)) for q in reqs] + [f"{get(hi) - get(lo):+.2f}" for _, lo, hi in grows]

    P("**The step, ms** (GPU time with the profiler's cost subtracted; the wall time and what lies outside the kernels "
      "are under the profiler)")
    P()
    keys = list(reqs[0]["m"]["tot"])
    gpu = lambda q: q["m"]["tot"]["GPU time of the step"]
    table(["Step"] + cols + gcols, [[k] + rowvals(lambda q, k=k: q["m"]["tot"][k]) for k in keys] +
          [["server's ms a step, profiled"] + rowvals(lambda q: q["srv"]),
           ["server's ms a step, unprofiled load"] + rowvals(lambda q: q["ref"]),
           ["outside the kernels, unprofiled (est.: the unprofiled step less the GPU time)"] +
           rowvals(lambda q: q["ref"] - gpu(q))], file)
    P("**Where the growth goes, ms a step** (BUCKETS: the first rule that matches a launch group wins; each bucket on "
      "the main pass's cards and the draft's, then its sum)")
    P()
    rows = []
    for b, _ in BUCKETS:
        for g in gnames:
            if any(abs(q["m"]["buckets"][g][b]) > 0.004 for q in reqs):
                rows.append([f"{b}: {g}"] + rowvals(lambda q, b=b, g=g: q["m"]["buckets"][g][b]))
        rows.append([f"**{b}**"] + rowvals(lambda q, b=b: sum(q["m"]["buckets"][g][b] for g in gnames)))
    rows.append(["**GPU time of the step**"] + rowvals(gpu))
    rows.append(["**outside the kernels, unprofiled (est.)**"] + rowvals(lambda q: q["ref"] - gpu(q)))
    rows.append(["**the server's step, unprofiled load**"] + rowvals(lambda q: q["ref"]))
    table(["Where"] + cols + gcols, rows, file)
    for g in gnames:
        names = [p for p in kp.NAMES if any(abs(q["m"]["parts"][g][p]) > 0.004 for q in reqs)]
        P(f"**By part, ms a step: {g}**")
        P()
        table(["Part"] + cols + gcols, [[p] + rowvals(lambda q, p=p, g=g: q["m"]["parts"][g][p]) for p in names] +
              [["**sum**"] + rowvals(lambda q, g=g: sum(q["m"]["parts"][g].values()))], file)
    # the launch groups that grow most
    lab_keys = {(g, lab) for q in reqs for g in gnames for lab in q["m"]["labels"][g]}
    val = lambda q, g, lab: q["m"]["labels"][g].get(lab, 0.0)
    if grows:
        _, lo, hi = grows[-1] if len(grows) < 2 else grows[1]  # the first pair's pass 2
        order = sorted(lab_keys, key=lambda k: -(val(hi, *k) - val(lo, *k)))
    else:
        order = sorted(lab_keys, key=lambda k: -val(reqs[-1], *k))
    P(f"**The launch groups that grow most, ms a step (launch groups a step in the last decode)**, sorted by "
      f"{(grows[1] if len(grows) > 1 else grows[-1])[0] if grows else 'the last decode'}")
    P()
    top = order[:a.top]
    rows = [[f"{g}: `{lab[:90]}` ({reqs[-1]['m']['nlab'][g].get(lab, 0):.0f})"] + rowvals(lambda q, g=g, lab=lab: val(q, g, lab))
            for g, lab in top]
    rest = order[a.top:]
    rows.append([f"the other {len(rest)} launch groups"] + rowvals(lambda q: sum(val(q, g, lab) for g, lab in rest)))
    table(["Launch group"] + cols + gcols, rows, file)
    P("**Per card, ms a step**")
    P()
    table(["Card"] + cols + gcols, [[f"main, card {d}"] + rowvals(lambda q, d=d: q["m"]["card"][d]) for d in devs] +
          [[f"draft, card {d}"] + rowvals(lambda q, d=d: q["m"]["draft_card"][d])
           for d in sorted({d for q in reqs for d in q["m"]["draft_card"]})], file)

    w = lambda name: open(os.path.join(a.out, name), "w")
    with w("requests.tsv") as f:
        f.write("trace\trequest\tkind\tdepth\tpass\tmain_passes\tprompt_passes\tsteps\tdraft_passes_a_step\t"
                "typical_steps\tserver_ms_step_profiled\tserver_ms_step_unprofiled\n")
        dq = {(q["trace"], q["name"]): q for q in reqs}
        for c in cuts:
            q = dq.get((c[0], c[1]))
            f.write("\t".join(map(str, c)) + "\t" + ("\t".join([str(q["q"]["prompt"]), str(q["q"]["steps"]),
                    ",".join(f"{k}:{v}" for k, v in q["q"]["by_dp"].items()), str(q["q"]["typical"]), f2(q["srv"]),
                    f2(q["ref"])]) if q else "\t\t\t\t\t") + "\n")
    with w("totals.tsv") as f:
        f.write("trace\trequest\tdepth\tpass\ttypical_steps\twhat\tms_a_step\n")
        for q in reqs:
            for k, v in list(q["m"]["tot"].items()) + [("server's ms a step, profiled", q["srv"]),
                                                      ("server's ms a step, unprofiled load", q["ref"])]:
                f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{q['q']['typical']}\t{k}\t{f2(v) if k != 'draft passes a step' else v}\n")
    with w("parts.tsv") as f:
        f.write("trace\trequest\tdepth\tpass\tgroup\tpart\tms_a_step\n")
        for q in reqs:
            for g in gnames:
                for p in kp.NAMES:
                    f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{g}\t{p}\t{q['m']['parts'][g][p]:.3f}\n")
    with w("groups.tsv") as f:
        f.write("trace\trequest\tdepth\tpass\tgroup\tlabel\tgroups_a_step\tms_a_step\n")
        for q in reqs:
            for g in gnames:
                for lab, v in sorted(q["m"]["labels"][g].items(), key=lambda kv: -kv[1]):
                    f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{g}\t{lab}\t"
                            f"{q['m']['nlab'][g][lab]:.2f}\t{v:.4f}\n")
    with w("buckets.tsv") as f:
        f.write("trace\trequest\tdepth\tpass\tgroup\twhere\tms_a_step\n")
        for q in reqs:
            for g in gnames:
                for b, _ in BUCKETS:
                    f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{g}\t{b}\t{q['m']['buckets'][g][b]:.3f}\n")
            f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t\toutside the kernels, unprofiled (est.)\t"
                    f"{f2(q['ref'] - gpu(q))}\n")
    with w("per_card.tsv") as f:
        f.write("trace\trequest\tdepth\tpass\tcard\twhat\tms_a_step\n")
        for q in reqs:
            for d, v in sorted(q["m"]["card"].items()):
                f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{d}\tmain pass\t{v:.3f}\n")
            for d, v in sorted(q["m"]["draft_card"].items()):
                f.write(f"{q['trace']}\t{q['name']}\t{q['depth']}\t{q['pass']}\t{d}\tdraft passes\t{v:.3f}\n")
    # every pass of every decode, typical or not
    with w("steps.tsv") as f:
        f.write("trace\trequest\tpass_in_trace\tstep\tkind\ttypical\tdraft_passes\tstray\tmain_raw_ms\tdraft_raw_ms\t"
                "wall_ms\n")
        for q in reqs:
            for p in q["q"]["all"]:
                f.write("%s\t%s\t%d\t%d\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%s\n" % (
                    q["trace"], q["name"], p["j"], p["i"], p["kind"], p["typical"], p["n_dp"], p["stray"],
                    p["us"] / 1000, sum(o["us"] for o in p["dr"]) / 1000, f2(p["wall"])))
    return infos, reqs, grows


def self_test(a, vp, kp):
    """A synthetic deep trace of three cards (a load, the client's warm-up, a prime and two decodes at each of two
    depths) and a shallow one (a load, two decodes), with parts that grow with depth by known amounts."""
    C = 8.0
    import types
    rows, n_eval, t = [], collections.Counter(), [0.0]

    def ev(d, groups):
        t[0] += sum(us for _, us in groups) / 1000 + 0.3
        for k, (lab, us) in enumerate(groups):
            rows.append([d, n_eval[d], f"{t[0]:.3f}", k, lab, f"{us + C:.2f}"])
        n_eval[d] += 1

    def layer(depth):
        return [("RMS_NORM+MUL node_7", 20.0), ("MUL_MAT attn_q:q4_0:2560x4096", 300.0),
                ("GET_ROWS node_8", 15.0 + depth / 256.0),
                ("LIGHTNING_INDEXER indexer_score", 10.0 + depth / 1024.0), ("FLASH_ATTN_EXT node_9", 40.0),
                ("MUL_MAT_ID ffn_up_exps:q2_0:2560x512", 200.0)]

    def main_pass(depth, scale=1.0, last_extra=()):
        for d in range(3):
            g = [(lab, us * scale) for lab, us in layer(depth)]
            ev(d, g + (list(last_extra) if d == 2 else []))

    def draft(n, depth):
        for _ in range(n):
            ev(1, [("GET_ROWS mtp_tok_embd", 30.0)])
            ev(2, [("RMS_NORM+MUL mtp_enorm", 10.0), ("FLASH_ATTN_EXT node_3", 5.0 + depth / 512.0),
                   ("MUL_MAT mtp_head:q8_0:2560x8192", 900.0)])

    head = [("MUL_MAT output:q5_k:2560x248320", 2000.0)]

    def decode(depth, n_steps=12):
        main_pass(depth, 4.0)                        # the prompt
        for i in range(n_steps):
            main_pass(depth, 1.6 if i == 6 else 1.0, head)
            draft(3 if i == 8 else 4, depth)
        main_pass(depth, 1.0, head)                  # the last pass, no step

    def gap():
        t[0] += 5000.0

    def write(path):
        with open(path, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["device", "eval", "host_ms", "group", "label", "us"])
            wr.writerows(rows)
            f.write("2,999,1.0,3,MUL_MAT cut")       # a line cut at the server's exit
        rows.clear(); n_eval.clear(); t[0] = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        main_pass(0, 3.0); draft(1, 0)                # the load
        gap(); main_pass(0, 2.0); [main_pass(0) or draft(4, 0) for _ in range(3)]  # the client's warm-up
        req = [("warmup", 0, 0)]
        for depth in (16384, 49152):
            gap()
            for _ in range(5):                        # the prime: prompt passes with the draft's
                main_pass(depth, 20.0); draft(4, depth)
            main_pass(depth, 1.0, head)
            req.append(("prime", depth, 0))
            for pas in (1, 2):
                gap(); decode(depth)
                req.append(("decode", depth, pas))
        deep = os.path.join(tmp, "deep.csv"); write(deep)
        main_pass(0, 3.0)
        for _ in range(2):
            gap(); decode(256)
        shallow = os.path.join(tmp, "shallow.csv"); write(shallow)
        reqcsv = os.path.join(tmp, "requests.csv")
        with open(reqcsv, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["arm", "load", "kind", "pass", "depth", "gen_n", "gen_ms", "draft_accepted"])
            for load, ms in (("warm", 100.0), ("prof", 130.0)):
                for kind, depth, pas in req:
                    wr.writerow(["x", load, kind, pas, depth, 40, 10 * (ms + depth / 1000), 30])
        b = types.SimpleNamespace(deep=deep, shallow=shallow, shallow_label="S", shallow_depth="256",
                                  requests=reqcsv, load="prof", ref_load="warm", grow="16384:49152",
                                  other_regex="mtp_", gap_ms=3000.0, skip=3, tol=0.12, event_us=C, top=10,
                                  out=os.path.join(tmp, "out"))
        with open(os.path.join(tmp, "report.md"), "w") as f:
            infos, reqs, grows = run(b, vp, kp, f)
        checks, fails = 0, []

        def check(cond, what):
            nonlocal checks
            checks += 1
            if not cond:
                fails.append(what)

        near = lambda x, y: abs(x - y) < 1e-6
        check(infos["deep"]["pieces"] == [1, 4, 6, 14, 14, 6, 14, 14], f"deep pieces {infos['deep']['pieces']}")
        check(infos["shallow"]["pieces"] == [1, 14, 14], f"shallow pieces {infos['shallow']['pieces']}")
        check(infos["deep"]["unassigned"] == 0 and infos["deep"]["disorder"] == 0, "deep: every draft evaluation in a step")
        check([q["name"] for q in reqs] == ["S r1", "S r2", "16384 p1", "16384 p2", "49152 p1", "49152 p2"],
              f"decodes {[q['name'] for q in reqs]}")
        for q in reqs:
            # 12 steps: the prompt pass, then 12 steps; three skipped, one with three draft passes, one slow
            check(q["q"]["prompt"] == 1 and q["q"]["steps"] == 12, f"{q['name']}: {q['q']['prompt']} prompt, {q['q']['steps']} steps")
            check(q["q"]["typical"] == 7 and q["q"]["mode"] == 4, f"{q['name']}: {q['q']['typical']} typical")
            m = q["m"]
            depth = int(q["depth"])
            main = 3 * sum(us for _, us in layer(depth)) / 1000 + 2.0
            dr = 4 * (30.0 + 10.0 + 5.0 + depth / 512.0 + 900.0) / 1000
            check(near(m["tot"]["GPU time of the step"], main + dr), f"{q['name']}: GPU time of the step")
            check(near(m["tot"]["draft"], dr), f"{q['name']}: the draft's passes")
            check(near(m["parts"]["main, card 2"]["output layer"], 2.0), f"{q['name']}: the head on the last card")
            n_rows = 3 * 6 + 1 + 4 * 4
            bk = m["buckets"]
            check(near(sum(sum(bk[g].values()) for g in bk), m["tot"]["GPU time of the step"]),
                  f"{q['name']}: the buckets add up")
            check(near(bk["main, cards 0 to 1"][BUCKETS[0][0]], 2 * (15.0 + depth / 256.0) / 1000)
                  and near(bk["main, card 2"][BUCKETS[0][0]], (15.0 + depth / 256.0) / 1000),
                  f"{q['name']}: the gather before the indexer")
            check(near(bk["main, cards 0 to 1"][BUCKETS[1][0]], 2 * (10.0 + depth / 1024.0) / 1000),
                  f"{q['name']}: the indexer's scores")
            check(near(bk["draft"][BUCKETS[4][0]], 4 * (5.0 + depth / 512.0) / 1000)
                  and near(bk["main, card 2"][BUCKETS[5][0]], 2.0), f"{q['name']}: the draft's attention, the head")
            # wall: each evaluation adds its GPU time and 0.3 ms
            # the wall: each evaluation adds its GPU time without the profiler's cost and 0.3 ms
            check(abs(m["tot"]["outside the kernels"] - ((3 + 8) * 0.3 - n_rows * C / 1000)) < 2e-3, f"{q['name']}: outside")
            check(near(m["tot"]["the profiler's subtracted cost"], n_rows * C / 1000), f"{q['name']}: profiler's cost")
            check(near(sum(m["card"].values()) + sum(m["draft_card"].values()), m["tot"]["GPU time of the step"]),
                  f"{q['name']}: per card adds up")
        lo, hi = reqs[3], reqs[5]
        check(near(hi["m"]["labels"]["main, cards 0 to 1"]["LIGHTNING_INDEXER indexer_score"] -
                   lo["m"]["labels"]["main, cards 0 to 1"]["LIGHTNING_INDEXER indexer_score"], 2 * 32.0 / 1000),
              "the indexer's growth on cards 0 to 1")
        check(near(hi["m"]["labels"]["draft"]["FLASH_ATTN_EXT node"] - lo["m"]["labels"]["draft"]["FLASH_ATTN_EXT node"],
                   4 * 64.0 / 1000), "the draft's attention growth")
        check(near(hi["m"]["tot"]["GPU time of the step"] - lo["m"]["tot"]["GPU time of the step"],
                   3 * 32.0 / 1000 + 4 * 64.0 / 1000 + 3 * 128.0 / 1000), "the step's growth is the three")
        check(near(lo["srv"], 10 * (130.0 + 16.384) / 10) and near(lo["ref"], 10 * (100.0 + 16.384) / 10), "server steps")
        check(len(grows) == 2 and grows[0][0] == "49152 − 16384, p1", f"grows {[g[0] for g in grows]}")
        for name in ("requests.tsv", "parts.tsv", "buckets.tsv", "groups.tsv", "per_card.tsv", "totals.tsv", "steps.tsv"):
            check(os.path.getsize(os.path.join(b.out, name)) > 0, f"{name} written")
        check(sum(1 for _ in open(os.path.join(b.out, "requests.tsv"))) == 1 + 1 + 7 + 1 + 2, "requests.tsv: every piece")
    print(f"depth_parts self-test: {checks} checks, {len(fails)} failed")
    for x in fails:
        print("  FAILED:", x)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-parts", required=True); ap.add_argument("--kpass-parts", required=True)
    ap.add_argument("--deep"); ap.add_argument("--requests"); ap.add_argument("--load", default="prof")
    ap.add_argument("--ref-load", default="")
    ap.add_argument("--shallow", default=""); ap.add_argument("--shallow-label", default="shallow")
    ap.add_argument("--shallow-depth", default="")
    ap.add_argument("--grow", default="16384:49152,32768:49152")
    ap.add_argument("--event-us", type=float, default=0.0); ap.add_argument("--other-regex", default="mtp_")
    ap.add_argument("--gap-ms", type=float, default=3000.0); ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.12); ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=""); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    vp, kp = load_mod(a.verify_parts, "verify_parts"), load_mod(a.kpass_parts, "kpass_parts")
    if a.self_test:
        sys.exit(0 if self_test(a, vp, kp) else 1)
    if not a.out or not ((a.deep and a.requests) or a.shallow):
        ap.error("--out and --deep with --requests (or --shallow alone) are required")
    run(a, vp, kp)


if __name__ == "__main__":
    main()
