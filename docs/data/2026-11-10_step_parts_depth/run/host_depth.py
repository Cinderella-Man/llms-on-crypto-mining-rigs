#!/usr/bin/env python3
"""host_depth.py: what a served speculative step spends outside the kernels at two context depths, from the fork's host
profile (GGML_CUDA_PROFILE_HOST) and the QSA k-pool inputs' own counts (LLAMA_QSA_INPUT_STATS=1) of one deep-lane load
(bench/deep_lane with PAUSE, so that the profile falls into one piece per request; R64, Q1243).

  python3 -I host_depth.py --step-split <2026-11-10_served_step_split/run/step_split.py> \
      --host <host.csv> --log <server-<load>.log> --requests <requests.csv> --load host [--ref-load warm] \
      [--grow 16384:49152] [--gap-ms 2000] [--skip 3] [--tol 0.12] [--main-qsa-devices 9] [--draft-qsa-devices 1] \
      [--out <dir>] > host_depth.md
  python3 -I host_depth.py --step-split ... --self-test

The host profile has one line per graph evaluation and card (step_split.py reads it): the host's time at entry to and
exit from the backend's graph_compute and the GPU time between two events around the evaluation's work. The scheduler
copies a card's host inputs (the k-pool inputs among them) before it enters that card's graph_compute, so an upload
lies between the previous evaluation's exit and the card's entry.
  - A gap of more than --gap-ms without an evaluation cuts the profile into pieces: the load's own piece, then one per
    request of the load in requests.csv order (the client's warm-up, then per depth a prime and two decodes); the
    counts must agree, else the script stops. A prompt micro-batch at depth leaves the host idle for up to ~2.5 s, so
    --gap-ms sits between that and the deep lane's PAUSE. The first decode at a depth saves a context checkpoint
    after the tail's pass and can leave a longer gap there: a piece after the load's with fewer than --join-below main
    passes is joined to the next piece.
  - A main pass is one evaluation on each of the nine cards in order. The evaluations after it are the MTP draft's
    passes: each ends with an evaluation on the draft's last card (card 8), after one on its embedding card (card 4).
  - A step is a main pass and the draft passes after it. Its parts follow step_split.py (S37): verify_gpu (the nine
    cards' GPU time), verify_rest (the rest of the span to the first draft evaluation: the crossings, the rows back,
    the acceptance, the server before the draft), per draft pass lead (from its first entry to the last card's entry),
    gpu and turn (from the end of that work to the next pass's entry; the last turn is the server's work before the
    next verify pass, with the host's build of its inputs and the upload to card 0). Added here: per card of the main
    pass the host's gap before its entry (from the previous card's exit: that card's uploads) and its span; per draft
    pass the crossing (from the embedding card's exit to the last card's entry: the row's way through the host and
    the last card's uploads); and outside = the step less the GPU time of every evaluation in it.
  - Typical steps as step_split.py: after the request's prompt passes (a main pass above 1.5 times the median GPU
    time), the commonest number of draft passes, no stray evaluation, verify_gpu within --tol of their median, not
    among the first --skip steps.
The server log holds one line per evaluated micro-batch of either model with LLAMA_QSA_INPUT_STATS=1: the host time of
the k-pool inputs' set_input (us), their bytes per card (KiB), n_pool and the micro-batch's tokens. The log is written
in call order, so the lines between a request's launch_slot_ and release lines are that request's, and within it the
j-th line is the j-th pass (main or draft) of its piece of the profile; the counts must agree per request, else that
request's lines are left unmatched and the report says so.
Per decode request, the mean over its typical steps (and the median), and per --grow pair of depths each part's growth,
pass 1 against pass 1 and pass 2 against pass 2. --ref-load names an unprofiled load in requests.csv with the same
requests: the server's own ms a step (gen_ms / (gen_n - draft_accepted)) goes beside the profiled load's.
Writes to --out: pieces.tsv (the cut), steps.tsv (every generation step: parts, per card, the step's k-pool counts),
qsa.tsv (every k-pool line: request, pass kind, n_pool, tokens, us, KiB), summary.tsv; the Markdown report goes to
stdout. Standard library only."""
import argparse, collections, csv, importlib.util, os, re, statistics, sys, tempfile


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QSA_RE = re.compile(r"QSA k-pool inputs, last (\d+) ubatches: ([\d.]+) us on the host and ([\d.]+) KiB per device per "
                    r"ubatch \(n_pool (\d+), (\d+) tokens\)")
LAUNCH_RE = re.compile(r"launch_slot_: id +\d+ \| task (\d+) \|")
RELEASE_RE = re.compile(r"release: id +\d+ \| task (\d+) \|")


def read_log(path):
    """-> [(task or None, [qsa line dict])]: the load's lines first (task None), then one entry per request."""
    out = [(None, [])]
    cur = out[0][1]
    with open(path, errors="replace") as f:
        for line in f:
            m = LAUNCH_RE.search(line)
            if m:
                out.append((int(m.group(1)), []))
                cur = out[-1][1]
                continue
            m = RELEASE_RE.search(line)
            if m:
                cur = []  # lines after a release and before the next launch belong to no request
                continue
            m = QSA_RE.search(line)
            if m:
                cur.append(dict(every=int(m.group(1)), us=float(m.group(2)), kib=float(m.group(3)),
                                n_pool=int(m.group(4)), n_tokens=int(m.group(5))))
    return out


def read_requests(path, load):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["load"] == load]


def server_step(r):
    try:
        steps = int(r["gen_n"]) - int(r["draft_accepted"])
        return float(r["gen_ms"]) / steps if steps > 0 else float("nan")
    except (KeyError, ValueError):
        return float("nan")


def req_name(r):
    if r["kind"] == "decode":
        return f"{r['kind']} {r['depth']} p{r['pass']}"
    return f"{r['kind']} {r['depth']}"


def pass_seq(seg, main_devs, draft_last, ss):
    """-> [(kind, [rows])] in time order: 'main' (step_split.main_passes: one evaluation on each main card in order),
    'draft' (the evaluations between main passes, each pass up to one on draft_last) or 'stray' (evaluations between
    main passes after the last draft pass)."""
    out = []

    def drafts(lo, hi):
        cur = []
        for r in seg[lo:hi]:
            cur.append(r)
            if r[2] == draft_last:
                out.append(("draft", cur))
                cur = []
        if cur:
            out.append(("stray", cur))

    prev = 0
    for s, e in ss.main_passes(seg, main_devs):
        drafts(prev, s)
        out.append(("main", seg[s:e + 1]))
        prev = e + 1
    drafts(prev, len(seg))
    return out


def steps(seg, main_devs, draft_last, skip, tol, ss):
    """-> generation steps of a decode piece: step_split.steps_of's parts plus per card and per crossing; and the
    passes' order (index into pass_seq) of each step's main pass and draft passes."""
    passes = ss.main_passes(seg, main_devs)
    st = ss.steps_of(seg, passes, draft_last)
    for j, (x, p, q) in enumerate(zip(st, passes, passes[1:])):
        main = seg[p[0]:p[1] + 1]
        x["gap"] = [None] + [main[d][0] - main[d - 1][1] for d in range(1, len(main))]
        x["span"] = [r[1] - r[0] for r in main]
        between = seg[p[1] + 1:q[0]]
        dps, cur = [], []
        for b in between:
            cur.append(b)
            if b[2] == draft_last:
                dps.append(cur)
                cur = []
        for k, dp in enumerate(dps):
            x["passes"][k]["cross"] = dp[-1][0] - dp[-2][1] if len(dp) > 1 else 0.0
            x["passes"][k]["emb_span"] = sum(r[1] - r[0] for r in dp[:-1])
        x["gpu_all"] = x["verify_gpu_ms"] + sum(d["gpu"] + d["emb_gpu"] for d in x["passes"])
        x["outside"] = x["step_ms"] - x["gpu_all"]
        x["main_index"] = j
    if not st:
        return []
    med = statistics.median(x["verify_gpu_ms"] for x in st)
    g0 = 0
    while g0 < len(st) and st[g0]["verify_gpu_ms"] > 1.5 * med:
        g0 += 1
    gen = st[g0:]
    if not gen:
        return []
    mode = collections.Counter(x["n_dp"] for x in gen).most_common(1)[0][0]
    mmed = statistics.median(x["verify_gpu_ms"] for x in gen if x["n_dp"] == mode)
    for i, x in enumerate(gen):
        x["i"] = i
        x["typical"] = int(i >= skip and x["n_dp"] == mode and x["stray"] == 0
                           and abs(x["verify_gpu_ms"] / mmed - 1.0) <= tol)
    return gen


def attach_qsa(seg, gen, qsa, main_devs, draft_last, ss):
    """Matches the request's k-pool lines to its passes in order; -> (matched, passes, lines, pass_seq). Each step gets
    qsa_main (its verify pass's line) and qsa_draft (its draft passes' lines)."""
    seq = pass_seq(seg, main_devs, draft_last, ss)
    real = [k for k in seq if k[0] != "stray"]
    if len(real) != len(qsa):
        return False, len(real), len(qsa), seq
    main_line = [q for (kind, _), q in zip(real, qsa) if kind == "main"]
    # the lines after main pass j and before main pass j + 1 are its draft passes'
    after = collections.defaultdict(list)
    j = -1
    for (kind, _), q in zip(real, qsa):
        if kind == "main":
            j += 1
        elif j >= 0:
            after[j].append(q)
    for x in gen:
        x["qsa_main"] = main_line[x["main_index"]]
        x["qsa_draft"] = after[x["main_index"]][:x["n_dp"]]
    for (kind, _), q in zip(real, qsa):
        q["kind"] = kind
    return True, len(real), len(qsa), seq


def fmean(xs):
    xs = [v for v in xs if v is not None]
    return statistics.fmean(xs) if xs else float("nan")


def fmed(xs):
    xs = [v for v in xs if v is not None]
    return statistics.median(xs) if xs else float("nan")


def parts_of(typ, n_main, a):
    """-> [(label, fn)] of the step's parts for steps with the same number of draft passes."""
    n_dp = typ[0]["n_dp"]
    p = [("**the step**", lambda x: x["step_ms"]),
         ("GPU time, every evaluation", lambda x: x["gpu_all"]),
         ("**outside the kernels** (the step less that)", lambda x: x["outside"]),
         ("verify pass: GPU time of the nine cards", lambda x: x["verify_gpu_ms"]),
         ("verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft)",
          lambda x: x["verify_rest_ms"]),
         ("verify pass: the host's launch, card 0's entry to card 8's exit", lambda x: x["launch_ms"])]
    for k in range(n_dp):
        what = "catch-up" if k == 0 else f"draft {k}"
        p += [(f"draft pass {k} ({what}): crossing (card 4's exit to card 8's entry)",
               lambda x, k=k: x["passes"][k]["cross"]),
              (f"draft pass {k}: lead (card 4's entry to card 8's entry)", lambda x, k=k: x["passes"][k]["lead"]),
              (f"draft pass {k}: GPU time, card 4 and card 8", lambda x, k=k: x["passes"][k]["gpu"] + x["passes"][k]["emb_gpu"]),
              (f"draft pass {k}: turn" + (" (the server before the next verify pass)" if k == n_dp - 1 else ""),
               lambda x, k=k: x["passes"][k]["turn"])]
    p += [("the draft's crossings, all passes", lambda x: sum(d["cross"] for d in x["passes"])),
          ("the draft's turns before its last pass", lambda x: sum(d["turn"] for d in x["passes"][:-1]))]
    if a.qsa_ok:
        p += [("k-pool inputs: host us a step, all passes (set_input)",
               lambda x: x["qsa_main"]["us"] + sum(q["us"] for q in x["qsa_draft"]) if "qsa_main" in x else None),
              ("k-pool inputs: host us, the verify pass", lambda x: x["qsa_main"]["us"] if "qsa_main" in x else None),
              ("k-pool inputs: host us, the draft's passes", lambda x: sum(q["us"] for q in x["qsa_draft"]) if "qsa_main" in x else None),
              ("k-pool inputs: KiB a card, the verify pass", lambda x: x["qsa_main"]["kib"] if "qsa_main" in x else None),
              ("k-pool inputs: KiB, one draft pass (card 8)",
               lambda x: fmean([q["kib"] for q in x["qsa_draft"][1:]]) if "qsa_main" in x and len(x["qsa_draft"]) > 1 else None),
              (f"k-pool inputs: KiB a step to every card ({a.main_qsa_devices} in the verify pass, {a.draft_qsa_devices} a draft pass)",
               lambda x: x["qsa_main"]["kib"] * a.main_qsa_devices + sum(q["kib"] for q in x["qsa_draft"]) * a.draft_qsa_devices
               if "qsa_main" in x else None),
              ("k-pool inputs: KiB a step on the token's path (card 0's and every draft pass's)",
               lambda x: x["qsa_main"]["kib"] + sum(q["kib"] for q in x["qsa_draft"]) * a.draft_qsa_devices
               if "qsa_main" in x else None),
              ("k-pool inputs: n_pool of the verify pass", lambda x: x["qsa_main"]["n_pool"] if "qsa_main" in x else None)]
    for d in range(n_main):
        p.append((f"verify pass: card {d}'s host gap before entry" if d else "verify pass: card 0's host span",
                  (lambda x, d=d: x["gap"][d]) if d else (lambda x: x["span"][0])))
    return p


def run(a, ss, file=sys.stdout):
    def out(s=""):
        print(s, file=file)

    main_devs = [int(v) for v in a.main_devices.split(",")]
    draft_last = a.draft_last
    rows = ss.load_rows(a.host)
    segs = ss.cut(rows, a.gap_ms)
    # a request's own pause can cut it: the first decode at a depth saves a context checkpoint after the tail's pass,
    # which leaves its piece without the cards for seconds; a piece after the load's with fewer than --join-below main
    # passes is joined to the next piece
    joined = [segs[0]]
    carry = []
    for s in segs[1:]:
        if len(ss.main_passes(s, main_devs)) < a.join_below and s is not segs[-1]:
            carry += s
            continue
        joined.append(carry + s)
        carry = []
    n_cut = len(segs)
    segs = joined
    reqs = read_requests(a.requests, a.load)
    names = [req_name(r) for r in reqs]
    if len(segs) - 1 != len(reqs):
        raise SystemExit(f"host_depth: {len(segs)} pieces in the profile, {len(reqs)} requests of load {a.load} "
                         f"(+1 for the load's own piece expected)")
    log = read_log(a.log) if a.log else None
    if log is not None and len(log) - 1 != len(reqs):
        raise SystemExit(f"host_depth: {len(log) - 1} requests in the server log, {len(reqs)} in requests.csv")
    ref = {req_name(r): r for r in read_requests(a.requests, a.ref_load)} if a.ref_load else {}

    out(f"**Load `{a.load}`:** {len(rows)} evaluations in {n_cut} pieces (cut at gaps over {a.gap_ms:.0f} ms), "
        f"{len(segs)} after joining those with fewer than {a.join_below} main passes to the next; the first is the "
        f"load's own, the others are its {len(reqs)} requests.")
    out()
    out("| piece | request | evaluations | main passes | draft passes | k-pool lines | matched | generation steps | typical steps |")
    out("|---|---|---:|---:|---:|---:|---|---:|---:|")
    a.qsa_ok = log is not None
    all_gen, qsa_rows, piece_rows = {}, [], []
    lines0 = log[0][1] if log else []
    seq0 = pass_seq(segs[0], main_devs, draft_last, ss)
    out(f"| 0 | (the load) | {len(segs[0])} | {sum(k == 'main' for k, _ in seq0)} | {sum(k == 'draft' for k, _ in seq0)} "
        f"| {len(lines0)} | | | |")
    for q in lines0:
        qsa_rows.append(["(load)", "", q["n_pool"], q["n_tokens"], q["us"], q["kib"]])
    for idx, (seg, r, name) in enumerate(zip(segs[1:], reqs, names), start=1):
        gen = steps(seg, main_devs, draft_last, a.skip, a.tol, ss) if r["kind"] == "decode" else []
        lines = log[idx][1] if log else []
        ok, n_p, n_q, seq = attach_qsa(seg, gen, lines, main_devs, draft_last, ss) if log else \
            (False, 0, 0, pass_seq(seg, main_devs, draft_last, ss))
        if log and not ok:
            print(f"WARNING: {name}: {n_p} passes and {n_q} k-pool lines; its lines are left unmatched", file=sys.stderr)
        for q in lines:
            qsa_rows.append([name, q.get("kind", ""), q["n_pool"], q["n_tokens"], q["us"], q["kib"]])
        n_main = sum(k == "main" for k, _ in seq) if seq else len(ss.main_passes(seg, main_devs))
        n_draft = sum(k == "draft" for k, _ in seq)
        out(f"| {idx} | {name} | {len(seg)} | {n_main} | {n_draft} | {len(lines)} | {'yes' if ok else 'no'} "
            f"| {len(gen)} | {sum(x['typical'] for x in gen)} |")
        piece_rows.append([idx, name, len(seg), n_main, n_draft, len(lines), int(ok), len(gen), sum(x["typical"] for x in gen)])
        if gen:
            all_gen[name] = gen
    out()

    # the k-pool lines by request, pass kind and tokens
    if qsa_rows:
        out("**The k-pool inputs by request, pass and micro-batch size** (one line per evaluated micro-batch; the "
            "median, and in brackets the lowest and highest):")
        out()
        out("| request | pass | tokens | micro-batches | n_pool | host us | KiB a card |")
        out("|---|---|---:|---:|---|---|---|")
    elif log is not None:
        out("**No k-pool line in the server log** (LLAMA_QSA_INPUT_STATS logs at the library's info level, which "
            "llama-server prints from `-lv 4` on).")
    groups = collections.OrderedDict()
    for name, kind, n_pool, n_tok, us, kib in qsa_rows:
        groups.setdefault((name, kind, n_tok), []).append((n_pool, us, kib))
    for (name, kind, n_tok), v in groups.items():
        pools = sorted({p for p, _, _ in v})
        pr = f"{pools[0]}" if len(pools) == 1 else f"{pools[0]} to {pools[-1]}"
        us = [u for _, u, _ in v]
        kib = [k for _, _, k in v]
        out(f"| {name} | {kind or '?'} | {n_tok} | {len(v)} | {pr} | {fmed(us):.1f} ({min(us):.1f}, {max(us):.1f}) "
            f"| {fmed(kib):.1f} ({min(kib):.1f}, {max(kib):.1f}) |")
    out()

    summary = []
    dec = [n for n in names if n in all_gen]
    # the parts per decode, typical steps
    cols = [n for n in dec if any(x["typical"] for x in all_gen[n])]
    if not cols:
        out("No typical steps.")
        return
    n_dp = collections.Counter(x["n_dp"] for n in cols for x in all_gen[n] if x["typical"]).most_common(1)[0][0]
    typs = {n: [x for x in all_gen[n] if x["typical"] and x["n_dp"] == n_dp] for n in cols}
    parts = parts_of(typs[cols[0]], len(main_devs), a)
    means = {n: {} for n in cols}
    meds = {n: {} for n in cols}
    for label, fn in parts:
        for n in cols:
            means[n][label] = fmean([fn(x) for x in typs[n]])
            meds[n][label] = fmed([fn(x) for x in typs[n]])
    out(f"**The step by part, ms** (the mean over each decode's typical steps with {n_dp} draft passes; every column is "
        "one request of the one load):")
    out()
    out("| part | " + " | ".join(cols) + " |")
    out("|---|" + "---:|" * len(cols))
    out("| typical steps | " + " | ".join(str(len(typs[n])) for n in cols) + " |")
    for label, _ in parts:
        out(f"| {label} | " + " | ".join(f"{means[n][label]:.2f}" for n in cols) + " |")
    if ref:
        out("| the server's step, this load | " + " | ".join(f"{server_step(r):.2f}" for r, nm in zip(reqs, names) if nm in cols) + " |")
        out(f"| the server's step, load `{a.ref_load}` | " + " | ".join(f"{server_step(ref[n]):.2f}" if n in ref else "" for n in cols) + " |")
    out()
    out("Medians of the same:")
    out()
    out("| part | " + " | ".join(cols) + " |")
    out("|---|" + "---:|" * len(cols))
    for label, _ in parts:
        out(f"| {label} | " + " | ".join(f"{meds[n][label]:.2f}" for n in cols) + " |")
    out()
    for n in cols:
        for label, _ in parts:
            summary.append([n, label, f"{means[n][label]:.4f}", f"{meds[n][label]:.4f}", len(typs[n])])

    # growth between depths
    for pair in (a.grow.split(",") if a.grow else []):
        lo, hi = pair.split(":")
        gcols = []
        for p in ("1", "2"):
            n0, n1 = f"decode {lo} p{p}", f"decode {hi} p{p}"
            if n0 in means and n1 in means:
                gcols.append((p, n0, n1))
        if not gcols:
            continue
        out(f"**From {lo} to {hi} tokens, ms a step** (the deeper request less the shallower, pass against pass; "
            "the share is of the step's growth):")
        out()
        out("| part | " + " | ".join(f"p{p}: {lo}" + f" | p{p}: {hi} | p{p}: growth | p{p}: share" for p, _, _ in gcols) + " |")
        out("|---|" + "---:|---:|---:|---:|" * len(gcols))
        for label, _ in parts:
            cells = []
            for p, n0, n1 in gcols:
                g = means[n1][label] - means[n0][label]
                gs = means[n1]["**the step**"] - means[n0]["**the step**"]
                share = f"{100.0 * g / gs:.1f} %" if gs and not label.startswith("k-pool") else ""
                cells += [f"{means[n0][label]:.2f}", f"{means[n1][label]:.2f}", f"{g:+.2f}", share]
            out(f"| {label} | " + " | ".join(cells) + " |")
        out()

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "pieces.tsv"), "w") as f:
            f.write("piece\trequest\tevaluations\tmain_passes\tdraft_passes\tqsa_lines\tmatched\tgeneration_steps\ttypical_steps\n")
            for r in piece_rows:
                f.write("\t".join(str(v) for v in r) + "\n")
        with open(os.path.join(a.out, "qsa.tsv"), "w") as f:
            f.write("request\tpass\tn_pool\tn_tokens\thost_us\tkib_per_card\n")
            for r in qsa_rows:
                f.write("\t".join(str(v) for v in r) + "\n")
        with open(os.path.join(a.out, "summary.tsv"), "w") as f:
            f.write("request\tpart\tmean_ms\tmedian_ms\ttypical_steps\n")
            for r in summary:
                f.write("\t".join(str(v) for v in r).replace("**", "") + "\n")
        npass = max((x["n_dp"] for g in all_gen.values() for x in g), default=0)
        with open(os.path.join(a.out, "steps.tsv"), "w") as f:
            c = ["request", "i", "typical", "n_dp", "step_ms", "gpu_all_ms", "outside_ms", "verify_span_ms",
                 "verify_gpu_ms", "verify_rest_ms", "launch_ms"]
            c += [f"card{d}_gpu_ms" for d in main_devs] + [f"card{d}_gap_ms" for d in main_devs[1:]]
            c += [f"card{d}_span_ms" for d in main_devs]
            for k in range(npass):
                c += [f"lead{k}_ms", f"cross{k}_ms", f"emb_gpu{k}_ms", f"gpu{k}_ms", f"turn{k}_ms"]
            c += ["qsa_main_us", "qsa_main_kib", "qsa_main_n_pool", "qsa_main_tokens", "qsa_draft_us", "qsa_draft_kib",
                  "qsa_draft_tokens"]
            f.write("\t".join(c) + "\n")
            for name, g in all_gen.items():
                for x in g:
                    v = [name, x["i"], x["typical"], x["n_dp"]]
                    v += [f"{x[k]:.3f}" for k in ("step_ms", "gpu_all", "outside", "verify_span_ms", "verify_gpu_ms",
                                                  "verify_rest_ms", "launch_ms")]
                    v += [f"{g_:.3f}" for g_ in x["card_gpu"]] + [f"{g_:.3f}" for g_ in x["gap"][1:]]
                    v += [f"{s:.3f}" for s in x["span"]]
                    for k in range(npass):
                        if k < len(x["passes"]):
                            d = x["passes"][k]
                            v += [f"{d['lead']:.3f}", f"{d['cross']:.3f}", f"{d['emb_gpu']:.3f}", f"{d['gpu']:.3f}",
                                  f"{d['turn']:.3f}"]
                        else:
                            v += [""] * 5
                    if "qsa_main" in x:
                        qm, qd = x["qsa_main"], x["qsa_draft"]
                        v += [f"{qm['us']:.1f}", f"{qm['kib']:.1f}", qm["n_pool"], qm["n_tokens"],
                              ",".join(f"{q['us']:.1f}" for q in qd), ",".join(f"{q['kib']:.1f}" for q in qd),
                              ",".join(str(q["n_tokens"]) for q in qd)]
                    else:
                        v += [""] * 7
                    f.write("\t".join(str(t) for t in v) + "\n")


def self_test(a, ss):
    """A synthetic load: the load's piece, a warm-up, a prime and two decodes at each of two depths, with known parts
    and k-pool lines; then a log with one line too few."""
    fails, n_checks = [], [0]

    def check(name, got, want, tol=1e-6):
        n_checks[0] += 1
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        if not ok:
            fails.append(f"{name}: {got} != {want}")

    tmp = tempfile.mkdtemp(prefix="host_depth_")
    rows, log, reqs = [], ["0.00.000.001 I llama-server: loading\n"], []
    t = [0.0]

    def main_pass(gpu=10.0, gaps=0.3, span=0.4, n_tok=4, n_pool=100, us=50.0, kib=10.0, prompt=False):
        t0 = t[0]
        e = t0
        for d in range(9):
            rows.append((d, e, e + span, gpu * (3 if prompt else 1)))
            e += span + gaps
        log.append(f"0.00.000.000 I f: QSA k-pool inputs, last 1 ubatches: {us:.1f} us on the host and {kib:.1f} KiB per device per ubatch (n_pool {n_pool}, {n_tok} tokens)\n")
        t[0] = t0 + 9 * gpu * (3 if prompt else 1) + 5.0  # verify_rest 5 ms

    def draft_pass(lead=1.5, cross=1.0, gpu8=3.0, turn=0.5, n_tok=1, n_pool=100, us=40.0, kib=8.0):
        e4 = t[0]
        rows.append((4, e4, e4 + lead - cross, 0.2))
        e8 = e4 + lead
        rows.append((8, e8, e8 + 0.3, gpu8))
        log.append(f"0.00.000.000 I f: QSA k-pool inputs, last 1 ubatches: {us:.1f} us on the host and {kib:.1f} KiB per device per ubatch (n_pool {n_pool}, {n_tok} tokens)\n")
        t[0] = e8 + gpu8 + turn

    def request(kind, depth, p, n_steps, cross, n_pool, us_main, kib_main, task):
        log.append(f"0.00.000.000 I slot launch_slot_: id  0 | task {task} | processing task, is_child = 0\n")
        if kind == "prime":
            for _ in range(12):
                main_pass(prompt=True, n_tok=512, n_pool=n_pool, us=900.0, kib=kib_main * 100)
                for _ in range(4):
                    draft_pass(n_tok=128, n_pool=n_pool)
        else:
            main_pass(prompt=True, n_tok=64 if p == 1 else 4, n_pool=n_pool)
            draft_pass(n_tok=4, n_pool=n_pool)
            if p == 1:
                t[0] += 4000.0  # the checkpoint after the tail's pass: this piece is joined to the next

            for _ in range(n_steps):
                main_pass(n_pool=n_pool, us=us_main, kib=kib_main)
                draft_pass(cross=cross, lead=cross + 0.5, n_tok=4, n_pool=n_pool, us=us_main * 0.8, kib=kib_main * 0.8)
                for _ in range(3):
                    draft_pass(cross=cross, lead=cross + 0.5, n_pool=n_pool, us=us_main * 0.8, kib=kib_main * 0.8)
        log.append(f"0.00.000.000 I slot      release: id  0 | task {task} | stop processing\n")
        reqs.append(dict(arm="host", load="host", kind=kind, depth=str(depth), **{"pass": str(p)}, gen_n="256",
                         gen_ms="10000", draft_accepted="170"))
        t[0] += 5000.0

    main_pass(n_tok=1)  # the load's own warm-up evaluation
    draft_pass()
    t[0] += 5000.0
    request("warmup", 0, 0, 12, 1.0, 16, 10.0, 1.0, 0)
    task = 10
    for depth, cross, n_pool, us, kib in ((16384, 1.0, 4096, 100.0, 110.0), (49152, 2.5, 12288, 300.0, 330.0)):
        request("prime", depth, 0, 0, cross, n_pool, us, kib, task)
        request("decode", depth, 1, 20, cross, n_pool, us, kib, task + 1)
        request("decode", depth, 2, 20, cross, n_pool, us, kib, task + 2)
        task += 3
    host = os.path.join(tmp, "host.csv")
    with open(host, "w") as f:
        f.write("device,eval,host_enter_ms,host_exit_ms,gpu_us\n")
        for i, (d, e, x, g) in enumerate(rows):
            f.write(f"{d},{i},{e:.3f},{x:.3f},{1000 * g:.2f}\n")
    logp = os.path.join(tmp, "server-host.log")
    with open(logp, "w") as f:
        f.writelines(log)
    reqp = os.path.join(tmp, "requests.csv")
    cols = ["arm", "load", "kind", "pass", "depth", "gen_n", "gen_ms", "draft_accepted"]
    with open(reqp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in reqs:
            w.writerow(r)
    b = argparse.Namespace(**vars(a))
    b.host, b.log, b.requests, b.load, b.ref_load, b.out = host, logp, reqp, "host", None, os.path.join(tmp, "res")
    b.grow = "16384:49152"
    import io
    buf = io.StringIO()
    run(b, ss, file=buf)
    text = buf.getvalue()
    with open(os.path.join(b.out, "summary.tsv")) as f:
        summ = {(r["request"], r["part"]): float(r["mean_ms"]) for r in csv.DictReader(f, delimiter="\t")}
    with open(os.path.join(b.out, "pieces.tsv")) as f:
        pieces = list(csv.DictReader(f, delimiter="\t"))
    check("pieces", len(pieces), 7)
    check("each first decode's tail pass joined to its generation", "in 10 pieces" in text and "8 after joining" in text, True)
    check("every piece matched", all(p["matched"] == "1" for p in pieces), True)
    check("main passes of a decode", int(pieces[2]["main_passes"]), 21)
    check("draft passes of a decode", int(pieces[2]["draft_passes"]), 81)
    # 21 main passes: the tail's prompt pass and 20 steps' verify passes; 20 pairs, the first a prompt pass; skip 3
    check("generation steps", int(pieces[2]["generation_steps"]), 19)
    check("typical steps", int(pieces[2]["typical_steps"]), 16)
    for depth, cross, us, kib in ((16384, 1.0, 100.0, 110.0), (49152, 2.5, 300.0, 330.0)):
        n = f"decode {depth} p2"
        check(f"{n} verify gpu", summ[(n, "verify pass: GPU time of the nine cards")], 90.0, 1e-3)
        check(f"{n} verify rest", summ[(n, "verify pass: the rest of its span (crossings, rows back, acceptance, the server before the draft)")], 5.0, 1e-3)
        check(f"{n} crossing", summ[(n, "draft pass 2 (draft 2): crossing (card 4's exit to card 8's entry)")], cross, 1e-3)
        check(f"{n} gap", summ[(n, "verify pass: card 5's host gap before entry")], 0.3, 1e-3)
        check(f"{n} gpu all", summ[(n, "GPU time, every evaluation")], 90.0 + 4 * 3.2, 1e-3)
        step = 95.0 + 4 * (cross + 0.5 + 3.0 + 0.5)
        check(f"{n} step", summ[(n, "the step")], step, 1e-3)
        check(f"{n} outside", summ[(n, "outside the kernels (the step less that)")], step - 90.0 - 12.8, 1e-3)
        check(f"{n} qsa us", summ[(n, "k-pool inputs: host us a step, all passes (set_input)")], us + 4 * 0.8 * us, 1e-3)
        check(f"{n} qsa kib all", summ[(n, "k-pool inputs: KiB a step to every card (9 in the verify pass, 1 a draft pass)")],
              9 * kib + 4 * 0.8 * kib, 1e-3)
    check("growth table", "From 16384 to 49152 tokens" in text, True)
    # one k-pool line missing in a decode: that request is left unmatched, the others are kept
    with open(logp) as f:
        lines = f.readlines()
    k = max(i for i, l in enumerate(lines) if "task 11 |" in l and "launch" in l) + 5
    del lines[k]
    with open(logp, "w") as f:
        f.writelines(lines)
    buf = io.StringIO()
    run(b, ss, file=buf)
    with open(os.path.join(b.out, "pieces.tsv")) as f:
        pieces = list(csv.DictReader(f, delimiter="\t"))
    check("a short request unmatched", pieces[2]["matched"], "0")
    check("the others matched", sum(p["matched"] == "1" for p in pieces), 6)
    # a request missing from the profile: the script stops
    with open(reqp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writerow(dict(arm="host", load="host", kind="decode", depth="49152", **{"pass": "3"}, gen_n="1", gen_ms="1",
                        draft_accepted="0"))
    n_checks[0] += 1
    try:
        run(b, ss, file=io.StringIO())
        fails.append("a missing piece did not stop the script")
    except SystemExit:
        pass
    n = n_checks[0]
    print(f"self-test: {n - len(fails)} of {n} checks pass" + ("" if not fails else ": " + "; ".join(fails)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step-split", required=True, help="2026-11-10_served_step_split/run/step_split.py")
    ap.add_argument("--host")
    ap.add_argument("--log")
    ap.add_argument("--requests")
    ap.add_argument("--load", default="host")
    ap.add_argument("--ref-load")
    ap.add_argument("--grow", default="16384:49152")
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--join-below", type=int, default=10)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--main-devices", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--draft-last", type=int, default=8)
    ap.add_argument("--main-qsa-devices", type=int, default=9)
    ap.add_argument("--draft-qsa-devices", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    ss = load_mod(a.step_split, "step_split")
    if a.self_test:
        sys.exit(self_test(a, ss))
    if not (a.host and a.requests):
        ap.error("--host and --requests are needed")
    run(a, ss)


if __name__ == "__main__":
    main()
