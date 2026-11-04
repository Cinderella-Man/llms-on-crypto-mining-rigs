#!/usr/bin/env python3
"""prof_summary.py CSV [MAX_EVAL_MS] [--event-us C] [--wall-ms W] [--how auto|replay|all] [--tol-pct T]
prof_summary.py --self-test [-v]

A GGML_CUDA_PROFILE trace of the llama.cpp fork (Miner Llama) summarized per decoded token, in any of its modes
(GGML_CUDA_PROFILE_MODE, Q681):
  group       the default (columns device,eval,host_ms,group,label,us): one line per launch group of every evaluation,
              an event between every two groups and a stream sync after every evaluation;
  layer       one line per layer segment of every evaluation;
  sample=<N>  one line per timed launch group: every N-th group, rotating per evaluation;
  graph       one line per evaluation.
The last three files have the columns device,eval,host_ms,group,label,us,mode,graph,phase,how,groups,eval_groups,eval_us,
with the evaluation's own GPU time (eval_us) and how it ran: "replay" (captured CUDA graphs launched), "capture" or
"eager". Lines with group -1 are evaluations the profiler could not read ("(not ready)", "(unmatched)"): counted, skipped.

An evaluation on one device whose GPU time is below MAX_EVAL_MS (default 60) counts as a decode-sized one, so a server
trace with prompts mixed in needs no token count; a decode step is (decode-sized evaluations) / (devices), as one
token of plain decoding evaluates each card's graph once. --how auto (the default) keeps only replays when the trace
has any (the captures and eager evaluations of a warm-up run at another speed); all keeps every evaluation.

--event-us C subtracts the profiler's own cost, C microseconds, from every timed span (one event ends each span). C is
the per-group cost of the profiler's events on the card, measured by bench/megakernel_p106 as `graph+events` minus
`graph` microseconds per stage on its census6 mix (C140); 0 by default, which reports the raw times.

Reported per decode step: per launch group, per first op and per weight type (group and sample modes), per layer
(layer mode) or per card (all modes), each with its share of the step; the sum of the timed spans; the whole
evaluations' GPU time (light modes); the wall time per step of this run (median interval between the lowest
device's decode-sized evaluations, host clock); and, with --wall-ms W (the unprofiled time per token measured
separately, e.g. the server's predicted_per_token_ms on a run without the profiler), the sum against W: what the
cards' timelines do not hold (hops between cards, host work) is W minus the sum. --tol-pct T adds a "check:" line
(the sum within T % of W). The last line, "summary: key=value ...", is for scripts.

Runs on the quant machine or a rig (standard library only). scripts/cuda_profile_summary.py summarizes the last N
evaluations of a llama-bench trace instead.
"""
import argparse, collections, csv, io, math, os, statistics, sys, tempfile

LIGHT_COLUMNS = ["device", "eval", "host_ms", "group", "label", "us", "mode", "graph", "phase", "how", "groups",
                 "eval_groups", "eval_us"]
LEGACY_COLUMNS = LIGHT_COLUMNS[:6]


class Eval:
    """One evaluation of one device's graph: its timed spans as (group, label, us, n_groups)."""
    __slots__ = ("device", "eval", "host_ms", "graph", "phase", "how", "eval_us", "eval_groups", "spans")

    def __init__(self, device, ev, host_ms, graph=0, phase=0, how="?", eval_us=None, eval_groups=0):
        self.device, self.eval, self.host_ms = device, ev, host_ms
        self.graph, self.phase, self.how = graph, phase, how
        self.eval_us, self.eval_groups = eval_us, eval_groups
        self.spans = []


def read_trace(path):
    """-> (mode, {(device, eval): Eval}, Counter of skipped labels)."""
    evals, skipped, mode = {}, collections.Counter(), "group"
    with open(path, newline="") as f:
        rows = csv.DictReader(f)
        light = "mode" in (rows.fieldnames or [])
        for row in rows:
            d, ev, group = int(row["device"]), int(row["eval"]), int(row["group"])
            if light:
                mode = row["mode"]
                if group < 0:
                    skipped[row["label"]] += 1
                    continue
            e = evals.get((d, ev))
            if e is None:
                if light:
                    e = Eval(d, ev, float(row["host_ms"]), int(row["graph"]), int(row["phase"]), row["how"],
                             float(row["eval_us"]), int(row["eval_groups"]))
                else:
                    e = Eval(d, ev, float(row["host_ms"]))
                evals[(d, ev)] = e
            n = int(row["groups"]) if light else 1
            e.spans.append((group, row["label"], float(row["us"]), n))
    for e in evals.values():
        if e.eval_us is None:  # group mode: the spans tile the evaluation
            e.eval_us = sum(us for _, _, us, _ in e.spans)
            e.eval_groups = len(e.spans)
    return mode, evals, skipped


def events_per_span(mode):
    """Events inside an evaluation per timed span: one ends each span, and sample mode's spans also start with one."""
    return {"group": 1, "layer": 1, "sample": 2, "graph": 1}.get(mode, 1)


def summarize(path, max_ms=60.0, event_us=0.0, wall_ms=None, how="auto", tol_pct=None, top=30, out=sys.stdout):
    mode, evals, skipped = read_trace(path)
    p = lambda *a: print(*a, file=out)
    dec = {k: e for k, e in evals.items() if e.eval_us < max_ms * 1000}
    pre = len(evals) - len(dec)
    devs = sorted({d for d, _ in evals})
    n_steps = max(1, len(dec) // max(1, len(devs)))
    hows = collections.Counter(e.how for e in dec.values())
    use_replay = how == "replay" or (how == "auto" and hows.get("replay", 0) > 0)
    sel = [e for e in dec.values() if not use_replay or e.how == "replay"]
    n_all = collections.Counter((e.device, e.graph) for e in dec.values())
    n_sel = collections.Counter((e.device, e.graph) for e in sel)
    res = {"mode": mode, "steps": n_steps, "devices": len(devs), "decode_evals": len(dec), "used_evals": len(sel),
           "skipped": sum(skipped.values())}
    if mode == "group" and "?" in hows:
        p(f"{len(evals)} device evals: {len(dec)} decode-sized (< {max_ms} ms), {pre} larger; ~{n_steps} decode steps")
    else:
        p(f"{path}: mode {mode}, {len(evals)} device evals on {len(devs)} devices: {len(dec)} decode-sized "
          f"(< {max_ms:g} ms; {', '.join(f'{v} {k}' for k, v in sorted(hows.items()))}), {pre} larger; "
          f"~{n_steps} decode steps; used: {'replays' if use_replay else 'all'} ({len(sel)})")
    if skipped:
        p("skipped (the profiler could not read them): " + ", ".join(f"{v} {k}" for k, v in skipped.items()))
    if not sel:
        p("no decode-sized evaluation to summarize")
        res.update(sum_ms=float("nan"))
        return res

    # per (device, graph): the time per step of each label, from the mean over the used evaluations, weighted by how
    # often that graph runs per step (its decode-sized evaluations / steps)
    by_dg = collections.defaultdict(list)
    for e in sel:
        by_dg[(e.device, e.graph)].append(e)
    items = collections.Counter()   # (device, graph, label) -> us per step, corrected
    per_group = {}                  # sample mode: (device, graph, group) -> [corrected us]
    n_short = n_spans_used = 0
    evals_us = evals_corr_us = 0.0
    coverage = []
    for (d, g), es in by_dg.items():
        w = n_all[(d, g)] / n_steps / len(es)
        for e in es:
            evals_us += e.eval_us * w
            evals_corr_us += (e.eval_us - event_us * events_per_span(mode) * len(e.spans)) * w
        if mode == "sample":
            samples, labels = collections.defaultdict(list), {}
            for e in es:
                for group, label, us, _ in e.spans:
                    samples[group].append(us - event_us)
                    labels[group] = label
                    n_short += us < event_us
                    n_spans_used += 1
            ratio = n_all[(d, g)] / n_steps
            for group, v in samples.items():
                per_group[(d, g, group)] = v
                items[(d, g, labels[group])] += statistics.fmean(v) * ratio
            n_groups = statistics.median_low([e.eval_groups for e in es]) or 1
            coverage.append((len(samples) / n_groups, n_all[(d, g)]))
        else:
            for e in es:
                for _, label, us, _ in e.spans:
                    items[(d, g, label)] += (us - event_us) * w
                    n_short += us < event_us
                    n_spans_used += 1
    total = sum(items.values())
    per_dev = collections.Counter()
    per_label = collections.Counter()
    for (d, g, label), us in items.items():
        per_dev[d] += us
        per_label[label] += us
    corr = f", {event_us:g} µs per span subtracted" if event_us else ""
    p(f"decode: {total / 1000:.2f} ms of GPU time per token summed over cards{corr} "
      f"(≈ {1e6 / total:.2f} tok/s if the cards ran back to back with no gaps)" if total > 0 else
      f"decode: {total / 1000:.2f} ms per token summed over cards{corr}")
    p("per card, ms per token: " + " ".join(f"{d}:{per_dev[d] / 1000:.2f}" for d in devs))
    if event_us and n_spans_used:
        p(f"spans shorter than the event cost: {n_short} of {n_spans_used} ({100 * n_short / n_spans_used:.1f} %)")
    share = lambda us: 100 * us / total if total else float("nan")
    if mode in ("group", "sample"):
        by_op, by_type = collections.Counter(), collections.Counter()
        for label, us in per_label.items():
            ops, _, weights = label.partition(" ")
            by_op[ops.split("+")[0]] += us
            types = sorted({w.split(":")[1] for w in weights.split() if w.count(":") == 2})
            by_type["+".join(types) or "(no weights)"] += us
        p("\nby first op of the launch group (ms/token, share):")
        for k, us in by_op.most_common(15):
            p(f"  {us / 1000:7.2f} {share(us):5.1f} %  {k}")
        p("\nby weight type (ms/token, share):")
        for k, us in by_type.most_common(15):
            p(f"  {us / 1000:7.2f} {share(us):5.1f} %  {k}")
        p("\ntop launch groups (ms/token, share, label):")
        for k, us in per_label.most_common(top):
            p(f"  {us / 1000:7.2f} {share(us):5.1f} %  {k[:150]}")
        if mode == "sample":
            cmin = min(c for c, _ in coverage)
            cmain = statistics.fmean(c for c, _ in sorted(coverage, key=lambda x: -x[1])[:len(devs)])
            res["coverage"] = cmin
            n_samp = [len(v) for v in per_group.values()]
            p(f"\nsampling: {len(per_group)} launch groups timed, {min(n_samp)}-{max(n_samp)} samples each "
              f"(median {statistics.median(n_samp):g}); coverage of each graph's groups: lowest {100 * cmin:.0f} %, "
              f"main graphs {100 * cmain:.0f} %")
    elif mode == "layer":
        p("\nper card and layer (ms/token, share):")
        for (d, g, label), us in sorted(items.items(), key=lambda x: (x[0][0], x[0][1], layer_order(x[0][2]))):
            p(f"  card {d} graph {g}  {us / 1000:7.2f} {share(us):5.1f} %  {label}")
    elif mode == "graph":
        p("\nper card and graph (ms/token, share, evaluations used/decode-sized):")
        for (d, g, label), us in sorted(items.items()):
            p(f"  card {d} graph {g}  {us / 1000:7.2f} {share(us):5.1f} %  {n_sel[(d, g)]}/{n_all[(d, g)]}")

    # the wall time per step: intervals between the lowest device's decode-sized evaluations of its main graph
    d0 = devs[0]
    g0 = max((k for k in n_all if k[0] == d0), key=lambda k: n_all[k])[1]
    t = sorted((e.eval, e.host_ms) for e in dec.values() if e.device == d0 and e.graph == g0)
    gaps = [b[1] - a[1] for a, b in zip(t, t[1:]) if b[1] > a[1]]
    wall_run = statistics.median(gaps) if gaps else float("nan")
    near = [x for x in gaps if x <= 1.5 * wall_run]
    wall_run_mean = statistics.fmean(near) if near else float("nan")
    s_ms, e_ms, ec_ms = total / 1000, evals_us / 1000, evals_corr_us / 1000
    p("\nper token (one decode step):")
    p(f"  timed spans, summed over cards    {s_ms:8.2f} ms{corr}")
    if mode != "group":
        p(f"  whole evaluations (eval_us)       {e_ms:8.2f} ms raw, {ec_ms:.2f} ms without the events inside them")
    p(f"  wall, this run (host clock)       {wall_run:8.2f} ms median, {wall_run_mean:.2f} ms mean "
      f"({len(gaps)} intervals of card {d0}'s graph {g0})")
    res.update(sum_ms=s_ms, evals_ms=e_ms, evals_corr_ms=ec_ms, wall_run_ms=wall_run, event_us=event_us)
    if wall_ms:
        p(f"  wall, unprofiled (--wall-ms)      {wall_ms:8.2f} ms")
        p(f"  timed spans / unprofiled wall     {100 * s_ms / wall_ms:8.1f} %  (unprofiled wall minus the spans: "
          f"{wall_ms - s_ms:.2f} ms of hops, host work and gaps)")
        p(f"  this run's wall / unprofiled      {100 * wall_run / wall_ms:8.1f} %  (the profiler's own cost on a token)")
        res.update(wall_ref_ms=wall_ms, sum_vs_ref_pct=100 * (s_ms / wall_ms - 1),
                   wall_vs_ref_pct=100 * (wall_run / wall_ms - 1))
        if tol_pct is not None:
            ok = abs(s_ms / wall_ms - 1) * 100 <= tol_pct
            res["check"] = "pass" if ok else "fail"
            p(f"check: timed spans {s_ms:.2f} ms against the unprofiled {wall_ms:.2f} ms per token: "
              f"{100 * (s_ms / wall_ms - 1):+.1f} %, {'within' if ok else 'outside'} {tol_pct:g} % -> {res['check']}")
    p("summary: " + " ".join(f"{k}={fmt(v)}" for k, v in res.items()))
    return res


def layer_order(label):
    if label.startswith("layer "):
        return (1, int(label[6:]))
    return {"input": (0, 0), "other": (0, 1), "output": (2, 0)}.get(label, (3, 0))


def fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------------------------------------------------
# self-test: synthetic traces in every mode from a known token, written the way the fork writes them

def synth_truth():
    """Three cards; card c holds layers 3c..3c+2 (8 launch groups each) and card 2 also the output (2 groups)."""
    ops = ["RMS_NORM+MUL attn_norm", "MUL_MAT attn_q:q4_0:2560x4096", "ADD ffn_inp", "MUL_MAT_ID ffn_up_exps:q2_0:2560x512",
           "SILU ffn_act", "MUL_MAT ffn_down:q8_0:512x2560", "SCALE hc_scale", "GATED_DELTA_NET gdn"]
    cards = []
    for c in range(3):
        groups = []
        for layer in range(3 * c, 3 * c + 3):
            for j, op in enumerate(ops):
                us = 2.0 + ((7 * layer + 13 * j) % 23) * 4.25 + (180.0 if "MUL_MAT" in op else 0.0)
                groups.append((f"layer {layer}", op, us))
        if c == 2:
            groups += [("output", "RMS_NORM+MUL result_norm", 3.5), ("output", "MUL_MAT output:q6_k:2560x151936", 2400.0)]
        cards.append(groups)
    return cards


def synth_write(path, mode, cards, c_true, n_tok, sample_n=4, hop_us=500.0, host_us=2000.0, warm=2):
    """One file in the profiler's format; returns the modelled wall per token (ms) of this profiled run."""
    light = mode != "group"
    rows = []
    t_ms = 0.0
    n_eval = collections.Counter()
    wall_tok = 0.0
    # two prompt-sized evaluations per card first (larger than MAX_EVAL_MS)
    for c in range(len(cards)):
        for _ in range(2):
            ev = n_eval[c]
            n_eval[c] += 1
            if light:
                rows.append([c, ev, f"{t_ms:.3f}", 0, "graph" if mode == "graph" else "layer 0", "200000.00", mode, 1, 0,
                             "eager", len(cards[c]), len(cards[c]), "200000.00"])
            else:
                rows.append([c, ev, f"{t_ms:.3f}", 0, "MUL_MAT prompt", "200000.00"])
            t_ms += 200.0
    for tok in range(n_tok):
        tok_us = 0.0
        for c, groups in enumerate(cards):
            ev = n_eval[c]
            n_eval[c] += 1
            phase = (tok % sample_n) if mode == "sample" else 0
            # the first evaluations of a light mode's graph are the warm-up's eager and capture runs, 3x slower
            how = "replay"
            slow = 1.0
            if light and tok < (warm if mode != "sample" else warm + sample_n - 1):
                how, slow = ("eager" if tok == 0 else "capture"), 3.0
            true_us = sum(us for _, _, us in groups)
            if mode == "group":
                spans = [(k, op, us + c_true, 1) for k, (_, op, us) in enumerate(groups)]
                n_ev_inside = len(groups)
            elif mode == "layer":
                spans, k0 = [], 0
                for layer in dict.fromkeys(l for l, _, _ in groups):
                    ks = [k for k, g in enumerate(groups) if g[0] == layer]
                    spans.append((ks[0], layer, sum(groups[k][2] for k in ks) + c_true, len(ks)))
                n_ev_inside = len(spans)
            elif mode == "sample":
                spans = [(k, op, us + c_true, 1) for k, (_, op, us) in enumerate(groups) if k % sample_n == phase]
                n_ev_inside = 2 * len(spans)
            else:
                spans = [(0, "graph", None, len(groups))]
                n_ev_inside = 1
            eval_us = (true_us + c_true * n_ev_inside) * slow
            for group, label, us, n in spans:
                us = eval_us if us is None else us * slow
                if light:
                    rows.append([c, ev, f"{t_ms:.3f}", group, label, f"{us:.2f}", mode, 0, phase, how, n, len(groups),
                                 f"{eval_us:.2f}"])
                else:
                    rows.append([c, ev, f"{t_ms:.3f}", group, label, f"{us:.2f}"])
            step = eval_us / 1000 + (hop_us / 1000 if c + 1 < len(cards) else 0)
            t_ms += step
            tok_us += step * 1000
        t_ms += host_us / 1000
        tok_us += host_us
        if tok == n_tok - 1:
            wall_tok = tok_us / 1000
    if light:  # one evaluation the profiler could not read
        rows.append([0, n_eval[0], f"{t_ms:.3f}", -1, "(not ready)", "0.00", mode, 0, 0, "replay", 0, 0, "-1"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LIGHT_COLUMNS if light else LEGACY_COLUMNS)
        w.writerows(rows)
    return wall_tok


def self_test(verbose=False):
    cards = synth_truth()
    true_ms = sum(us for g in cards for _, _, us in g) / 1000
    per_layer = collections.Counter()
    for g in cards:
        for layer, _, us in g:
            per_layer[layer] += us / 1000
    hop_us, host_us = 50.0, 200.0  # a token's hops and host work, ~3 % of it as on the rig
    wall_ref = true_ms + 2 * hop_us / 1000 + host_us / 1000
    c_true, n_tok = 8.0, 40
    checks, fails = 0, []

    def check(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(what)

    near = lambda a, b, rel=1e-3: abs(a - b) <= rel * max(abs(b), 1e-9)
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("group", "layer", "sample", "graph"):
            path = os.path.join(tmp, f"prof-{mode}.csv")
            wall_run = synth_write(path, mode, cards, c_true, n_tok, hop_us=hop_us, host_us=host_us)
            buf = io.StringIO()
            r = summarize(path, event_us=c_true, wall_ms=wall_ref, tol_pct=5.0, out=buf)
            if verbose:
                print(f"===== {mode}\n{buf.getvalue()}")
            check(r["steps"] == n_tok, f"{mode}: {r['steps']} decode steps, expected {n_tok}")
            check(near(r["sum_ms"], true_ms), f"{mode}: sum {r['sum_ms']:.4f} ms, expected {true_ms:.4f}")
            check(near(r["evals_corr_ms"], true_ms), f"{mode}: whole evaluations without events "
                                                     f"{r['evals_corr_ms']:.4f} ms, expected {true_ms:.4f}")
            check(near(r["wall_run_ms"], wall_run, 2e-3), f"{mode}: wall {r['wall_run_ms']:.3f}, expected {wall_run:.3f}")
            check(r.get("check") == "pass", f"{mode}: the check against the unprofiled wall did not pass")
            check(near(r["sum_vs_ref_pct"], 100 * (true_ms / wall_ref - 1), 1e-2), f"{mode}: sum_vs_ref_pct")
            if mode != "group":
                check(r["skipped"] == 1, f"{mode}: {r['skipped']} skipped lines, expected 1")
                warm = 2 if mode != "sample" else 2 + 4 - 1
                check(r["used_evals"] == (n_tok - warm) * len(cards), f"{mode}: {r['used_evals']} evaluations used")
            if mode == "sample":
                check(near(r.get("coverage", 0), 1.0), f"sample: coverage {r.get('coverage')}")
            if mode == "layer":
                for layer, ms in per_layer.items():
                    check(f"{ms:7.2f} {100 * ms / true_ms:5.1f} %  {layer}" in buf.getvalue(), f"layer: {layer}'s line")
            # the raw sums with no correction: every span keeps its event
            buf0 = io.StringIO()
            r0 = summarize(path, out=buf0)
            n_spans = {"group": sum(len(g) for g in cards), "layer": 10, "sample": sum(len(g) for g in cards),
                       "graph": 3}[mode]
            check(near(r0["sum_ms"], true_ms + c_true * n_spans / 1000), f"{mode}: raw sum {r0['sum_ms']:.4f} ms")
            # --how all takes the warm-up's slow evaluations in
            if mode != "group":
                rall = summarize(path, event_us=c_true, how="all", out=io.StringIO())
                check(rall["used_evals"] == n_tok * len(cards) and rall["sum_ms"] > r["sum_ms"] * 1.01,
                      f"{mode}: --how all")
        # the original report's first line for a group-mode trace
        first = io.StringIO()
        summarize(os.path.join(tmp, "prof-group.csv"), out=first)
        check(first.getvalue().startswith(f"{n_tok * 3 + 6} device evals: {n_tok * 3} decode-sized (< 60.0 ms), 6 larger;"
                                          f" ~{n_tok} decode steps"), "group: the original first line")
        # a tolerance the sum misses
        r = summarize(os.path.join(tmp, "prof-graph.csv"), event_us=c_true, wall_ms=wall_ref, tol_pct=0.1,
                      out=io.StringIO())
        check(r.get("check") == "fail", "graph: a 0.1 % tolerance should fail")
    print(f"prof_summary self-test: {checks} checks, {len(fails)} failed")
    for f in fails:
        print("  FAILED:", f)
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("max_eval_ms", nargs="?", type=float, default=60.0)
    ap.add_argument("--event-us", type=float, default=0.0, help="profiler cost per timed span, µs (C140)")
    ap.add_argument("--wall-ms", type=float, help="unprofiled time per token, ms, measured separately")
    ap.add_argument("--how", choices=("auto", "replay", "all"), default="auto")
    ap.add_argument("--tol-pct", type=float, help="with --wall-ms: a check line, the sum within this many percent")
    ap.add_argument("--top", type=int, default=30, help="launch groups listed")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.verbose) else 1)
    if not a.csv:
        ap.error("CSV is required")
    summarize(a.csv, a.max_eval_ms, a.event_us, a.wall_ms, a.how, a.tol_pct, a.top)


if __name__ == "__main__":
    main()
