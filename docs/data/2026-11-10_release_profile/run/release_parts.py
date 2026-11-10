#!/usr/bin/env python3
"""release_parts.py: the parts of a k-token pass from release_prof.sh's group-mode trace, one request per k.

A group-mode GGML_CUDA_PROFILE trace names no graph, and kpass_parts.py finds the border between two requests from the
evaluations' GPU times. Here the border is known instead: a request of N prompt tokens in micro-batches of k evaluates
each card's graph a known number of times, in order. This script cuts each card's evaluations by those counts, drops
the passes that are not a plain k-token pass, and then calls kpass_parts.py's own estimator (parts_per_pass: every
span of every kept evaluation, the profiler's cost subtracted per span, by part of the model) and prof_summary.py's
report on each request.

  python3 -I release_parts.py --kpass-parts <kpass_parts.py> --prof-summary <scripts/prof_summary.py> --event-us 7.6 \
      --tag release --segments 1:340 4:85 5:69:3-67 [--other-regex mtp_] [--wall 1=57.1,4=124.9] --out <dir> <prof.csv>

--segments  k:count[:first-last] per request, in the trace's order: count evaluations per card, of which the passes
            first..last (1-based) are kept; by default 3..count, since the first pass of a request starts the sequence
            (a graph of its own) and the first passes at a new size pay one-off launches. x:count sets count
            evaluations per card aside (the one the server runs while it loads).
--other-regex  evaluations with a launch group whose label matches are not the model's passes (the MTP draft's own
            graphs): they are set aside before the count and reported per request, as ms per kept pass.
Writes <out>/<tag>-k<k>.csv (the kept rows), <tag>-k<k>.txt (prof_summary.py's report), parts.tsv (trace, k, part,
ms_per_pass), per_card.tsv (trace, k, card, ms_per_pass, passes, the largest kept evaluation over the median),
cut.tsv (what each request held and what was kept) and passes.tsv (every pass of every request: when the first and
the last card started it, the last card's and all cards' raw GPU time). Standard library only."""
import argparse, collections, csv, importlib.util, os, re, statistics, sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--kpass-parts", required=True); ap.add_argument("--prof-summary", required=True)
ap.add_argument("--event-us", type=float, default=0.0); ap.add_argument("--tag", required=True)
ap.add_argument("--segments", nargs="+", required=True); ap.add_argument("--other-regex", default="")
ap.add_argument("--wall", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
kp, ps = load(a.kpass_parts, "kpass_parts"), load(a.prof_summary, "prof_summary")
wall = {int(k): float(v) for k, v in (kv.split("=") for kv in a.wall.split(",") if kv)}
segs = []  # (k, count, first, last); k 0: set aside
for s in a.segments:
    f = s.split(":")
    k, n = (0 if f[0] == "x" else int(f[0])), int(f[1])
    first, last = (int(x) for x in f[2].split("-")) if len(f) > 2 else (3, n)
    segs.append((k, n, first, last))
other_rx = re.compile(a.other_regex) if a.other_regex else None
os.makedirs(a.out, exist_ok=True)

with open(a.trace, newline="") as f:
    rd = csv.DictReader(f)
    header, rows = rd.fieldnames, list(rd)
ev = collections.OrderedDict()  # (device, eval) -> {t, us, rows, other}
for r in rows:
    e = ev.setdefault((int(r["device"]), int(r["eval"])), {"t": float(r["host_ms"]), "us": 0.0, "rows": [], "other": False})
    e["rows"].append(r)
    e["us"] += float(r["us"])
    if other_rx and other_rx.search(r["label"]):
        e["other"] = True
devs = sorted({k[0] for k in ev})
model = {d: sorted((e for k, e in ev.items() if k[0] == d and not e["other"]), key=lambda e: e["t"]) for d in devs}
others = sorted((e for e in ev.values() if e["other"]), key=lambda e: e["t"])
want = sum(n for _, n, _, _ in segs)
for d in devs:
    if len(model[d]) != want:
        sys.exit(f"card {d} ran {len(model[d])} evaluations of the model, the segments hold {want}")

parts_rows, card_rows, cut_rows, pass_rows = [], [], [], []
off = 0
for i, (k, n, first, last) in enumerate(segs):
    if k == 0:
        off += n
        continue
    name = f"{a.tag}-k{k}" + ("" if [s[0] for s in segs].count(k) == 1 else f"-r{i + 1}")
    kept = {d: model[d][off + first - 1:off + last] for d in devs}
    t0 = min(model[d][off]["t"] for d in devs)
    t1 = max(model[d][off + n - 1]["t"] for d in devs)
    d0, d9 = devs[0], devs[-1]
    for j in range(n):
        pass_rows.append((name, k, j + 1, int(first - 1 <= j < last), model[d0][off + j]["t"], model[d9][off + j]["t"],
                          model[d9][off + j]["us"] / 1000, sum(model[d][off + j]["us"] for d in devs) / 1000))
    gaps = [b["t"] - x["t"] for x, b in zip(kept[d0], kept[d0][1:])]
    step = statistics.median(gaps) if gaps else float("nan")
    sub = os.path.join(a.out, name + ".csv")
    with open(sub, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for d in devs:
            for e in kept[d]:
                w.writerows(e["rows"])
    with open(os.path.join(a.out, name + ".txt"), "w") as f:
        ps.summarize(sub, max_ms=1e9, event_us=a.event_us, wall_ms=wall.get(k), tol_pct=5, out=f)
    _, parts = kp.parts_per_pass(ps, sub, a.event_us)
    n_kept = last - first + 1
    for p in kp.NAMES:
        parts_rows.append((name, k, p, parts[p]))
    gpu = sum(parts.values())
    parts_rows.append((name, k, "sum of the parts", gpu))
    for d in devs:
        us = [e["us"] - a.event_us * len(e["rows"]) for e in kept[d]]
        raw = [e["us"] for e in kept[d]]
        card_rows.append((name, k, d, statistics.fmean(us) / 1000, n_kept, max(raw) / statistics.median(raw)))
    # the evaluations set aside (the draft's own graphs) inside this request's time, per kept pass
    mine = [e for e in others if t0 <= e["t"] <= (min(model[d][off + n]["t"] for d in devs) if i + 1 < len(segs) else float("inf"))]
    o_ms = sum(e["us"] - a.event_us * len(e["rows"]) for e in mine) / 1000
    dropped = {d: [e["us"] / 1000 for j, e in enumerate(model[d][off:off + n]) if not first - 1 <= j < last] for d in devs}
    cut_rows.append((name, k, n, first, last, n_kept, t0, t1, model[d0][off + first - 1]["t"], kept[d0][-1]["t"], step, len(mine), o_ms,
                     ";".join(f"{d}:" + ",".join(f"{x:.1f}" for x in dropped[d]) for d in devs)))
    print(f"{name}: {n} passes per card, kept {first}-{last} ({n_kept}); cards' GPU time {gpu:.2f} ms a pass, {step:.2f} ms from pass to pass under the profiler"
          + (f", unprofiled {wall[k]:.2f}, hops, host and gaps {wall[k] - gpu:.2f}" if k in wall else "")
          + (f"; set aside: {len(mine)} evaluations, {o_ms:.1f} ms in all" if other_rx else ""))
    off += n

with open(os.path.join(a.out, "parts.tsv"), "w") as f:
    f.write("trace\tk\tpart\tms_per_pass\n")
    for name, k, p, v in parts_rows:
        f.write(f"{name}\t{k}\t{p}\t{v:.2f}\n")
with open(os.path.join(a.out, "per_card.tsv"), "w") as f:
    f.write("trace\tk\tcard\tms_per_pass\tpasses\tmax_over_median\n")
    for name, k, d, v, n_kept, mx in card_rows:
        f.write(f"{name}\t{k}\t{d}\t{v:.2f}\t{n_kept}\t{mx:.2f}\n")
with open(os.path.join(a.out, "passes.tsv"), "w") as f:
    f.write("trace\tk\tpass\tkept\tfirst_card_host_ms\tlast_card_host_ms\tlast_card_raw_ms\tall_cards_raw_ms\n")
    for row in pass_rows:
        f.write("%s\t%d\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.2f\n" % row)
with open(os.path.join(a.out, "cut.tsv"), "w") as f:
    f.write("trace\tk\tpasses\tfirst_kept\tlast_kept\tkept\tfirst_host_ms\tlast_host_ms\tfirst_kept_host_ms\tlast_kept_host_ms\t"
            "profiled_ms_pass_to_pass\tset_aside_evals\tset_aside_ms\t"
            "dropped_raw_ms_per_card\n")
    for row in cut_rows:
        f.write("%s\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\t%d\t%.1f\t%s\n" % row)
