#!/usr/bin/env python3
"""compare.py: R191 (GGML_CUDA_KEEPALIVE_EAGER=0) against R192 (the v4 preset's =1), the same deep-lane pair at 4,096
tokens on the same build, file and draft (Q1562).

  python3 -I compare.py --a-passes R192/passes.tsv --b-passes R191/passes.tsv \
      --a-tasks R192/tasks.tsv --b-tasks R191/tasks.tsv \
      --a-requests R192/requests.csv --b-requests R191/requests.csv \
      --a-gp R192/gp-g1.csv.gz --b-gp R191/gp-k1.csv[.gz] [--a-name EAGER=1 --b-name EAGER=0] \
      [--a-load g1 --b-load k1] [--gap-ms 3000] [--slow-ms 200] [--tsv compare.tsv] > compare.md

passes.tsv and tasks.tsv are graph_passes.py's and server_graph_log.py's tables (R192's run/), requests.csv the deep
lane's, gp the graph-mode profile (GGML_CUDA_PROFILE_MODE=graph; .gz read as is).

Prints:
- the main passes that are not replayed on every card (the prompt's last 4 tokens, the first and second verify pass,
  the request's last pass): GPU ms summed over the nine cards in each run's first request and repeat, and the first
  request's excess over its repeat in each run;
- the first verify pass card by card in both runs (from the profile);
- the keep-alive's counts per request (evaluations kernel by kernel, acted on, block stops, new sizes outside a graph
  and their stops; at EAGER=0 only the first is counted);
- per request of the timed load: prompt and generation time in both runs, and the first request against its repeat;
- each evaluation against the same one in the other run: per piece of the profile (the warm-up request, the prime,
  the first request, the repeat), each card's evaluations paired in order; the longest kernel-by-kernel evaluation,
  the largest increase, and every pair that grew by more than --slow-ms. A launch that waits for a running block
  (FN85: about 1 s) is inside its evaluation's event pair, so it shows here.
Standard library only.
"""
import argparse
import csv
import gzip
import sys


def opener(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def f2(v):
    return "" if v is None else f"{v:.2f}"


def sgn(v):
    return "" if v is None else f"{v:+.2f}"


def gp_pieces(path, gap_ms):
    rows = []
    with opener(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"d": int(r["device"]), "eval": int(r["eval"]), "t": float(r["host_ms"]),
                             "graph": int(r["graph"]), "how": r["how"], "us": float(r["eval_us"])})
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda e: e["t"])  # the profiler writes an evaluation at its graph's next evaluation
    pieces, cur, tmax = [], [], None
    for e in rows:
        if cur and e["t"] > tmax + gap_ms:
            pieces.append(cur)
            cur, tmax = [], None
        cur.append(e)
        tmax = e["t"] if tmax is None else max(tmax, e["t"])
    if cur:
        pieces.append(cur)
    return pieces


def piece_name(i, n):
    # the deep lane's timed load: the warm-up request, the prime, the first request, the repeat
    names = ["warm-up request (depth 0)", "prime (4,032 tokens)", "first request", "repeat"]
    return names[i] if n == 4 and i < 4 else f"piece {i}"


def main():
    ap = argparse.ArgumentParser()
    for s in ("a", "b"):
        ap.add_argument(f"--{s}-passes", required=True)
        ap.add_argument(f"--{s}-tasks", required=True)
        ap.add_argument(f"--{s}-requests", required=True)
        ap.add_argument(f"--{s}-gp", required=True)
    ap.add_argument("--a-name", default="EAGER=1")
    ap.add_argument("--b-name", default="EAGER=0")
    ap.add_argument("--a-load", default="g1")
    ap.add_argument("--b-load", default="k1")
    ap.add_argument("--first", default="d4096p1")
    ap.add_argument("--repeat", default="d4096p2")
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--slow-ms", type=float, default=200.0)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    runs = [("a", a.a_name, a.a_load), ("b", a.b_name, a.b_load)]
    P, F, R = a.first, a.repeat, None
    out = []
    w = out.append

    # --- the passes that are not replayed on every card
    passes = {s: read_tsv(getattr(a, f"{s}_passes")) for s, _, _ in runs}

    def special(rows):
        res = {}
        for r in rows:
            for req in (P, F):
                how = r.get(f"how_{req}", "")
                if not how:
                    continue
                key = None
                if r["pass"] == "0":
                    key = "0: the prompt's last 4 tokens"
                elif r["pass"] == "1":
                    key = "1: the first verify pass"
                elif r["pass"] == "2":
                    key = "2: the second verify pass"
                elif r.get("tag", "").startswith(req + " last") or r.get("tag", "") == req + " last":
                    key = "last pass"
                if key is None and set(how) - {"R"}:
                    key = f"{r['pass']}: other, not replayed"
                if key == "last pass" and set(how) == {"R"}:
                    # the profiler writes an evaluation at its graph's next evaluation, so the load's very last pass
                    # has no line; the last pass written is the one before it
                    res.setdefault(key, {})[req] = ("not written", None, str(int(r["pass"]) + 1))
                elif key:
                    res.setdefault(key, {})[req] = (how, fnum(r.get(f"gpu_ms_{req}")), r["pass"])
        return res

    sp = {s: special(passes[s]) for s, _, _ in runs}
    keys = []
    for s, _, _ in runs:
        for k in sp[s]:
            if k not in keys:
                keys.append(k)
    order = ["0: the prompt's last 4 tokens", "1: the first verify pass", "2: the second verify pass", "last pass"]
    keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
    w(f"# R191 against R192: the same pair, {a.b_name} against {a.a_name}\n")
    w("## The passes that are not replayed on every card\n")
    w("How on cards 0 to 8 (E kernel by kernel, C captured, R replayed, - not written) and GPU ms summed over the cards"
      " written. \"First − repeat\" is the first request's excess over its repeat in the same run.\n")
    hdr = "| pass |"
    sep = "|---|"
    for s, name, _ in runs:
        hdr += f" {name} first | GPU ms | {name} repeat | GPU ms | first − repeat |"
        sep += "---|---:|---|---:|---:|"
    w(hdr)
    w(sep)
    tsv = []
    for k in keys:
        line = f"| {k} |"
        rec = {"pass": k}
        for s, name, _ in runs:
            p = sp[s].get(k, {}).get(P)
            q = sp[s].get(k, {}).get(F)
            pm = p[1] if p else None
            qm = q[1] if q else None
            d = pm - qm if pm is not None and qm is not None else None
            line += (f" {p[0] if p else ''} {('(' + p[2] + ')') if p and k.startswith('last') else ''} | {f2(pm)} |"
                     f" {q[0] if q else ''} {('(' + q[2] + ')') if q and k.startswith('last') else ''} | {f2(qm)} |"
                     f" {sgn(d)} |")
            rec.update({f"{s}_first_how": p[0] if p else "", f"{s}_first_ms": f2(pm),
                        f"{s}_repeat_how": q[0] if q else "", f"{s}_repeat_ms": f2(qm), f"{s}_excess_ms": f2(d)})
        w(line)
        tsv.append(rec)
    # the replayed passes: medians of both runs' requests, as a check that nothing else moved
    w("")
    w("Replayed passes (every card R), the median GPU ms over the nine cards per pass:\n")
    w("| run | first request | repeat | passes (first, repeat) |")
    w("|---|---:|---:|---|")
    for s, name, _ in runs:
        vals = {P: [], F: []}
        for r in passes[s]:
            for req in (P, F):
                how = r.get(f"how_{req}", "")
                v = fnum(r.get(f"gpu_ms_{req}"))
                if how and set(how) == {"R"} and v is not None:
                    vals[req].append(v)
        med = {k: (sorted(v)[len(v) // 2] if v else None) for k, v in vals.items()}
        w(f"| {name} | {f2(med[P])} | {f2(med[F])} | {len(vals[P])}, {len(vals[F])} |")

    # --- the first verify pass card by card, from the profiles
    pieces = {s: gp_pieces(getattr(a, f"{s}_gp"), a.gap_ms) for s, _, _ in runs}
    w("\n## The first verify pass and the prompt's last 4 tokens, card by card\n")
    w("GPU ms of each card's evaluation (how in parentheses): the main model's first and second evaluation on each card"
      " in the request (card 8: its first and second evaluations that are not the draft's are found by graph_passes.py;"
      " here card 8 is shown as the pass total minus cards 0 to 7).\n")
    w("| pass | run | request | " + " | ".join(f"card {d}" for d in range(8)) + " | card 8 | sum |")
    w("|---|---|---|" + "---|" * 10)
    for which, j in (("prompt's last 4", 0), ("first verify", 1)):
        for s, name, _ in runs:
            ps = pieces[s]
            for req_i, req in ((2, P), (3, F)):
                if len(ps) <= req_i:
                    continue
                cells = []
                tot = 0.0
                for d in range(8):
                    ev = [e for e in ps[req_i] if e["d"] == d]
                    if len(ev) > j:
                        e = ev[j]
                        cells.append(f"{e['us'] / 1000:.2f} ({e['how'][0].upper()})")
                        tot += e["us"] / 1000
                    else:
                        cells.append("")
                pk = "0: the prompt's last 4 tokens" if j == 0 else "1: the first verify pass"
                full = sp[s].get(pk, {}).get(req)
                c8 = full[1] - tot if full and full[1] is not None else None
                w(f"| {which} | {name} | {'first' if req == P else 'repeat'} | " + " | ".join(cells)
                  + f" | {f2(c8)} | {f2(full[1]) if full else ''} |")

    # --- keep-alive counts
    w("\n## The keep-alive's counts per request\n")
    w("From the line the server prints after each reply (the difference to the reply before). At EAGER=0 the switch"
      " counts the evaluations that run kernel by kernel and acts on none: \"acted on\" and the stops stay 0, and"
      " shapes new to a card are not tracked.\n")
    w("| run | request | gen tokens | kernel by kernel | acted on (new shapes) | block stops | new sizes outside a graph |"
      " their stops | graphs reused (llama) | debug: warm-up done | warm-up reset | update failed |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, name, _ in runs:
        for r in read_tsv(getattr(a, f"{s}_tasks")):
            if not r.get("request"):
                continue
            w(f"| {name} | {r['request']} | {r['gen_n']} | {r['ka_eager']} | {r['ka_new_shapes']} | {r['ka_stops']} |"
              f" {r['ka_new_sizes_outside']} | {r['ka_outside_stops']} | {r['graphs_reused']} | {r['dbg_warmup_done']} |"
              f" {r['dbg_warmup_reset']} | {r['dbg_update_failed']} |")

    # --- request times
    w("\n## Request times in the timed load (ms)\n")
    reqs = {}
    for s, name, load in runs:
        reqs[s] = [r for r in read_csv(getattr(a, f"{s}_requests")) if r.get("load") == load]
    w("| request | " + " | ".join(f"{name} prompt ms | {name} gen ms" for _, name, _ in runs)
      + " | prompt: " + f"{runs[1][1]} − {runs[0][1]}" + " | gen: " + f"{runs[1][1]} − {runs[0][1]}" + " | reply md5 the same |")
    w("|---|" + "---:|---:|" * 2 + "---:|---:|---|")
    n = max(len(v) for v in reqs.values())
    gen = {s: {} for s, _, _ in runs}
    for i in range(n):
        ra = reqs["a"][i] if i < len(reqs["a"]) else None
        rb = reqs["b"][i] if i < len(reqs["b"]) else None
        label = (ra or rb)["kind"] + " " + (ra or rb)["pass"] + " at " + (ra or rb)["depth"]
        pa, ga = (fnum(ra["prompt_ms"]), fnum(ra["gen_ms"])) if ra else (None, None)
        pb, gb = (fnum(rb["prompt_ms"]), fnum(rb["gen_ms"])) if rb else (None, None)
        same = "yes" if ra and rb and ra["text_md5"] == rb["text_md5"] else "no"
        w(f"| {label} | {f2(pa)} | {f2(ga)} | {f2(pb)} | {f2(gb)} |"
          f" {sgn(pb - pa) if pa is not None and pb is not None else ''} |"
          f" {sgn(gb - ga) if ga is not None and gb is not None else ''} | {same} |")
        for s, r in (("a", ra), ("b", rb)):
            if r and r["kind"] == "decode":
                gen[s][r["pass"]] = (fnum(r["prompt_ms"]), fnum(r["gen_ms"]), r["text_md5"],
                                     r.get("draft_n"), r.get("draft_accepted"))
    w("\nThe first request against its repeat in each run:\n")
    w("| run | first gen ms | repeat gen ms | generation: first − repeat | first prompt ms | repeat prompt ms |"
      " drafted, accepted (first; repeat) | same reply |")
    w("|---|---:|---:|---:|---:|---:|---|---|")
    for s, name, _ in runs:
        f1, f2_ = gen[s].get("1"), gen[s].get("2")
        if not (f1 and f2_):
            continue
        w(f"| {name} | {f2(f1[1])} | {f2(f2_[1])} | {sgn(f1[1] - f2_[1])} | {f2(f1[0])} | {f2(f2_[0])} |"
          f" {f1[3]}, {f1[4]}; {f2_[3]}, {f2_[4]} | {'yes' if f1[2] == f2_[2] else 'no'} |")

    # --- the long evaluations (FN85's wait would be here)
    w("\n## Each evaluation against the same one in the other run (a wait of about 1 s for a running block would show"
      " here)\n")
    w(f"Per piece of the graph-mode profile, each card's evaluations are paired in order with the other run's ({runs[1][1]}"
      f" against {runs[0][1]}; the same requests, so the same evaluations). A launch that waits for a running block is"
      f" inside its evaluation's event pair. Listed: the evaluations of each run, the longest kernel-by-kernel one, and"
      f" the largest increase of any evaluation; every pair that grew by more than {a.slow_ms:.0f} ms is listed below.\n")
    w(f"| piece | evaluations ({runs[0][1]}; {runs[1][1]}) | kernel by kernel | longest kernel-by-kernel ms"
      f" ({runs[0][1]}; {runs[1][1]}) | largest increase ms | where (card, how, order on the card) |"
      f" pairs over +{a.slow_ms:.0f} ms |")
    w("|---|---|---|---|---:|---|---:|")
    big = []
    pa_, pb_ = pieces["a"], pieces["b"]
    for i in range(max(len(pa_), len(pb_))):
        A = pa_[i] if i < len(pa_) else []
        B = pb_[i] if i < len(pb_) else []
        name = piece_name(i, len(pb_))
        ea = [e for e in A if e["how"] == "eager"]
        eb = [e for e in B if e["how"] == "eager"]
        la = max((e["us"] for e in ea), default=0) / 1000
        lb = max((e["us"] for e in eb), default=0) / 1000
        best, where, over = None, "", 0
        for d in sorted({e["d"] for e in A} | {e["d"] for e in B}):
            xa = [e for e in A if e["d"] == d]
            xb = [e for e in B if e["d"] == d]
            for k in range(min(len(xa), len(xb))):
                inc = (xb[k]["us"] - xa[k]["us"]) / 1000
                if best is None or inc > best:
                    best = inc
                    where = f"card {d}, {xa[k]['how']}/{xb[k]['how']}, {k}"
                if inc > a.slow_ms:
                    over += 1
                    big.append((name, d, k, xa[k], xb[k], inc))
        w(f"| {name} | {len(A)}; {len(B)} | {len(ea)}; {len(eb)} | {la:.2f}; {lb:.2f} | {sgn(best)} | {where} | {over} |")
    if big:
        w(f"\nEvery pair that grew by more than {a.slow_ms:.0f} ms:\n")
        w(f"| piece | card | order on the card | how ({runs[0][1]}/{runs[1][1]}) | {runs[0][1]} ms | {runs[1][1]} ms |"
          " increase |")
        w("|---|---:|---:|---|---:|---:|---:|")
        for name, d, k, xa, xb, inc in big:
            w(f"| {name} | {d} | {k} | {xa['how']}/{xb['how']} | {xa['us'] / 1000:.2f} | {xb['us'] / 1000:.2f} |"
              f" {sgn(inc)} |")
    print("\n".join(out))
    if a.tsv:
        cols = list(tsv[0].keys()) if tsv else []
        with open(a.tsv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            wr.writeheader()
            wr.writerows(tsv)


if __name__ == "__main__":
    sys.exit(main())
