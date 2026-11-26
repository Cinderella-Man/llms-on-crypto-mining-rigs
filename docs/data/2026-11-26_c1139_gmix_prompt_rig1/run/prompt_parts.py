#!/usr/bin/env python3
"""prompt_parts.py: the prompt reads of a deep-lane load taken apart, two model files compared (C1139, Q1872).

  python3 -I prompt_parts.py --kpass-parts <2026-11-04_kpass_profile/run/kpass_parts.py> \
      --group pmix=gp-pg.csv.gz,requests-pg.csv --group gmix=gp-gg.csv.gz,requests-gg.csv \
      [--host pmix=hp-ph.csv,requests-ph.csv --host gmix=...] [--plain pmix=requests-pu1.csv --plain gmix=...] \
      --base pmix --other gmix [--kinds prime] [--event-us 7.6] [--gap-ms 3000] [--tsv parts.tsv] > prompt.md

The deep lane (bench/deep_lane) pauses before every request (PAUSE), so a profiler file cuts at its idle gaps (more
than --gap-ms after the latest time seen) into one piece per request, after the server's own load-time evaluations.
The last N pieces are the N requests of the load's requests CSV, in order; the pieces kept are the requests of
--kinds (default: the primes, the deep lane's prompt reads with no output). Every piece's evaluation count is printed
beside its request, so the match can be checked: a prime of P tokens read in micro-batches of U has about P/U main
evaluations per card, a generation request many more.

--group: a GGML_CUDA_PROFILE file in group mode (device, eval, host_ms, group, label, us: every launch group of every
evaluation timed, the stream synchronized after each evaluation). An evaluation with a launch group whose label holds
"mtp_" is the MTP draft's (its catch-up over the prompt, on the last card). Each launch group's time less --event-us
(the profiler's own cost per span, R125's 7.6) is counted:
  - by card (the main model's evaluations and the draft's);
  - by part: R41's parts (kpass_parts.py's part_of, first match wins), the draft's evaluations as their own part;
  - by product: the label without types and shapes (products.py's product_of), with the types each file holds;
  - the routed experts by projection and weight type: every plain "MUL_MAT_ID ffn_<gate|up|down>_exps:<type>:<shape>"
    launch group (one layer's product for one micro-batch), its count, ms and ms a launch, against Q2_0's ms a launch
    of the same projection in the same file (the gate and up matrices are all 2,560 x 640 per expert, so a launch's
    time compares per weight), and the excess over Q2_0 (launches x the difference a launch).
--host: a GGML_CUDA_PROFILE_HOST file (device, eval, host_enter_ms, host_exit_ms, gpu_us, queued; nothing
synchronized): per card, the host time inside its evaluations (exit - enter) and their GPU time, and the piece's
span.
--plain: a load with no profiler: each kept request's prompt_n and prompt_ms (the server's timings).
Pricing (when --plain has both files and --group both): f = the base file's plain prompt time / its group-profile GPU
time (how much of the cards' summed time the pipelined read takes), then for the other file: f x its GPU time is what
it would take if it pipelined as well, and the rest of its plain time is the pipeline loss. Each GPU-time difference
is priced at f (est.).
--tsv writes every value in long form: section, request, key, file, count, ms.
Standard library only."""
import argparse
import collections
import csv
import gzip
import importlib.util
import re
import sys


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def opener(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def read_requests(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def req_name(r):
    return f"{r['kind']} d{r['depth']}" + (f" p{r['pass']}" if r["kind"] in ("decode", "sample") else "")


def cut_pieces(rows_iter, gap_ms):
    """rows_iter yields (t_start, t_end, row); -> [[row, ...], ...] cut where t_start passes the latest end + gap."""
    pieces, last = [], None
    for t0, t1, r in rows_iter:
        if last is None or t0 - last > gap_ms:
            pieces.append([])
        pieces[-1].append(r)
        last = t1 if last is None else max(last, t1)
    return pieces


def match(pieces, reqs, main_of):
    """-> [(request, piece)] for the last len(reqs) pieces; raises when there are fewer pieces than requests."""
    if len(pieces) < len(reqs):
        raise SystemExit(f"{len(pieces)} pieces for {len(reqs)} requests")
    return list(zip(reqs, pieces[len(pieces) - len(reqs):])), [main_of(p) for p in pieces]


def product_of(label):
    head, _, rest = label.partition(" ")
    names, types = [], []
    for tok in rest.split():
        m = re.match(r"^([^:]+):([^:]+):(\S+)$", tok)
        if m:
            names.append(m.group(1))
            types.append(m.group(2))
        else:
            names.append(tok)
    return head + " " + " ".join(names), types


ROUTED = re.compile(r"^MUL_MAT_ID ffn_(gate|up|down)_exps:([a-z0-9_]+):(\d+x\d+)$")


def group_piece(rows, part_of, event_us):
    """rows: group-profile rows of one request -> dict of sums."""
    ev = collections.OrderedDict()
    for r in rows:
        ev.setdefault((r["d"], r["e"]), []).append(r)
    s = {"card": collections.defaultdict(lambda: [0, 0.0, 0, 0.0]),  # main evals, main ms, draft evals, draft ms
         "part": collections.Counter(), "product": collections.defaultdict(lambda: [0.0, 0, set()]),
         "routed": collections.defaultdict(lambda: [0, 0.0]), "evals": 0, "main0": 0}
    for (d, e), rs in ev.items():
        draft = any("mtp_" in r["label"] for r in rs)
        c = s["card"][d]
        ms = sum(max(r["us"] - event_us, 0.0) for r in rs) / 1000.0
        if draft:
            c[2] += 1
            c[3] += ms
        else:
            c[0] += 1
            c[1] += ms
            if d == 0:
                s["main0"] += 1
        for r in rs:
            m = max(r["us"] - event_us, 0.0) / 1000.0
            s["part"]["MTP draft (its catch-up over the prompt)" if draft else part_of(r["label"])] += m
            if not draft:
                p, types = product_of(r["label"])
                x = s["product"][p]
                x[0] += m
                x[1] += 1
                x[2].update(types)
                mm = ROUTED.match(r["label"])
                if mm:
                    y = s["routed"][(mm.group(1), mm.group(2))]
                    y[0] += 1
                    y[1] += m
    s["evals"] = len(ev)
    return s


def read_group(path):
    def it():
        with opener(path) as f:
            for r in csv.DictReader(f):
                try:
                    t = float(r["host_ms"])
                    x = {"d": int(r["device"]), "e": int(r["eval"]), "us": float(r["us"]), "label": r["label"] or ""}
                    int(r["group"])
                except (KeyError, TypeError, ValueError):
                    continue  # a line cut at the server's exit
                yield t, t, x
    return cut_pieces(it(), A.gap_ms)


def read_host(path):
    rows = []
    with opener(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["host_enter_ms"]), float(r["host_exit_ms"]),
                             {"d": int(r["device"]), "e": int(r["eval"]), "t0": float(r["host_enter_ms"]),
                              "t1": float(r["host_exit_ms"]), "gpu": float(r["gpu_us"]) / 1000.0}))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    return cut_pieces(iter(rows), A.gap_ms)


def host_piece(rows):
    c = collections.defaultdict(lambda: [0, 0.0, 0.0])  # evals, host ms, GPU ms
    for r in rows:
        x = c[r["d"]]
        x[0] += 1
        x[1] += r["t1"] - r["t0"]
        x[2] += r["gpu"]
    span = max(r["t1"] for r in rows) - min(r["t0"] for r in rows)
    return c, span


def kv(s):
    name, _, rest = s.partition("=")
    return name, rest.split(",")


ap = argparse.ArgumentParser()
ap.add_argument("--kpass-parts", required=True)
ap.add_argument("--group", action="append", default=[])
ap.add_argument("--host", action="append", default=[])
ap.add_argument("--plain", action="append", default=[])
ap.add_argument("--base", required=True)
ap.add_argument("--other", required=True)
ap.add_argument("--kinds", default="prime")
ap.add_argument("--event-us", type=float, default=7.6)
ap.add_argument("--gap-ms", type=float, default=3000.0)
ap.add_argument("--min-ms", type=float, default=0.05, help="products under this difference (ms a request) summed in one row")
ap.add_argument("--tsv")
A = ap.parse_args()
KP = load_mod(A.kpass_parts, "kpass_parts")
kinds = set(A.kinds.split(","))
F = [A.base, A.other]
long = []
P = print


def kept(reqs):
    return [r for r in reqs if r["kind"] in kinds]


G, H, PL = {}, {}, {}
P(f"# Prompt reads by part, product and weight type: `{A.base}` against `{A.other}`\n")
P(f"Kept requests: {', '.join(sorted(kinds))}. The pieces of each profiler file, matched to the load's requests (main"
  " evaluations on card 0 per piece):\n")
for spec in A.group:
    name, (trace, reqf) = kv(spec)
    reqs = read_requests(reqf)
    pieces = read_group(trace)
    pairs, mains = match(pieces, reqs, lambda p: len({r["e"] for r in p if r["d"] == 0}))
    P(f"- group `{name}` ({trace.split('/')[-1]}): {len(pieces)} pieces, {len(reqs)} requests; "
      + "; ".join(f"{req_name(q)}: {len({r['e'] for r in p if r['d'] == 0})}" for q, p in pairs))
    G[name] = {req_name(q): (q, group_piece(p, KP.part_of, A.event_us)) for q, p in pairs if q["kind"] in kinds}
for spec in A.host:
    name, (trace, reqf) = kv(spec)
    reqs = read_requests(reqf)
    pieces = read_host(trace)
    pairs, _ = match(pieces, reqs, len)
    P(f"- host `{name}` ({trace.split('/')[-1]}): {len(pieces)} pieces, {len(reqs)} requests; "
      + "; ".join(f"{req_name(q)}: {len({r['e'] for r in p if r['d'] == 0})}" for q, p in pairs))
    H[name] = {req_name(q): (q,) + host_piece(p) for q, p in pairs if q["kind"] in kinds}
for spec in A.plain:
    name, (reqf,) = kv(spec)
    PL[name] = {req_name(q): q for q in read_requests(reqf) if q["kind"] in kinds}
P()

names = []
for d in (G, H, PL):
    for f in d.values():
        for k in f:
            if k not in names:
                names.append(k)
names.sort(key=lambda k: int(k.split()[1][1:]))

for rq in names:
    P(f"## Request {rq}\n")
    # prompt sizes and plain speeds
    P("| File | load | prompt tokens read | prompt ms | tok/s |")
    P("|---|---|---:|---:|---:|")
    for src, d in (("plain", PL), ("group", G), ("host", H)):
        for f in F:
            if f in d and rq in d[f]:
                q = d[f][rq] if src == "plain" else d[f][rq][0]
                n, ms = int(q["prompt_n"]), float(q["prompt_ms"])
                P(f"| `{f}` | {src} ({q['load']}) | {n:,} | {ms:,.1f} | {n * 1000 / ms:.1f} |")
                long.append(("prompt", rq, src, f, n, f"{ms:.3f}"))
    P()
    if all(f in G and rq in G[f] for f in F):
        gb, go = G[A.base][rq][1], G[A.other][rq][1]
        ntok = int(G[A.base][rq][0]["prompt_n"])
        P(f"### Group profile: GPU time by part (ms for the request, all cards)\n")
        P(f"Main evaluations on card 0: `{A.base}` {gb['main0']}, `{A.other}` {go['main0']}.\n")
        long += [("evals", rq, "card0_main", A.base, gb["main0"], ""), ("evals", rq, "card0_main", A.other, go["main0"], "")]
        P(f"| Part | `{A.base}` | `{A.other}` | difference | share of the difference |")
        P("|---|---:|---:|---:|---:|")
        tb, to = sum(gb["part"].values()), sum(go["part"].values())
        for k in KP.NAMES + ["MTP draft (its catch-up over the prompt)"]:
            b, o = gb["part"].get(k, 0.0), go["part"].get(k, 0.0)
            if b == 0 and o == 0:
                continue
            P(f"| {k} | {b:,.1f} | {o:,.1f} | {o - b:+,.1f} | {100 * (o - b) / (to - tb):.1f} % |")
            long += [("part", rq, k, A.base, "", f"{b:.3f}"), ("part", rq, k, A.other, "", f"{o:.3f}")]
        P(f"| **all** | **{tb:,.1f}** | **{to:,.1f}** | **{to - tb:+,.1f}** | 100 % |\n")
        P("### Group profile: GPU time by card (ms; main model's evaluations + the draft's)\n")
        P(f"| Card | `{A.base}` main | `{A.base}` draft | `{A.other}` main | `{A.other}` draft | difference |")
        P("|---:|---:|---:|---:|---:|---:|")
        for d in sorted(set(gb["card"]) | set(go["card"])):
            b, o = gb["card"][d], go["card"][d]
            P(f"| {d} | {b[1]:,.1f} ({b[0]}) | {b[3]:,.1f} ({b[2]}) | {o[1]:,.1f} ({o[0]}) | {o[3]:,.1f} ({o[2]}) | "
              f"{o[1] + o[3] - b[1] - b[3]:+,.1f} |")
            long += [("card", rq, str(d), A.base, b[0], f"{b[1]:.3f}"), ("card_draft", rq, str(d), A.base, b[2], f"{b[3]:.3f}"),
                     ("card", rq, str(d), A.other, o[0], f"{o[1]:.3f}"), ("card_draft", rq, str(d), A.other, o[2], f"{o[3]:.3f}")]
        P("\n(evaluations in parentheses)\n")
        P("### Routed experts by projection and weight type (plain launch groups: one layer, one micro-batch)\n")
        P("| Projection | type | file | launches | ms | ms a launch | x Q2_0 (same file) | excess over Q2_0, ms |")
        P("|---|---|---|---:|---:|---:|---:|---:|")
        excess = collections.Counter()
        for f, s in ((A.base, gb), (A.other, go)):
            for proj in ("gate", "up", "down"):
                ref = s["routed"].get((proj, "q2_0"))
                ref_ms = ref[1] / ref[0] if ref and ref[0] else None
                for (pj, ty), (n, ms) in sorted(s["routed"].items()):
                    if pj != proj:
                        continue
                    per = ms / n
                    ratio = f"{per / ref_ms:.2f}" if ref_ms else "-"
                    ex = n * (per - ref_ms) if ref_ms else 0.0
                    if f == A.other:
                        excess[ty] += ex
                    P(f"| {proj} | {ty} | `{f}` | {n} | {ms:,.1f} | {per:.3f} | {ratio} | {ex:+,.1f} |")
                    long.append(("routed", rq, f"{proj}:{ty}", f, n, f"{ms:.3f}"))
        P()
        P("### Products by difference (the label without types and shapes, all cards)\n")
        P(f"| Product | `{A.base}` types | `{A.other}` types | `{A.base}` launches | `{A.other}` launches | `{A.base}` ms | "
          f"`{A.other}` ms | difference |")
        P("|---|---|---|---:|---:|---:|---:|---:|")
        prods = sorted(set(gb["product"]) | set(go["product"]),
                       key=lambda p: -abs(go["product"].get(p, [0.0])[0] - gb["product"].get(p, [0.0])[0]))
        rb = ro = 0.0
        rn = 0
        for p in prods:
            b = gb["product"].get(p, [0.0, 0, set()])
            o = go["product"].get(p, [0.0, 0, set()])
            long += [("product", rq, p, A.base, b[1], f"{b[0]:.3f}"), ("product", rq, p, A.other, o[1], f"{o[0]:.3f}")]
            if abs(o[0] - b[0]) < A.min_ms * 1:
                rb += b[0]
                ro += o[0]
                rn += 1
                continue
            P(f"| `{p[:110]}` | {', '.join(sorted(b[2])) or '-'} | {', '.join(sorted(o[2])) or '-'} | {b[1]} | {o[1]} | "
              f"{b[0]:,.1f} | {o[0]:,.1f} | {o[0] - b[0]:+,.1f} |")
        P(f"| the other {rn} products (each under {A.min_ms} ms apart) | | | | | {rb:,.1f} | {ro:,.1f} | {ro - rb:+,.1f} |")
        mb = sum(v[1] for v in gb["card"].values())
        mo = sum(v[1] for v in go["card"].values())
        P(f"| **the main model's evaluations** | | | | | **{mb:,.1f}** | **{mo:,.1f}** | **{mo - mb:+,.1f}** |\n")
        # pricing
        if all(f in PL and rq in PL[f] for f in F):
            pb, po = float(PL[A.base][rq]["prompt_ms"]), float(PL[A.other][rq]["prompt_ms"])
            fct = pb / tb
            pipe = po - fct * to
            P("### Pricing (est.)\n")
            P(f"- Plain prompt time: `{A.base}` {pb:,.1f} ms, `{A.other}` {po:,.1f} ms, gap {po - pb:+,.1f} ms "
              f"({100 * (po - pb) / pb:+.1f} %).")
            P(f"- f = {pb:,.1f} / {tb:,.1f} = {fct:.4f} (`{A.base}`'s plain time over its summed GPU time).")
            P(f"- `{A.other}` at f: {fct * to:,.1f} ms; its GPU-time difference priced at f: {fct * (to - tb):+,.1f} ms; "
              f"the rest of its plain time, the pipeline loss: {pipe:+,.1f} ms.")
            for ty, ex in sorted(excess.items(), key=lambda kv: -kv[1]):
                if ty == "q2_0":
                    continue
                P(f"- routed gate/up in {ty}: {ex:+,.1f} ms GPU over Q2_0's rate, est. {fct * ex:+,.1f} ms of plain time.")
                long.append(("price", rq, ty, A.other, "", f"{fct * ex:.3f}"))
            long += [("price", rq, "f", "", "", f"{fct:.5f}"), ("price", rq, "pipeline_loss", A.other, "", f"{pipe:.3f}"),
                     ("price", rq, "gpu_diff_at_f", A.other, "", f"{fct * (to - tb):.3f}")]
            P()
    if all(f in H and rq in H[f] for f in F):
        P("### Host profile: per card, the host time inside its evaluations and their GPU time (ms)\n")
        P(f"| Card | `{A.base}` evals | `{A.base}` host | `{A.base}` GPU | `{A.other}` evals | `{A.other}` host | "
          f"`{A.other}` GPU | host difference |")
        P("|---:|---:|---:|---:|---:|---:|---:|---:|")
        cb, cbs = H[A.base][rq][1], H[A.base][rq][2]
        co, cos = H[A.other][rq][1], H[A.other][rq][2]
        for d in sorted(set(cb) | set(co)):
            b, o = cb[d], co[d]
            P(f"| {d} | {b[0]} | {b[1]:,.1f} | {b[2]:,.1f} | {o[0]} | {o[1]:,.1f} | {o[2]:,.1f} | {o[1] - b[1]:+,.1f} |")
            long += [("host", rq, str(d), A.base, b[0], f"{b[1]:.3f}"), ("host_gpu", rq, str(d), A.base, b[0], f"{b[2]:.3f}"),
                     ("host", rq, str(d), A.other, o[0], f"{o[1]:.3f}"), ("host_gpu", rq, str(d), A.other, o[0], f"{o[2]:.3f}")]
        hb, ho = sum(v[1] for v in cb.values()), sum(v[1] for v in co.values())
        gb2, go2 = sum(v[2] for v in cb.values()), sum(v[2] for v in co.values())
        P(f"| all | | {hb:,.1f} | {gb2:,.1f} | | {ho:,.1f} | {go2:,.1f} | {ho - hb:+,.1f} |")
        P(f"\nThe piece's span: `{A.base}` {cbs:,.1f} ms, `{A.other}` {cos:,.1f} ms.\n")
        long += [("host_span", rq, "", A.base, "", f"{cbs:.3f}"), ("host_span", rq, "", A.other, "", f"{cos:.3f}")]

if A.tsv:
    with open(A.tsv, "w") as f:
        f.write("section\trequest\tkey\tfile\tcount\tms\n")
        for r in long:
            f.write("\t".join(str(x) for x in r) + "\n")
