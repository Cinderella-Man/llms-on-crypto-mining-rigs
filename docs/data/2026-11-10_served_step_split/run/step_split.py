#!/usr/bin/env python3
"""step_split.py: one served speculative step of the MTP draft line, part by part, from the fork's
GGML_CUDA_PROFILE_HOST file (S37, Q961).

  python3 -I step_split.py HOST_PROFILE.csv [--load d1off] [--requests long,greedy,smoke] [--gap-ms 2000]
      [--main-devices 0,1,2,3,4,5,6,7,8] [--skip 3] [--tsv steps.tsv] > split.md

The file has one line per graph evaluation and card: device, eval, host_enter_ms, host_exit_ms, gpu_us (the host time
at entry to and exit from the backend's graph_compute; the GPU time between two events around the evaluation's work).
The profile waits at entry for the card's previous evaluation, so an evaluation's work starts at its entry.

Requests: the driver pauses between requests, so a gap of more than --gap-ms without an evaluation cuts the trace;
the last pieces with at least ten main passes are the requests, named by --requests in order.
A main pass is one evaluation on each main card in order. The host queues all of them within a few ms and the cards
pass the residual on through staged copies, so the host's clock says when a main pass was launched and when the next
evaluation began, not when each card worked. The evaluations between two main passes are the draft's: a draft pass
ends with an evaluation on the draft's last card (the highest card seen between main passes), preceded by one on the
card of its token embeddings unless the row cache is on.

A step runs from a main pass's first entry to the next main pass's first entry. Its parts add up to it:
  verify_span   from the main pass's first entry to the first draft evaluation's entry: the nine cards' work
                (verify_gpu, the sum of their GPU times; they work one after another), and verify_rest = the rest:
                the eight crossings, the sampled rows' and hidden rows' way back, the acceptance test, the server's
                work before the draft
  per draft pass p (0 = the catch-up over the accepted tokens, 1 ... = the drafting passes):
    lead_p      from the pass's first entry to its last card's entry: the embedding card's evaluation and the row's
                crossing through the host (0 with one card a pass); for pass 1 also the wait for pass 0's work
    gpu_p       the GPU time on the draft's last card (emb_gpu_p: on the embedding card, inside lead_p)
    turn_p      from the end of that work to the next pass's first entry (for the last pass: to the next main pass):
                the sampled id's way back, the host's sampling, the next batch. Negative for pass 0, which the host
                does not wait for; turn_last is the server's work between the draft and the next verify pass
Generation steps are those after the large passes a request starts with (a main pass with more than 1.5 times the
request's median GPU time is a prompt micro-batch). A typical step has the commonest number of draft passes and a
main pass within 12 % of their median GPU time (this drops a prompt's last small micro-batch, which looks like a step,
and a request's last verify pass where it pays one-off launches); the first --skip steps of a request are left out.
The tables are over typical steps; --tsv gets every generation step.
"""
import argparse
import csv
import statistics
import sys
from collections import Counter


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["host_enter_ms"]), float(r["host_exit_ms"]), int(r["device"]),
                             float(r["gpu_us"]) / 1000.0))
            except (KeyError, TypeError, ValueError):
                continue  # a line cut at the server's exit
    rows.sort()
    return rows


def cut(rows, gap_ms):
    segs, cur, last = [], [], None
    for r in rows:
        if cur and r[0] - last > gap_ms:
            segs.append(cur)
            cur = []
        last = r[1] if not cur else max(last, r[1])
        cur.append(r)
    if cur:
        segs.append(cur)
    return segs


def main_passes(seg, main_devs):
    passes, i, n = [], 0, len(main_devs)
    while i < len(seg):
        if seg[i][2] == main_devs[0] and i + n <= len(seg) and [seg[i + k][2] for k in range(n)] == main_devs:
            passes.append((i, i + n - 1))
            i += n
        else:
            i += 1
    return passes


def steps_of(seg, passes, draft_last):
    out = []
    for p, q in zip(passes, passes[1:]):
        main = seg[p[0]:p[1] + 1]
        between = seg[p[1] + 1:q[0]]
        dps, cur = [], []
        for b in between:
            cur.append(b)
            if b[2] == draft_last:
                dps.append(cur)
                cur = []
        t0, t_next = main[0][0], seg[q[0]][0]
        s = dict(step_ms=t_next - t0, verify_gpu_ms=sum(x[3] for x in main), launch_ms=main[-1][1] - t0,
                 card_gpu=[x[3] for x in main], n_dp=len(dps), stray=len(cur), n_evals=len(between), passes=[])
        s["verify_span_ms"] = (dps[0][0][0] if dps else t_next) - t0
        s["verify_rest_ms"] = s["verify_span_ms"] - s["verify_gpu_ms"]
        for k, dp in enumerate(dps):
            first_enter, last = dp[0][0], dp[-1]
            nxt = dps[k + 1][0][0] if k + 1 < len(dps) else t_next
            s["passes"].append(dict(lead=last[0] - first_enter, gpu=last[3], emb_gpu=sum(x[3] for x in dp[:-1]),
                                    turn=nxt - (last[0] + last[3]), evals=len(dp)))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--load", default="load")
    ap.add_argument("--requests", default="long,greedy,smoke")
    ap.add_argument("--gap-ms", type=float, default=2000.0)
    ap.add_argument("--main-devices", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    main_devs = [int(x) for x in a.main_devices.split(",")]
    names = a.requests.split(",")
    rows = load_rows(a.profile)
    segs = cut(rows, a.gap_ms)
    seg_passes = [main_passes(s, main_devs) for s in segs]
    print(f"{a.load}: {len(rows)} evaluations in {len(segs)} pieces (cut at gaps over {a.gap_ms:.0f} ms): "
          + ", ".join(f"{len(s)} evaluations and {len(p)} main passes" for s, p in zip(segs, seg_passes)))
    req = [(s, p) for s, p in zip(segs, seg_passes) if len(p) >= 10]
    if len(req) != len(names):
        print(f"WARNING: {len(req)} pieces with at least ten main passes for {len(names)} requests; naming the last ones",
              file=sys.stderr)
        print(f"WARNING: {len(req)} pieces with at least ten main passes for {len(names)} requests")
    req = req[-len(names):]
    names = names[-len(req):] if len(req) < len(names) else names
    between_devs = Counter()
    for s, ps in req:
        for p, q in zip(ps, ps[1:]):
            between_devs.update(x[2] for x in s[p[1] + 1:q[0]])
    draft_last = max(between_devs) if between_devs else main_devs[-1]
    print(f"evaluations between main passes, by card: "
          + ", ".join(f"card {d}: {n}" for d, n in sorted(between_devs.items())) + f"; the draft's last card: {draft_last}")
    print()

    all_steps = []
    print("| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |")
    print("|---|---:|---:|---:|---|---:|")
    for name, (s, ps) in zip(names, req):
        st = steps_of(s, ps, draft_last)
        if not st:
            continue
        med = statistics.median(x["verify_gpu_ms"] for x in st)
        g0 = 0
        while g0 < len(st) and st[g0]["verify_gpu_ms"] > 1.5 * med:
            g0 += 1
        gen = st[g0:]
        mode = Counter(x["n_dp"] for x in gen).most_common(1)[0][0]
        mmed = statistics.median(x["verify_gpu_ms"] for x in gen if x["n_dp"] == mode)
        for i, x in enumerate(gen):
            x["request"], x["i"] = name, i
            x["typical"] = int(i >= a.skip and x["n_dp"] == mode and x["stray"] == 0
                               and abs(x["verify_gpu_ms"] / mmed - 1.0) <= 0.12)
        all_steps += gen
        kinds = ", ".join(f"{k}: {n}" for k, n in sorted(Counter(x["n_dp"] for x in gen).items()))
        print(f"| {name} | {len(ps)} | {g0} | {len(gen)} | {kinds} | {sum(x['typical'] for x in gen)} |")
    print()

    if a.tsv:
        npass = max((x["n_dp"] for x in all_steps), default=0)
        with open(a.tsv, "w") as f:
            cols = ["load", "request", "i", "typical", "n_dp", "n_evals", "step_ms", "verify_span_ms", "verify_gpu_ms",
                    "verify_rest_ms", "launch_ms"] + [f"card{d}_gpu_ms" for d in main_devs]
            for k in range(npass):
                cols += [f"lead{k}_ms", f"emb_gpu{k}_ms", f"gpu{k}_ms", f"turn{k}_ms"]
            f.write("\t".join(cols) + "\n")
            for x in all_steps:
                v = [a.load, x["request"], x["i"], x["typical"], x["n_dp"], x["n_evals"]]
                v += [f"{x[c]:.3f}" for c in ["step_ms", "verify_span_ms", "verify_gpu_ms", "verify_rest_ms", "launch_ms"]]
                v += [f"{g:.3f}" for g in x["card_gpu"]]
                for k in range(npass):
                    if k < len(x["passes"]):
                        d = x["passes"][k]
                        v += [f"{d['lead']:.3f}", f"{d['emb_gpu']:.3f}", f"{d['gpu']:.3f}", f"{d['turn']:.3f}"]
                    else:
                        v += ["", "", "", ""]
                f.write("\t".join(str(c) for c in v) + "\n")

    def table(steps, title):
        if not steps:
            return
        n_dp = steps[0]["n_dp"]
        parts = [("step", lambda x: x["step_ms"]),
                 ("verify_span", lambda x: x["verify_span_ms"]),
                 ("  verify_gpu (the nine cards)", lambda x: x["verify_gpu_ms"]),
                 ("  verify_rest (crossings, rows back, acceptance, server)", lambda x: x["verify_rest_ms"])]
        for k in range(n_dp):
            what = "catch-up" if k == 0 else f"draft {k}"
            parts += [(f"pass {k} ({what}): lead", lambda x, k=k: x["passes"][k]["lead"]),
                      (f"pass {k}: gpu", lambda x, k=k: x["passes"][k]["gpu"]),
                      (f"pass {k}: turn" + (" (the server before the next verify)" if k == n_dp - 1 else ""),
                       lambda x, k=k: x["passes"][k]["turn"])]
        parts += [("the draft's span (all passes)", lambda x: x["step_ms"] - x["verify_span_ms"]),
                  ("  the draft's GPU time on its last card", lambda x: sum(d["gpu"] for d in x["passes"])),
                  ("  the leads (crossings)", lambda x: sum(d["lead"] for d in x["passes"])),
                  ("  the turns of the passes before the last", lambda x: sum(d["turn"] for d in x["passes"][:-1])),
                  ("  the draft's GPU time on the embedding card", lambda x: sum(d["emb_gpu"] for d in x["passes"])),
                  ("the host's launch of the main pass (inside verify_span)", lambda x: x["launch_ms"])]
        print(f"**{title}: {len(steps)} typical steps, {n_dp} draft passes a step, "
              f"{statistics.median(sum(d['evals'] for d in x['passes']) for x in steps):.0f} draft evaluations**")
        print()
        print("| part | mean, ms | median | p10 | p90 |")
        print("|---|---:|---:|---:|---:|")
        for label, fn in parts:
            xs = [fn(x) for x in steps]
            print(f"| {label} | {statistics.fmean(xs):.2f} | {statistics.median(xs):.2f} | {pct(xs, 0.1):.2f} | {pct(xs, 0.9):.2f} |")
        print()
        print("verify_gpu per card, median ms: "
              + ", ".join(f"{statistics.median(x['card_gpu'][i] for x in steps):.2f}" for i in range(len(main_devs))))
        print()

    typ = [x for x in all_steps if x["typical"]]
    table(typ, f"{a.load}, all requests")
    for name in names:
        table([x for x in typ if x["request"] == name], f"{a.load}, {name}")


if __name__ == "__main__":
    main()
