#!/usr/bin/env python3
"""r181_replans.py: every re-plan of the compute buffers in a llama-server log written with LLAMA_UBATCH_TRACE=1, each
against the request and the micro-batch it happened in, and grouped by cause.

    python3 -I r181_replans.py OUTDIR server.log

The trace's lines (pascal-p106-v5 602691153; all at the server's default verbosity):
  galloc-trigger: <node|src|leaf> i 'name' (op) ne = [a, b, c, d] needs N bytes in BUF, planned M
                  (or "nodes A -> B" / "leafs A -> B": the graph's own count changed): why the plan does not fit
  sched-replan: #k backends_changed = b [(node|leaf i 'name' DEV0 -> DEV1)] size_changed = s graph_size p -> q
                nodes = n leafs = l splits = sp drain = D plan = P us: the re-plan (drain: the wait for every card)
  ubatch-trace: dec n_tokens = N n_seqs = S n_outputs = O graph = reused|rebuilt [why = W] ... alloc = A ... total = T us
                one micro-batch of the context's own decode path, written after it was submitted
  decode-trace: dec n_tokens = N n_outputs = O ...: one decode call (a prompt call, a token, a re-decode)
A re-plan's trigger and its sched-replan line come before the ubatch-trace line of the micro-batch they belong to. A
re-plan with no ubatch-trace line before the next decode-trace, request or end is outside the decode path's micro-batches
(the pipelined slots' own schedulers, which the release count does not include). The release count
(`graph reallocs = S / D` in print_timing) leaves out re-plans after a change of devices; `counted` marks the others.

Writes OUTDIR/replans.tsv (one row a re-plan), OUTDIR/requests.tsv (one row a request), OUTDIR/causes.tsv (the
re-plans grouped by scheduler, pass and cause, and whether the cards still had work in flight) and OUTDIR/fit.tsv (what
a re-plan with work in flight costs a request's prompt time, by least squares over the requests; an estimate). Standard
library only.
"""
import os
import re
import sys
from collections import OrderedDict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^\s*([0-9]+(?:\.[0-9]+){3,4})\s+[A-Z]\s")
TRIG = re.compile(r"galloc-trigger: (.*)$")
TRIG_T = re.compile(r"(node|src|leaf) ([0-9]+) '([^']*)'(?: of '([^']*)')? \(([^)]*)\) ne = \[([^\]]*)\] needs ([0-9]+) bytes"
                    r"(?: in (\S+), planned ([0-9]+)|, planned none)")
REPLAN = re.compile(r"sched-replan: #([0-9]+) backends_changed = ([01])(?: \((node|leaf) ([0-9]+) '([^']*)' (\S+) -> (\S+)\))? "
                    r"size_changed = ([01]) graph_size ([0-9-]+) -> ([0-9-]+) nodes = ([0-9]+) leafs = ([0-9]+) "
                    r"splits = ([0-9]+) drain = ([0-9-]+) plan = ([0-9-]+) us")
UB = re.compile(r"ubatch-trace: (\w+) n_tokens = ([0-9]+) n_seqs = ([0-9]+) n_outputs = ([0-9]+) graph = (\w+)"
                r"(?: why = (\S+))? pre = ([0-9-]+) build = ([0-9-]+) alloc = ([0-9-]+) inputs = ([0-9-]+) "
                r"submit = ([0-9-]+) card0 = ([0-9-]+) total = ([0-9-]+) us")
DEC = re.compile(r"decode-trace: (\w+) n_tokens = ([0-9]+) n_outputs = ([0-9]+)")
LAUNCH = re.compile(r"launch_slot_: id\s+[0-9]+ \| task ([0-9-]+) \| processing task")
PE = re.compile(r"\| task ([0-9-]+) \| prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
EV = re.compile(r"\| task ([0-9-]+) \|\s+eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GR = re.compile(r"\| task ([0-9-]+) \|\s+graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
NUM = re.compile(r"[0-9]+")


def ts_seconds(line):
    m = TS.match(line)
    if not m:
        return None
    p = [int(x) for x in m.group(1).split(".")]
    us = p[-1] + 1000 * p[-2]
    s = p[-3] + 60 * p[-4] + (3600 * p[-5] if len(p) == 5 else 0)
    return s + us / 1e6


def cause_of(r):
    """A short cause key: the node that changed device, or the tensor that outgrew its place (digits dropped from
    layer-numbered names), or the graph's count."""
    if r["backends_changed"]:
        return f"device change: {r['bc_kind']} '{NUM.sub('#', r['bc_name'])}' {r['bc_from']} -> {r['bc_to']}"
    t = r["triggers"]
    if not t:
        return "no trigger line"
    first = t[0]
    m = TRIG_T.match(first)
    if m:
        kind, _, name, reader, op = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        who = f"'{NUM.sub('#', name)}'" + (f" of '{NUM.sub('#', reader)}'" if reader else "")
        return f"outgrew its place: {kind} {who} ({op})" + ("" if m.group(9) else ", planned none")
    return "count: " + NUM.sub("#", first)


def main():
    outdir, path = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    replans, reqs = [], OrderedDict()
    pending_trig, pending = [], []  # trigger lines, re-plans waiting for their micro-batch
    req, ub_i, last_dec = None, 0, None

    def flush(where):
        for r in pending:
            r["where"] = where
        pending.clear()

    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        t = ts_seconds(line)
        m = TRIG.search(line)
        if m:
            pending_trig.append(m.group(1))
            continue
        m = REPLAN.search(line)
        if m:
            r = dict(k=int(m.group(1)), t=t, task=req, backends_changed=int(m.group(2)), bc_kind=m.group(3) or "",
                     bc_idx=m.group(4) or "", bc_name=m.group(5) or "", bc_from=m.group(6) or "", bc_to=m.group(7) or "",
                     size_changed=int(m.group(8)), gs_from=int(m.group(9)), gs_to=int(m.group(10)), nodes=int(m.group(11)),
                     leafs=int(m.group(12)), splits=int(m.group(13)), drain_us=int(m.group(14)), plan_us=int(m.group(15)),
                     triggers=pending_trig, where="", ub_i="", ub_tokens="", ub_outputs="", ub_why="", ub_alloc_us="",
                     dec_tokens=last_dec[0] if last_dec else "", dec_outputs=last_dec[1] if last_dec else "")
            pending_trig = []
            replans.append(r)
            pending.append(r)
            if req is not None:
                reqs[req]["replans"].append(r)
            continue
        m = UB.search(line)
        if m:
            if req is not None:
                reqs[req]["ubatches"] += 1
            for r in pending:
                r.update(where="micro-batch", ub_i=ub_i, ub_tokens=int(m.group(2)), ub_outputs=int(m.group(4)),
                         ub_why=m.group(6) or m.group(5), ub_alloc_us=int(m.group(9)))
            pending.clear()
            pending_trig = []
            ub_i += 1
            continue
        m = DEC.search(line)
        if m:
            flush("outside a micro-batch")
            last_dec = (int(m.group(2)), int(m.group(3)))
            ub_i = 0
            continue
        m = LAUNCH.search(line)
        if m:
            flush("outside a micro-batch")
            req = m.group(1)
            reqs[req] = dict(task=req, t0=t, prompt_ms="", prompt_tokens="", gen_ms="", gen_tokens="", count_same="",
                             count_diff="", replans=[], ubatches=0)
            continue
        m = PE.search(line)
        if m and m.group(1) in reqs:
            reqs[m.group(1)].update(prompt_ms=float(m.group(2)), prompt_tokens=int(m.group(3)))
            continue
        m = EV.search(line)
        if m and m.group(1) in reqs:
            reqs[m.group(1)].update(gen_ms=float(m.group(2)), gen_tokens=int(m.group(3)))
            continue
        m = GR.search(line)
        if m and m.group(1) in reqs:
            reqs[m.group(1)].update(count_same=int(m.group(2)), count_diff=int(m.group(3)))
            continue
    flush("outside a micro-batch")

    for r in replans:
        r["counted"] = int(not r["backends_changed"])
        r["cause"] = cause_of(r)

    cols = ["k", "t_s", "task", "where", "dec_tokens", "dec_outputs", "ub_i", "ub_tokens", "ub_outputs", "ub_why",
            "backends_changed", "changed", "size_changed", "graph_size", "nodes", "leafs", "splits", "drain_us", "plan_us",
            "ub_alloc_us", "counted", "cause", "triggers"]
    with open(os.path.join(outdir, "replans.tsv"), "w") as f:
        print("\t".join(cols), file=f)
        for r in replans:
            row = [r["k"], f"{r['t']:.3f}" if r["t"] is not None else "", r["task"] or "", r["where"], r["dec_tokens"],
                   r["dec_outputs"], r["ub_i"], r["ub_tokens"], r["ub_outputs"], r["ub_why"], r["backends_changed"],
                   f"{r['bc_kind']} {r['bc_idx']} '{r['bc_name']}' {r['bc_from']} -> {r['bc_to']}" if r["backends_changed"] else "",
                   r["size_changed"], f"{r['gs_from']} -> {r['gs_to']}", r["nodes"], r["leafs"], r["splits"], r["drain_us"],
                   r["plan_us"], r["ub_alloc_us"], r["counted"], r["cause"], " | ".join(r["triggers"])]
            print("\t".join(str(x) for x in row), file=f)

    prev = (0, 0)
    with open(os.path.join(outdir, "requests.tsv"), "w") as f:
        print("req\ttask\tprompt_tokens\tprompt_ms\tgen_tokens\tgen_ms\tmicro_batches\treplans\treplans_counted\t"
              "replans_cards_busy\trelease_count_same\trelease_count_changed\tdrain_ms\tplan_ms", file=f)
        for i, q in enumerate(reqs.values()):
            same = diff = ""
            if q["count_same"] != "":
                same, diff = q["count_same"] - prev[0], q["count_diff"] - prev[1]
                prev = (q["count_same"], q["count_diff"])
            rp = q["replans"]
            print("\t".join(str(x) for x in [
                i, q["task"], q["prompt_tokens"], q["prompt_ms"], q["gen_tokens"], q["gen_ms"], q["ubatches"], len(rp),
                sum(r["counted"] for r in rp),
                sum(1 for r in rp if scheduler_of(r) == "shared" and in_flight(r) == "yes"), same, diff,
                f"{sum(r['drain_us'] for r in rp) / 1000:.1f}",
                f"{sum(r['plan_us'] for r in rp) / 1000:.1f}"]), file=f)

    groups = OrderedDict()
    for r in replans:
        key = (scheduler_of(r), pass_of(r), r["cause"], in_flight(r), r["counted"])
        g = groups.setdefault(key, dict(n=0, drain=0, plan=0, tasks=set(), ub_tokens=OrderedDict()))
        g["n"] += 1
        g["drain"] += r["drain_us"]
        g["plan"] += r["plan_us"]
        if r["task"] is not None:
            g["tasks"].add(r["task"])
        g["ub_tokens"][r["ub_tokens"]] = g["ub_tokens"].get(r["ub_tokens"], 0) + 1
    with open(os.path.join(outdir, "causes.tsv"), "w") as f:
        print("scheduler\tpass\tcause\tcards_busy\tcounted\treplans\trequests\tdrain_ms\tplan_ms\tmicro_batch_tokens",
              file=f)
        for (sched, pas, cause, busy, counted), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
            ubt = ", ".join(f"{k}:{v}" for k, v in sorted(g["ub_tokens"].items(), key=lambda kv: -kv[1]))
            print(f"{sched}\t{pas}\t{cause}\t{busy}\t{counted}\t{g['n']}\t{len(g['tasks'])}\t{g['drain'] / 1000:.1f}\t"
                  f"{g['plan'] / 1000:.1f}\t{ubt}", file=f)

    # what a re-plan with work in flight costs a request's prompt time: least squares over the requests,
    # prompt ms = a + b * prompt tokens + c * (re-plans of the shared scheduler while the cards were busy)
    X, Y = [], []
    for q in reqs.values():
        if q["prompt_ms"] == "":
            continue
        k = sum(1 for r in q["replans"] if scheduler_of(r) == "shared" and in_flight(r) == "yes")
        X.append((1.0, float(q["prompt_tokens"]), float(k)))
        Y.append(q["prompt_ms"])
    coef, se = ols(X, Y)
    n_busy = sum(1 for r in replans if scheduler_of(r) == "shared" and in_flight(r) == "yes")
    with open(os.path.join(outdir, "fit.tsv"), "w") as f:
        print("term\tms\tstandard_error_ms", file=f)
        for name, c, s in zip(("intercept (a request)", "a prompt token", "a re-plan while the cards were busy"), coef, se):
            print(f"{name}\t{c:.3f}\t{s:.3f}", file=f)
        print(f"requests\t{len(Y)}\t", file=f)
        print(f"re-plans while the cards were busy\t{n_busy}\t", file=f)
        print(f"est. their cost (s)\t{coef[2] * n_busy / 1000:.1f}\t{se[2] * n_busy / 1000:.1f}", file=f)
        print(f"prompt time of all requests (s)\t{sum(Y) / 1000:.1f}\t", file=f)


BUSY_US = 50000  # a drain this long means the cards still had micro-batches in flight (this run: idle drains are at most
                 # 34.08 ms, busy ones at least 490.37 ms)


def scheduler_of(r):
    return "shared" if r["where"] == "micro-batch" else "pipelined slot"


def pass_of(r):
    if r["where"] != "micro-batch":
        return "a pipelined token"
    if r["ub_tokens"] == 1:
        return "a 1-token pass (no pipelined guess)"
    if r["ub_outputs"] and r["ub_tokens"] <= 8:
        return f"the prompt's last {r['ub_tokens']} tokens, with the output"
    return "a prompt micro-batch"


def in_flight(r):
    return "yes" if r["drain_us"] >= BUSY_US else "no"


def ols(X, Y):
    """Least squares with standard errors, for a few columns (Gauss-Jordan on the normal equations)."""
    p = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(p)] for i in range(p)]
    M = [A[i][:] + [float(i == j) for j in range(p)] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(p):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    inv = [row[p:] for row in M]
    xty = [sum(x[i] * y for x, y in zip(X, Y)) for i in range(p)]
    coef = [sum(inv[i][j] * xty[j] for j in range(p)) for i in range(p)]
    res = [y - sum(c * v for c, v in zip(coef, x)) for x, y in zip(X, Y)]
    s2 = sum(e * e for e in res) / (len(Y) - p)
    return coef, [(s2 * inv[i][i]) ** 0.5 for i in range(p)]


if __name__ == "__main__":
    main()
