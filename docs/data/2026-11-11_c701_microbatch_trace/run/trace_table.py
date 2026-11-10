#!/usr/bin/env python3
"""C701 (Q1238): tables from the LLAMA_UBATCH_TRACE lines of one log (llama-server's log or a tool's stderr).

Usage: python3 -I trace_table.py <log> <out dir> [--min-tokens N]

Writes to <out dir>:
  calls.tsv    one row per decode call (decode-trace): its context, tokens, the host's prep, init and wait, and over
               its micro-batches: count, rebuilt and reused, re-plans, the sums of card0, submit, the splits' w and l
  ubatch.tsv   one row per micro-batch (ubatch-trace) with its call, its fields, and from its evaluation (the
               sched-trace line before it): the input copy, splits, wait and card0
  splits.tsv   one row per split of those evaluations: backend, inputs, nodes, w, c, l, a CPU split's first node
  replan.tsv   one row per sched-replan line with the micro-batch it came before and the galloc-trigger lines before it
  summary.txt  per context and micro-batch size: micro-batches, rebuilt by reason, card0 median and sum; per backend the
               sums of w, c and l; the re-plans
--min-tokens: summary.txt and the per-backend sums count only micro-batches of at least N tokens (default 1).
A sched-trace line belongs to the next ubatch-trace line; one with none (an evaluation outside a micro-batch, such as
a pipelined decode step) is counted as unassigned.
"""
import re
import statistics
import sys
from collections import defaultdict

RE_CALL = re.compile(r"decode-trace: (\w+) n_tokens = (\d+) n_outputs = (\d+) prep = (-?\d+) init = (-?\d+) wait = (-?\d+) us")
RE_UB = re.compile(r"ubatch-trace: (\w+) n_tokens = (\d+) n_seqs = (\d+) n_outputs = (\d+) graph = (\w+)(?: why = (\S+))? "
                   r"pre = (-?\d+) build = (-?\d+) alloc = (-?\d+) inputs = (-?\d+) submit = (-?\d+) card0 = (-?\d+) total = (-?\d+) us")
RE_SCHED = re.compile(r"sched-trace: copy = (\d+) splits = (\d+) wait = (-?\d+) card0 = (-?\d+) total = (-?\d+) us \|(.*)$")
RE_SPLIT = re.compile(r"^(.+?):i(\d+):n(\d+):w(-?\d+):c(-?\d+):l(-?\d+)(?::(.*))?$")  # an RPC backend's name holds ':'
RE_REPLAN = re.compile(r"sched-replan: (.*)$")
RE_TRIG = re.compile(r"galloc-(?:trigger|grow): (.*)$")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    log, out = sys.argv[1], sys.argv[2]
    min_tokens = 1
    if "--min-tokens" in sys.argv:
        min_tokens = int(sys.argv[sys.argv.index("--min-tokens") + 1])

    calls = []          # dicts
    cur_call = {}       # context -> index into calls
    ubs = []
    splits = []
    replans = []
    pending_sched = None
    pending_trig = []
    pending_replans = []
    unassigned = 0

    with open(log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_CALL.search(line)
            if m:
                ctx = m.group(1)
                calls.append(dict(call=len(calls), ctx=ctx, n_tokens=int(m.group(2)), n_outputs=int(m.group(3)),
                                  prep=int(m.group(4)), init=int(m.group(5)), wait=int(m.group(6)),
                                  ubatches=0, rebuilt=0, reused=0, replans=0, card0=0, submit=0, w=0, l=0))
                cur_call[ctx] = len(calls) - 1
                continue
            m = RE_TRIG.search(line)
            if m:
                pending_trig.append(line[line.index("galloc-"):])
                continue
            m = RE_REPLAN.search(line)
            if m:
                r = dict(replan=len(replans), text=m.group(1), triggers=" | ".join(pending_trig), ubatch=-1)
                pending_trig = []
                replans.append(r)
                pending_replans.append(r)
                continue
            m = RE_SCHED.search(line)
            if m:
                if pending_sched is not None:
                    unassigned += 1
                sp = []
                for tok in m.group(6).split():
                    s = RE_SPLIT.match(tok)
                    if s:
                        sp.append(dict(backend=s.group(1), inputs=int(s.group(2)), nodes=int(s.group(3)), w=int(s.group(4)),
                                       c=int(s.group(5)), l=int(s.group(6)), first=s.group(7) or ""))
                pending_sched = dict(copy=int(m.group(1)), splits=int(m.group(2)), wait=int(m.group(3)),
                                     card0=int(m.group(4)), total=int(m.group(5)), sp=sp)
                continue
            m = RE_UB.search(line)
            if m:
                ctx = m.group(1)
                u = dict(ubatch=len(ubs), call=cur_call.get(ctx, -1), ctx=ctx, n_tokens=int(m.group(2)),
                         n_seqs=int(m.group(3)), n_outputs=int(m.group(4)), graph=m.group(5), why=m.group(6) or "",
                         pre=int(m.group(7)), build=int(m.group(8)), alloc=int(m.group(9)), inputs=int(m.group(10)),
                         submit=int(m.group(11)), card0=int(m.group(12)), total=int(m.group(13)),
                         copy="", n_splits="", s_wait="", s_card0="")
                s = pending_sched
                pending_sched = None
                if s is not None:
                    u.update(copy=s["copy"], n_splits=s["splits"], s_wait=s["wait"], s_card0=s["card0"])
                    for i, x in enumerate(s["sp"]):
                        splits.append(dict(ubatch=u["ubatch"], ctx=ctx, n_tokens=u["n_tokens"], split=i, **x))
                for r in pending_replans:
                    r["ubatch"] = u["ubatch"]
                u["replans"] = len(pending_replans)
                pending_replans = []
                ubs.append(u)
                if u["call"] >= 0:
                    c = calls[u["call"]]
                    c["ubatches"] += 1
                    c["rebuilt" if u["graph"] == "rebuilt" else "reused"] += 1
                    c["replans"] += u["replans"]
                    c["card0"] += max(u["card0"], 0)
                    c["submit"] += u["submit"]
                    if s is not None:
                        c["w"] += sum(x["w"] for x in s["sp"])
                        c["l"] += sum(x["l"] for x in s["sp"])
    if pending_sched is not None:
        unassigned += 1

    def write(name, rows, cols):
        with open(f"{out}/{name}", "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    write("calls.tsv", calls, ["call", "ctx", "n_tokens", "n_outputs", "prep", "init", "wait", "ubatches", "rebuilt",
                               "reused", "replans", "card0", "submit", "w", "l"])
    write("ubatch.tsv", ubs, ["ubatch", "call", "ctx", "n_tokens", "n_seqs", "n_outputs", "graph", "why", "replans", "pre",
                              "build", "alloc", "inputs", "submit", "card0", "total", "copy", "n_splits", "s_wait", "s_card0"])
    write("splits.tsv", splits, ["ubatch", "ctx", "n_tokens", "split", "backend", "inputs", "nodes", "w", "c", "l", "first"])
    write("replan.tsv", replans, ["replan", "ubatch", "text", "triggers"])

    with open(f"{out}/summary.txt", "w", encoding="utf-8") as f:
        f.write(f"log {log}: {len(calls)} decode calls, {len(ubs)} micro-batches, {len(splits)} splits, "
                f"{len(replans)} re-plans, {unassigned} evaluations outside a micro-batch; min tokens {min_tokens}\n\n")
        groups = defaultdict(list)
        for u in ubs:
            if u["n_tokens"] >= min_tokens:
                groups[(u["ctx"], u["n_tokens"])].append(u)
        f.write("ctx\tn_tokens\tubatches\treused\trebuilt by reason\tcard0 median us\tcard0 sum us\treplans\n")
        for (ctx, n), us in sorted(groups.items()):
            why = defaultdict(int)
            for u in us:
                if u["graph"] == "rebuilt":
                    why[u["why"]] += 1
            c0 = [u["card0"] for u in us if u["card0"] >= 0]
            f.write(f"{ctx}\t{n}\t{len(us)}\t{sum(1 for u in us if u['graph'] == 'reused')}\t"
                    f"{', '.join(f'{k} {v}' for k, v in sorted(why.items())) or '-'}\t"
                    f"{statistics.median(c0) if c0 else -1}\t{sum(c0)}\t{sum(u['replans'] for u in us)}\n")
        f.write("\nbackend\tsplits\tw sum us\tc sum us\tl sum us\tl max us\n")
        per = defaultdict(lambda: [0, 0, 0, 0, 0])
        for s in splits:
            if s["n_tokens"] < min_tokens:
                continue
            p = per[s["backend"]]
            p[0] += 1
            p[1] += s["w"]
            p[2] += s["c"]
            p[3] += s["l"]
            p[4] = max(p[4], s["l"])
        for b, p in sorted(per.items()):
            f.write(f"{b}\t{p[0]}\t{p[1]}\t{p[2]}\t{p[3]}\t{p[4]}\n")
        f.write("\nre-plans (first 20)\n")
        for r in replans[:20]:
            f.write(f"#{r['replan']} before micro-batch {r['ubatch']}: {r['text']}"
                    f"{' <- ' + r['triggers'] if r['triggers'] else ''}\n")


if __name__ == "__main__":
    main()
