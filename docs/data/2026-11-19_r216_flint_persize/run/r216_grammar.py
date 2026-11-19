#!/usr/bin/env python3
"""r216_grammar.py: the one-token passes of a llama-server log written with LLAMA_UBATCH_TRACE=1 (the tokens Flint writes
under the tool-call grammar, one decode call each, without pipelined guesses), and the 4-token output pass of each prompt.

    python3 -I r216_grammar.py LABEL=server.log ... > grammar.tsv

For each log: the time from one one-token micro-batch's trace line to the next, in runs of one-token passes with nothing
else between them (a gap of 1 s or more, a wait for the client, is left out), per request for the first three requests
(the first request's prompt is the same in every run of the agent check) and over the whole log: count, median, mean;
and the medians of the trace's own parts (pre, build, alloc, inputs, submit, card0, total, in us) of the one-token
passes whose graph was reused and of the 4-token output passes. Standard library only.
"""
import re
import sys

TS = re.compile(r"^\s*([0-9]+(?:\.[0-9]+){3,4})\s+[A-Z]\s")
UB = re.compile(r"ubatch-trace: (\w+) n_tokens = ([0-9]+) n_seqs = ([0-9]+) n_outputs = ([0-9]+) graph = (\w+)"
                r"(?: why = (\S+))? pre = ([0-9-]+) build = ([0-9-]+) alloc = ([0-9-]+) inputs = ([0-9-]+) "
                r"submit = ([0-9-]+) card0 = ([0-9-]+) total = ([0-9-]+) us")
LAUNCH = re.compile(r"launch_slot_: id\s+[0-9]+ \| task ([0-9-]+) \| processing task")
PARTS = ["pre", "build", "alloc", "inputs", "submit", "card0", "total"]


def ts_seconds(line):
    m = TS.match(line)
    if not m:
        return None
    p = [int(x) for x in m.group(1).split(".")]
    us = p[-1] + 1000 * p[-2]
    s = p[-3] + 60 * p[-4] + (3600 * p[-5] if len(p) == 5 else 0)
    return s + us / 1e6


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else ""


def main():
    print("run\twhat\tcount\tmedian\tmean\t" + "\t".join(f"{p}_us" for p in PARTS))
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        req, prev = -1, None
        gaps = {}
        parts = {"1-token reused": {p: [] for p in PARTS}, "4-token output pass": {p: [] for p in PARTS}}
        for line in open(path, errors="replace"):
            if LAUNCH.search(line):
                req += 1
                prev = None
                continue
            m = UB.search(line)
            if not m:
                continue
            n, outs, graph = int(m.group(2)), int(m.group(4)), m.group(5)
            key = "1-token reused" if n == 1 and graph == "reused" else "4-token output pass" if n == 4 and outs else None
            if key:
                for i, p in enumerate(PARTS):
                    parts[key][p].append(int(m.group(7 + i)))
            t = ts_seconds(line)
            if n == 1 and t is not None:
                if prev is not None and req >= 0 and (t - prev) < 1.0:
                    gaps.setdefault(req, []).append((t - prev) * 1000)
                prev = t
            else:
                prev = None
        rows = [(f"one-token gap ms, request {r}", gaps.get(r, [])) for r in (0, 1, 2)]
        rows.append(("one-token gap ms, all requests", [g for v in gaps.values() for g in v]))
        for what, v in rows:
            print(f"{label}\t{what}\t{len(v)}\t{med(v):.2f}\t{sum(v) / len(v):.2f}" + "\t" * len(PARTS) if v else
                  f"{label}\t{what}\t0\t\t" + "\t" * len(PARTS))
        for what, d in parts.items():
            print(f"{label}\t{what} (trace parts)\t{len(d['total'])}\t\t\t" + "\t".join(str(med(d[p])) for p in PARTS))


if __name__ == "__main__":
    main()
