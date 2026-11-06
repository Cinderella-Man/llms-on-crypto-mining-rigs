#!/usr/bin/env python3
"""step_costs.py NAME=SERVER_LOG...: what a decode step costs in each arm, from llama-server's own per-request lines.

Per request the server prints its generation time and tokens ("eval time = X ms / N tokens"), with a draft its accepted
tokens ("draft acceptance = ... (A accepted / G generated)") and with LLAMA_SPEC_ADAPT its steps by verify width
("adaptive draft (on) = ... steps by verify width (1: n1, 3: n3)"). A request's first token comes from its prompt pass,
so it takes N - 1 decode tokens in N - 1 - A steps. Per arm this prints the requests, decode tokens, steps, tokens a
step, ms a token and ms a step over the session, and for an arm with the adaptive lines the steps by verify width.
Runs anywhere with Python 3 (no packages).
"""
import re
import sys
from collections import Counter, defaultdict

EVAL = re.compile(r"task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
ACC = re.compile(r"task (\d+) \| draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")
ADAPT = re.compile(r"task (\d+) \| adaptive draft \(on\) = .*steps by verify width \(([^)]*)\)")


def read(path):
    req = defaultdict(dict)
    for line in open(path, errors="replace"):
        m = EVAL.search(line)
        if m:
            req[m[1]].update(ms=float(m[2]), tok=int(m[3]))
            continue
        m = ACC.search(line)
        if m:
            req[m[1]]["acc"] = int(m[2])
            continue
        m = ADAPT.search(line)
        if m:
            req[m[1]]["w"] = {int(k): int(v) for k, v in (p.split(":") for p in m[2].split(","))}
    return [r for r in req.values() if r.get("tok", 0) > 1]


def main():
    print("arm\trequests\tdecode_tokens\tsteps\ttokens_per_step\tms_per_token\tms_per_step\tsteps_by_verify_width")
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        rs = read(path)
        ms = sum(r["ms"] for r in rs)
        tok = sum(r["tok"] - 1 for r in rs)
        steps = sum(r["tok"] - 1 - r.get("acc", 0) for r in rs)
        w = Counter()
        for r in rs:
            w.update(r.get("w", {}))
        print("\t".join([name, str(len(rs)), str(tok), str(steps), "%.3f" % (tok / steps), "%.1f" % (ms / tok),
                         "%.1f" % (ms / steps), " ".join("%d:%d" % kv for kv in sorted(w.items())) or "-"]))


if __name__ == "__main__":
    main()
