#!/usr/bin/env python3
"""R81 (Q1295): cut llama-server's log into its requests, so that trace_table.py can read one request at a time.

Usage: python3 -I split_requests.py <server.log> <out dir>

A request starts at the server's `launch_slot_ ... processing task` line and runs to the next one; req00.log holds
the lines before the first (the load and the warm-up). ANSI colour codes are removed. Writes <out dir>/req<NN>.log and
<out dir>/requests.tsv: per request its prompt calls (`prompt processing, n_tokens`), the server's prompt and eval
times, and the trace lines of each kind.
"""
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KINDS = ["decode-trace", "ubatch-trace", "sched-trace", "sched-replan", "galloc-trigger", "galloc-grow"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    log, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    reqs = [dict(req=0, lines=[], calls=[], prompt_n="", prompt_ms="", eval_n="", eval_ms="")]
    for raw in open(log, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw)
        if "launch_slot_" in line and "processing task" in line:
            reqs.append(dict(req=len(reqs), lines=[], calls=[], prompt_n="", prompt_ms="", eval_n="", eval_ms=""))
        r = reqs[-1]
        r["lines"].append(line)
        if "prompt processing, n_tokens" in line:
            m = re.search(r"n_tokens =\s*(\d+)", line)
            if m:
                r["calls"].append(m.group(1))
        m = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
        if m:
            r["prompt_ms"], r["prompt_n"] = m.group(1), m.group(2)
        m = re.search(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
        if m and "prompt eval" not in line:
            r["eval_ms"], r["eval_n"] = m.group(1), m.group(2)
    with open(f"{out}/requests.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(["req", "lines", "prompt_n", "prompt_ms", "eval_n", "eval_ms", "prompt_calls_logged"] + KINDS) + "\n")
        for r in reqs:
            with open(f"{out}/req{r['req']:02d}.log", "w", encoding="utf-8") as g:
                g.writelines(r["lines"])
            counts = [sum(1 for x in r["lines"] if k + ":" in x) for k in KINDS]
            f.write("\t".join(str(v) for v in [r["req"], len(r["lines"]), r["prompt_n"], r["prompt_ms"], r["eval_n"],
                                               r["eval_ms"], " ".join(r["calls"])] + counts) + "\n")


if __name__ == "__main__":
    main()
