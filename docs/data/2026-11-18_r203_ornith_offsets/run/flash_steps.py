#!/usr/bin/env python3
"""R203 (Q1592): Flash-Next's decode speed by context, with the draft's acceptance taken out.

    python3 -I flash_steps.py ARM=SERVER_LOG=TURNS_CSV ...

For each request of each arm, the server log's print_timing lines give the eval time, the tokens generated, the
draft's accepted tokens and its mean length (tokens a decode step); the turns table (agent/<arm>/turns.csv) gives
the context the request started at (cache_n + prompt_n), joined on (prompt tokens, generated tokens). A request is
put in the bucket of its middle context (its start plus half the tokens it generated), since a long reply grows
its own context. A decode step is one verify pass of the target plus the draft's passes:
steps = accepted / (mean length - 1). Time a step depends on the cards and the context; tokens a second also depend
on how many drafted tokens the target accepted. Prints one Markdown table: per arm and context bucket, requests,
tokens, decode s, tok/s, acceptance, tokens a step and ms a step. Standard library only.
"""
import csv
import re
import sys

BUCKETS = [(0, 4000), (4000, 8000), (8000, 16000), (16000, 10 ** 9)]
TIMING = re.compile(r"task (\d+) \|\s+(prompt eval time|eval time|draft acceptance) = (.*)$")


def requests(log):
    """task id -> {prompt_n, eval_ms, n, acc, drafted, mean_len}, in log order."""
    out = {}
    for ln in open(log, errors="replace"):
        m = TIMING.search(ln)
        if not m:
            continue
        r = out.setdefault(int(m.group(1)), {})
        what, rest = m.group(2), m.group(3)
        if what == "prompt eval time":
            r["prompt_n"] = int(re.search(r"/\s*(\d+) tokens", rest).group(1))
        elif what == "eval time":
            ms, n = re.search(r"([\d.]+) ms /\s*(\d+) tokens", rest).groups()
            r["eval_ms"], r["n"] = float(ms), int(n)
        else:
            acc, dr, ml = re.search(r"\(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)", rest).groups()
            r["acc"], r["drafted"], r["mean_len"] = int(acc), int(dr), float(ml)
    return [r for r in out.values() if "eval_ms" in r and "prompt_n" in r]


def contexts(turns):
    """(prompt_n, predicted_n) -> list of start contexts, for the join."""
    out = {}
    for t in csv.DictReader(open(turns)):
        if not t.get("prompt_n") or not t.get("predicted_n"):
            continue
        key = (int(t["prompt_n"]), int(t["predicted_n"]))
        out.setdefault(key, []).append(int(t["prompt_n"]) + int(t["cache_n"]))
    return out


def main(args):
    print("| arm | context at the request's middle | requests | tokens | decode s | tok/s | acceptance | tokens a step "
          "| ms a step |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for spec in args:
        arm, log, turns = spec.split("=", 2)
        ctx = contexts(turns)
        rows, unjoined = [], 0
        for r in requests(log):
            c = ctx.get((r["prompt_n"], r["n"]))
            if not c:
                unjoined += 1
                continue
            rows.append((c.pop(0) + r["n"] // 2, r))
        for lo, hi in BUCKETS + [(0, 10 ** 9)]:
            sel = [r for c, r in rows if lo <= c < hi]
            if not sel:
                continue
            n = sum(r["n"] for r in sel)
            ms = sum(r["eval_ms"] for r in sel)
            acc = sum(r.get("acc", 0) for r in sel)
            dr = sum(r.get("drafted", 0) for r in sel)
            steps = sum(r["acc"] / (r["mean_len"] - 1) for r in sel if r.get("mean_len", 1) > 1)
            label = "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo:,} to {hi:,}" if hi < 10 ** 9 else f"{lo:,} and up")
            print(f"| {arm} | {label} | {len(sel)} | {n} | {ms / 1000:.1f} | {n / ms * 1000:.2f} | "
                  f"{acc / dr if dr else 0:.3f} | {n / steps if steps else 0:.2f} | {ms / steps if steps else 0:.2f} |")
        if unjoined:
            print(f"| {arm} | not joined to a turn | {unjoined} | | | | | | |")


if __name__ == "__main__":
    main(sys.argv[1:])
