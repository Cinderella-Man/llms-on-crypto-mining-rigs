#!/usr/bin/env python3
"""R203 (Q1592): Ornith's decode speed by context, the same buckets as run/flash_steps.py.

    python3 -I ornith_steps.py ARM=SERVER_LOG=TURNS_CSV ...

Ornith runs without a draft, so a decode step is one generated token: ms a step = eval time / tokens. For each request
of each arm, the server log's print_timing lines give the prompt tokens, the eval time and the tokens generated; the
turns table (agent/<arm>/turns.csv) gives the context the request started at (cache_n + prompt_n), joined on (prompt
tokens, generated tokens). A request is put in the bucket of its middle context (its start plus half the tokens it
generated), since a long reply grows its own context. Prints one Markdown table: per arm and context bucket,
requests, tokens, decode s, tok/s and ms a step. Standard library only.
"""
import csv
import re
import sys

BUCKETS = [(0, 4000), (4000, 8000), (8000, 16000), (16000, 10 ** 9)]
TIMING = re.compile(r"task (\d+) \|\s+(prompt eval time|eval time) = (.*)$")


def requests(log):
    """task id -> {prompt_n, eval_ms, n}, in log order."""
    out = {}
    for ln in open(log, errors="replace"):
        m = TIMING.search(ln)
        if not m:
            continue
        r = out.setdefault(int(m.group(1)), {})
        what, rest = m.group(2), m.group(3)
        if what == "prompt eval time":
            r["prompt_n"] = int(re.search(r"/\s*(\d+) tokens", rest).group(1))
        else:
            ms, n = re.search(r"([\d.]+) ms /\s*(\d+) tokens", rest).groups()
            r["eval_ms"], r["n"] = float(ms), int(n)
    return [r for r in out.values() if "eval_ms" in r and "prompt_n" in r and r["n"] > 0]


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
    print("| arm | context at the request's middle | requests | tokens | decode s | tok/s | ms a step |")
    print("|---|---|---:|---:|---:|---:|---:|")
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
            label = "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo:,} to {hi:,}" if hi < 10 ** 9 else f"{lo:,} and up")
            print(f"| {arm} | {label} | {len(sel)} | {n} | {ms / 1000:.1f} | {n / ms * 1000:.2f} | {ms / n:.2f} |")
        if unjoined:
            print(f"| {arm} | not joined to a turn | {unjoined} | | | | |")


if __name__ == "__main__":
    main(sys.argv[1:])
