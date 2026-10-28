"""Whether two labels of a replay (bench/agent_check/replay.py rows) generated the same tokens: per trace and request,
the reply length, prompt tokens, cache hits, finish reason and tool calls must agree (the rows hold no text, and an
exact engine that generated other tokens shows up as other reply lengths or cache hits). Prints the disagreements
and a summary line; exit status 1 if any.

usage: python pair_check.py replay.jsonl [--a base] [--b stair]
"""
import argparse
import json
import sys

FIELDS = ("predicted_n", "prompt_n", "cache_n", "finish_reason", "n_tool_calls", "max_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--a", default="base")
    ap.add_argument("--b", default="stair")
    a = ap.parse_args()
    by = {}
    for line in open(a.rows):
        r = json.loads(line)
        by.setdefault(r["label"], {})[(r["trace"], r["idx"])] = r
    A, B = by.get(a.a, {}), by.get(a.b, {})
    keys = sorted(set(A) | set(B))
    bad = 0
    for k in keys:
        if k not in A or k not in B:
            print(f"{k}: only in {a.a if k in A else a.b}")
            bad += 1
            continue
        diff = [f"{f} {A[k].get(f)} / {B[k].get(f)}" for f in FIELDS if A[k].get(f) != B[k].get(f)]
        if diff:
            print(f"{k}: " + ", ".join(diff))
            bad += 1
    gen = sum(r.get("predicted_n") or 0 for r in A.values())
    print(f"PAIR_CHECK {a.a} vs {a.b}: {len(keys)} requests, {bad} differ; {gen} generated tokens in {a.a}")
    sys.exit(1 if bad else 0)


main()
