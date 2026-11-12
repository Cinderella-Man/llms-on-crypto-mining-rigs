#!/usr/bin/env python3
"""summary.py: FN85's two summary tables (Q1410) from fn85.py's requests.tsv and waits.tsv.

  python3 -I summary.py TABLES > summary.md

One table per request (the first after the server starts: greedy; the second: long), one line per load: the server's
generation time outside the profiled steps, each wait for a keep-alive block inside the generation (by card and
place), the host idles of 100 ms or more that are not waits, and the remainder.
"""
import csv
import os
import sys


def main():
    d = sys.argv[1]
    req = list(csv.DictReader(open(os.path.join(d, "requests.tsv")), delimiter="\t"))
    waits = list(csv.DictReader(open(os.path.join(d, "waits.tsv")), delimiter="\t"))

    def w(load, request, where, card, main=None, min_in_gen=100.0):
        xs = [x for x in waits if x["load"] == load and x["request"] == request and x["where"] == where
              and x["card"] == card and float(x["in_gen_ms"]) >= min_in_gen
              and (main is None or x["evaluation"].startswith("main") == main)]
        return " + ".join(f"{float(x['in_gen_ms']):.1f}" for x in xs) or "—"

    f = lambda x: f"{float(x):.1f}"  # noqa: E731
    print("### The first request (79 prompt tokens, 256 generated, temperature 0)")
    print()
    print("| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting pass | idle before the "
          "first verify pass | other, head | wait, card 0, last verify pass | idle after it | wait, card 8, last "
          "catch-up | other, tail |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in req:
        if r["request"] != "greedy":
            continue
        L = r["load"]
        print(f"| {L} | {f(r['predicted_ms'])} | {f(r['rest_ms'])} | **{f(r['outside_ms'])}** | {w(L, 'greedy', 'head', '8')} | "
              f"{f(r['head_idle_ms'])} | {f(r['head_other_ms'])} | {w(L, 'greedy', 'tail', '0', True)} | "
              f"{f(r['tail_idle_ms'])} | {w(L, 'greedy', 'tail', '8')} | {f(r['tail_other_ms'])} |")
    print()
    print("### The second request (2,760 prompt tokens, 256 generated, temperature 1.0)")
    print()
    print("| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting passes | other, head | "
          "wait, card 4, last step | other, tail |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in req:
        if r["request"] != "long":
            continue
        L = r["load"]
        print(f"| {L} | {f(r['predicted_ms'])} | {f(r['rest_ms'])} | **{f(r['outside_ms'])}** | {w(L, 'long', 'head', '8')} | "
              f"{f(r['head_other_ms'])} | {w(L, 'long', 'tail', '4')} | {f(r['tail_other_ms'])} |")


if __name__ == "__main__":
    main()
