#!/usr/bin/env python3
"""Tables from the logs of multitoken_perf.sh: time per product for 1-4 tokens with llama.cpp's kernels (off), the
Pascal kernels (on) and the Pascal kernels with staged activations (stage).

Usage: multitoken_perf_table.py <dir with off.log, on.log, stage.log>

For each product it prints the time per call in microseconds for n = 1..4 tokens in each run, the speed-up of the
Pascal kernels over llama.cpp's, and the cost of an n-token call relative to one token ("x1"), which is what a
speculative-decoding verify step pays per product. The last table sums the times of one Flint token's worth of calls
per type (calls per token as in Flint: 49 attention q, 98 k/v, 49 attention output, 47 shared-expert gate, up and down,
47 expert gate, up and down, 1 output head), unfused, as perf mode times them.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"^\s+([A-Z_]+)\((.*)\):\s+(\d+) runs -\s+([0-9.]+) us/run")

# (op, rows, K) -> (label, calls per Flint token)
SHAPES = {
    ("MUL_MAT", 4096, 2048): ("attn_q 2048->4096", 49),
    ("MUL_MAT", 512, 2048): ("attn_k/v 2048->512", 98),
    ("MUL_MAT", 2048, 4096): ("attn_output 4096->2048", 49),
    ("MUL_MAT", 1024, 2048): ("shexp gate/up 2048->1024", 94),
    ("MUL_MAT", 2048, 1024): ("shexp down 1024->2048", 47),
    ("MUL_MAT", 150144, 2048): ("output 2048->150144", 1),
    ("MUL_MAT_ID", 512, 2048): ("experts gate or up 2048->512", 94),
    ("MUL_MAT_ID", 2048, 512): ("experts down 512->2048", 47),
}


def parse(path):
    """{(type, label, n): us}"""
    res = {}
    if not path.exists():
        return res
    for line in path.read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        op, params, _, us = m.groups()
        kv = dict(p.split("=", 1) for p in re.split(r",(?=[a-z_]+=)", params) if "=" in p)
        typ = kv["type_a"]
        n, rows, k = int(kv["n"]), int(kv["m"]), int(kv["k"])
        key = (op, rows, k)
        if key not in SHAPES:
            continue
        res[(typ, SHAPES[key][0], n)] = float(us)
    return res


def main():
    d = Path(sys.argv[1])
    runs = {name: parse(d / f"{name}.log") for name in ("off", "on", "stage")}
    types = sorted({k[0] for r in runs.values() for k in r})
    labels = [v[0] for v in SHAPES.values()]
    calls = {v[0]: v[1] for v in SHAPES.values()}

    def fmt(x):
        return f"{x:8.1f}" if x is not None else "       -"

    for typ in types:
        print(f"\n## {typ}: us per call (n = tokens)\n")
        print("| product | n | off | on | stage | on vs off | stage vs on | on x1 | off x1 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for label in labels:
            for n in range(1, 5):
                v = {r: runs[r].get((typ, label, n)) for r in runs}
                if all(x is None for x in v.values()):
                    continue
                on1, off1 = runs["on"].get((typ, label, 1)), runs["off"].get((typ, label, 1))
                sp = f"{v['off'] / v['on']:.2f}x" if v["off"] and v["on"] else "-"
                st = f"{v['on'] / v['stage']:.2f}x" if v["on"] and v["stage"] else "-"
                xon = f"{v['on'] / on1:.2f}" if v["on"] and on1 else "-"
                xoff = f"{v['off'] / off1:.2f}" if v["off"] and off1 else "-"
                print(f"| {label} | {n} | {fmt(v['off'])} | {fmt(v['on'])} | {fmt(v['stage'])} | {sp} | {st} | {xon} | {xoff} |")

    print("\n## One Flint step's worth of these calls, ms (n = tokens per step)\n")
    print("| type | n | off | on | stage |")
    print("|---|---|---|---|---|")
    for typ in types:
        for n in range(1, 5):
            tot = defaultdict(float)
            for r in runs:
                for label in labels:
                    us = runs[r].get((typ, label, n))
                    if us is None:
                        tot[r] = float("nan")
                    else:
                        tot[r] += us * calls[label] / 1000.0
            print(f"| {typ} | {n} | {tot['off']:.2f} | {tot['on']:.2f} | {tot['stage']:.2f} |")


if __name__ == "__main__":
    main()
