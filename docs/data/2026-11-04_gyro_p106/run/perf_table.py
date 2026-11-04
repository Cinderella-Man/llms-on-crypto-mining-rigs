#!/usr/bin/env python3
"""Q895: test-backend-ops perf output (console format) of p106_tq.sh -> one TSV row per case.

  python3 -I perf_table.py [--ours <kcurve perf_experts.tsv>] <arm>=<perf_*.txt> ... > perf.tsv

Per case: microseconds per call, the weights the call reads (routed experts: the expected number of distinct experts
of n tokens under test-backend-ops' routing, n_used distinct experts per token drawn uniformly from n_mats), and from
those picoseconds per weight read, GB/s of weight bytes, and picoseconds per weight-token (the product's work).
--ours adds Miner Llama's Q2_0 kernels at the same cases, the `on` column of the k-curve's perf_experts.tsv
(docs/data/2026-11-04_kcurve_p106/), as arm `miner-llama`.
"""
import re
import sys

BYTES_PER_WEIGHT = {"tq2_t": 34 / 128, "q2_0": 18 / 64, "q8_0": 34 / 32}
LINE = re.compile(r"^\s*(MUL_MAT(?:_ID|_HADAMARD)?)\((.*)\):\s+(\d+) runs -\s+([0-9.]+) us/run")


def params(s):
    out = {}
    for part in re.split(r",(?![^\[]*\])", s):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def row(arm, op, p, us):
    t = p["type_a"]
    m, k, n = int(p["m"]), int(p["k"]), int(p["n"])
    if op == "MUL_MAT_HADAMARD":
        # n 128-wide blocks: 20 a token for the gate/up input, 50 (10 experts x 640) for the down input
        part, tokens = ("hadamard_gate_up", n // 20) if n % 50 else ("hadamard_down", n // 50)
        return [arm, "fwht", part, m, k, tokens, "-", f"{us:.2f}", "-", "-", "-", "-", "-"]
    if op == "MUL_MAT_ID":
        mats, used = int(p["n_mats"]), int(p["n_used"])
        distinct = mats * (1 - (1 - used / mats) ** n)
        part = "gate_up" if m < k else "down"
    else:
        used, distinct, part = 1, 1.0, "dense"
    weights = distinct * m * k
    return [arm, t, part, m, k, n, used, f"{us:.2f}", f"{distinct:.2f}", f"{weights / 1e6:.2f}",
            f"{us * 1e6 / weights:.3f}", f"{weights * BYTES_PER_WEIGHT[t] / us / 1e3:.1f}",
            f"{us * 1e6 / (n * used * m * k):.3f}"]


def main(argv):
    ours = None
    if argv[:1] == ["--ours"]:
        ours, argv = argv[1], argv[2:]
    print("\t".join(["arm", "type", "part", "m", "k", "n", "n_used", "us", "distinct_experts", "weights_read_M",
                     "ps_per_weight_read", "weight_gbs", "ps_per_weight_token"]))
    for spec in argv:
        arm, _, path = spec.partition("=")
        for line in open(path, encoding="utf-8", errors="replace"):
            mt = LINE.match(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if mt:
                print("\t".join(map(str, row(arm, mt.group(1), params(mt.group(2)), float(mt.group(4))))))
    if ours:
        for line in open(ours, encoding="utf-8"):
            name, _, rest = line.partition("\t")
            if not name.startswith("type_a=q2_0") or "keep=" in name:
                continue
            p = params(name)
            if p.get("n_mats") != "128" or p.get("n_used") != "10":
                continue
            us_on = float(rest.split("\t")[1])
            print("\t".join(map(str, row("miner-llama", "MUL_MAT_ID", p, us_on))))


if __name__ == "__main__":
    main(sys.argv[1:])
