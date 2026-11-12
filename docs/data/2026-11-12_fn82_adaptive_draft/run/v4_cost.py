#!/usr/bin/env python3
"""An estimated v4 verify step at 1 to 8 tokens (FN82, Q1402), until R121 maps it on the rig.

    python3 v4_cost.py --r120b DIR --c805 DIR --r41 DIR [--step3 128.92] [--pass-ms 3.645] --out DIR

V(k) is the step without its drafting passes at a verify width of k tokens. It is anchored at k = 4 on R86's v4 step
(the 3-token draft's step, 128.92 ms, less its three drafting passes of 3.645 ms) and moves from there part by part:

- the weight products the v4 kernels change (Q3_K, IQ4_XS, Q4_K, Q5_K, Q6_K one-op products): each type's ms a pass
  at 4 tokens from R120b's served profile (served/products.tsv; Q3_K and Q6_K from the loads with their kept switches,
  `on`, the others from `off`: v4c2 keeps `_Q3K_MT`, `_GLU_MT` and `_Q6K_MT` and leaves `_DMT_KT` and the F32 switch
  out), times the type's µs a product at k tokens over its µs at 4, summed over the type's measured shapes on one P106,
  from R120b's perf step (arm `mt` up to 4 tokens, Q6_K `kern`; from 5 tokens the faster of `mt` and `kern`, which is
  what `_RO8` does once C809 declines its one-warp launches). Q5_K is split into the output layer (served part "output
  layer", the 248,320 x 2,560 shape) and the rest (its other shapes). Q3_K is split by width (served/by_name.tsv):
  `ssm_out` (K = 6,144), which C726's kernel leaves to the k-column kernel from 5 tokens on, from C805's rig run (arm
  `q3x1`: `_Q3K_MT=1` and `_KCOL_EXACT=1`, the exact 6-8-token shape); the 2,560-wide products (attn_qkv, attn_gate,
  attn_q and the small ones), which C726's kernel reads once at 2-8 tokens but which no P106 run has timed past 4
  tokens, scaled as the IQ4_XS and Q4_K read-once products of the same width (6,144, 10,240 and 12,288 x 2,560);
- every other part from R41's v3 profile at 1, 4 and 5 tokens (parts.tsv, the plain passes of the release line:
  traces release-k1, release-k4 and release-k5; hops, host and gaps from its README: 3.26,
  5.84 and 6.97 ms): the part's change from 4 tokens, interpolated linearly between 1 and 4, and past 5 continued at
  its 4-to-5 slope, except the hyper-connections (BF16) and the router, which jump at 5 tokens (a different kernel)
  and are held at their 5-token time from 6 to 8.

Everything here is an estimate (est.): the product times are single kernels on one card, the other parts are v3's, and
nothing is measured as one pass. Output: v4_est.tsv (k, ms: the table adapt_replay.py --table and LLAMA_SPEC_ADAPT_COST
take) and v4_est_parts.tsv (each part's change from 4 tokens).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

R41_HOPS = {1: 3.26, 4: 5.84, 5: 6.97}  # R41 README: the unprofiled pass less the cards' GPU time
JUMP_AT_5 = {"hyper-connections (BF16)", "router"}
TYPED = {"attention and recurrent projections", "output layer", "token embeddings", "sum of the parts"}


def tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r120b", type=Path, required=True)
    ap.add_argument("--c805", type=Path, required=True)
    ap.add_argument("--r41", type=Path, required=True)
    ap.add_argument("--step3", type=float, default=128.92)
    ap.add_argument("--pass-ms", type=float, default=3.645)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    K = range(1, 9)

    # served ms a pass at 4 tokens
    prod = tsv(a.r120b / "served/products.tsv")
    parts = tsv(a.r120b / "served/parts.tsv")

    def served(rows, col, arm, key=None):
        return mean(float(r[col]) for r in rows if r["run"].endswith(arm) and (key is None or key(r)))

    out_layer = mean(float(r["ms_per_pass"]) for r in parts if r["run"].endswith("off") and r["part"] == "output layer")
    by_name = tsv(a.r120b / "served/by_name.tsv")
    q3k_ssm = served(by_name, "ms_per_pass", "on", lambda r: r["type"] == "q3_K" and r["name"] == "ssm_out")
    ms4 = {
        "q3_K K=6144 (ssm_out)": q3k_ssm,
        "q3_K K=2560": served(prod, "q3_K", "on") - q3k_ssm,
        "q6_K": served(prod, "q6_K", "on"),
        "iq4_xs": served(prod, "iq4_xs", "off"),
        "q4_K": served(prod, "q4_K", "off"),
        "q5_K output": out_layer,
        "q5_K rest": served(prod, "q5_K", "off") - out_layer,
    }

    # µs a product at 1-8 tokens, per shape
    wide = tsv(a.r120b / "perf/wide.tsv")
    us = defaultdict(lambda: defaultdict(float))  # group -> k -> µs (sum over shapes)
    for r in wide:
        t, m, n = r["type"], int(r["m"]), int(r["n"])
        mt = (float(r["mt-1"]) + float(r["mt-2"])) / 2
        kern = (float(r["kern-1"]) + float(r["kern-2"])) / 2
        if t == "q6_K":
            v = kern
        elif n <= 4:
            v = mt
        else:
            v = min(mt, kern)
        if t == "q5_K":
            g = "q5_K output" if m == 248320 else "q5_K rest"
        else:
            g = t
        us[g][n] += v
        if t in ("iq4_xs", "q4_K") and int(r["k"]) == 2560:
            us["q3_K K=2560"][n] += v  # the proxy: the read-once kernels of the same width
    c805 = tsv(a.c805 / "rig_perf.tsv")
    for k in K:
        us["q3_K K=6144 (ssm_out)"][k] = mean(float(r["q3x1 us/run"]) for r in c805
                                              if r["type"] == "q3_K" and int(r["n"]) == k)

    # R41's v3 parts
    r41 = defaultdict(dict)
    for r in tsv(a.r41 / "parts.tsv"):
        if r["trace"] in ("release-k1", "release-k4", "release-k5"):  # the plain passes of the release line
            r41[r["part"]][int(r["k"])] = float(r["ms_per_pass"])
    r41["hops, host and gaps"] = dict(R41_HOPS)

    def v3_at(part, k):
        v = r41[part]
        if k <= 4:
            return v[1] + (v[4] - v[1]) * (k - 1) / 3
        slope = 0.0 if part in JUMP_AT_5 else v[5] - v[4]
        return v[5] + (k - 5) * slope

    rows, deltas = [], defaultdict(dict)
    for g in ms4:
        for k in K:
            deltas[g][k] = ms4[g] * (us[g][k] / us[g][4] - 1.0)
    for part in r41:
        if part in TYPED:
            continue
        for k in K:
            deltas[part][k] = v3_at(part, k) - v3_at(part, 4)

    v4 = a.step3 - 3 * a.pass_ms
    table = []
    for k in K:
        table.append({"k": k, "ms": v4 + sum(deltas[g][k] for g in deltas)})
    for i, r in enumerate(table):
        r["column_ms"] = r["ms"] - table[i - 1]["ms"] if i else ""
        r["step_at_draft_k_minus_1"] = r["ms"] + (r["k"] - 1) * a.pass_ms
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "v4_est.tsv", "w") as f:
        f.write("k\tms\tcolumn_ms\tstep_at_draft_k_minus_1\n")
        for r in table:
            col = f"{r['column_ms']:.3f}" if r["column_ms"] != "" else ""
            f.write(f"{r['k']}\t{r['ms']:.3f}\t{col}\t{r['step_at_draft_k_minus_1']:.3f}\n")
    with open(a.out / "v4_est_parts.tsv", "w") as f:
        f.write("part\tms_at_4\t" + "\t".join(f"change_at_{k}" for k in K) + "\n")
        for g in deltas:
            base = ms4.get(g, r41.get(g, {}).get(4, ""))
            base = f"{base:.3f}" if base != "" else ""
            f.write(f"{g}\t{base}\t" + "\t".join(f"{deltas[g][k]:+.3f}" for k in K) + "\n")
    for r in table:
        print(r["k"], f"{r['ms']:.3f}", r["column_ms"] if r["column_ms"] == "" else f"{r['column_ms']:.3f}")


if __name__ == "__main__":
    main()
