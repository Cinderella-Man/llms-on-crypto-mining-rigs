#!/usr/bin/env python3
"""What a verify column may cost for a longer MTP draft to pay (FN80, Q1396), from draft_table.py's tables.

    python3 breakeven.py TABLES [--step3 128.92] [--pass-ms 3.645] [--cols 8,10,12,15.36,21.36] > breakeven.md

A draft of d tokens is priced as the 3-token step plus (d - 3) x (one verify column + one drafting pass):
ms(d) = step3 + (d - 3) x (c + pass_ms). It pays against the 3-token draft when tau(d) / ms(d) > tau(3) / step3,
so the largest column cost c at which it still pays is step3 x (tau(d) / tau(3) - 1) / (d - 3) - pass_ms.
Defaults (est., until R121 measures the v4 step at each width): step3 = R86's v4 served step at a 3-token draft
(128.92 ms, the greedy request's median), pass_ms = R86's drafting pass (lead + GPU + turn, passes 1 and 2: 3.645 ms);
the column costs to read against are R41's v3 pass times: 15.36 ms a column from 1 to 4 tokens, 21.36 from 4 to 5.
Tables: tokens a step pooled per arm (arms.tsv) with compare.tsv's 95 % intervals over requests, and the same text
(cut.tsv: one load's verify steps cut to each shorter draft), where the arms' different replies drop out.
"""
import argparse
import csv
from pathlib import Path


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", type=Path)
    ap.add_argument("--step3", type=float, default=128.92)
    ap.add_argument("--pass-ms", type=float, default=3.645)
    ap.add_argument("--cols", default="8,10,12,15.36,21.36")
    a = ap.parse_args()
    cols = [float(x) for x in a.cols.split(",")]

    def ms(d, c):
        return a.step3 + (d - 3) * (c + a.pass_ms)

    def breakeven(ratio, d):
        return a.step3 * (ratio - 1.0) / (d - 3) - a.pass_ms

    arms = {r["arm"]: r for r in rows(a.tables / "arms.tsv")}
    comp = {r["arm"]: r for r in rows(a.tables / "compare.tsv")}
    t3 = float(arms["d3"]["tokens_a_step_pooled"])
    out = [f"step3 {a.step3} ms, drafting pass {a.pass_ms} ms (est.)\n",
           "## Each draft length against d3, pooled (est. costs)\n",
           "| arm | tokens a step | ratio to d3 | 95 % over requests | break-even column, ms | its 95 % range | "
           + " | ".join(f"tok/s at {c:g} ms" for c in cols) + " |",
           "|---|---:|---:|---|---:|---|" + "---:|" * len(cols)]
    for arm in sorted(arms, key=lambda x: int(arms[x]["n_max"])):
        d = int(arms[arm]["n_max"])
        t = float(arms[arm]["tokens_a_step_pooled"])
        speeds = " | ".join(f"{t / ms(d, c) * 1000:.2f}" for c in cols)
        if d == 3:
            out.append(f"| {arm} | {t:.4f} | 1 | | | | {speeds} |")
            continue
        lo, hi = (t3 + float(comp[arm]["lo_requests"])) / t3, (t3 + float(comp[arm]["hi_requests"])) / t3
        out.append(f"| {arm} | {t:.4f} | {t / t3:.4f} | {lo:.4f} to {hi:.4f} | {breakeven(t / t3, d):.2f} | "
                   f"{breakeven(lo, d):.2f} to {breakeven(hi, d):.2f} | {speeds} |")

    out += ["\n## The same text: each load's verify steps cut to a shorter draft (accepted + 1 per verify step)\n",
            "Cumulative: the length L against 3 on the same steps; marginal: L against L - 1, with every column "
            "before it at the same cost.\n",
            "| load | L | tokens a verify step | ratio to 3 | break-even column against 3, ms | added by L | "
            "break-even column for L against L - 1, ms |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    cut = rows(a.tables / "cut.tsv")
    seen = set()
    for r in cut:
        if r["arm"] in seen or r["label"] != r["arm"] + "-1":
            continue
        loads = [x for x in cut if x["label"] == r["label"]]
        seen.add(r["arm"])
        tau = {int(x["cut_to"]): float(x["tokens_a_verify_step"]) for x in loads}
        for L in sorted(tau):
            if L < 3:
                continue
            if L == 3:
                out.append(f"| {r['label']} | 3 | {tau[3]:.4f} | 1 | | | |")
                continue
            delta = tau[L] - tau[L - 1]
            # marginal: delta x ms(L - 1) = tau(L - 1) x (c + pass), with ms(L - 1) = step3 + (L - 4)(c + pass)
            x = a.step3 * delta / (tau[L - 1] - (L - 4) * delta)
            out.append(f"| {r['label']} | {L} | {tau[L]:.4f} | {tau[L] / tau[3]:.4f} | "
                       f"{breakeven(tau[L] / tau[3], L):.2f} | {delta:+.4f} | {x - a.pass_ms:.2f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
