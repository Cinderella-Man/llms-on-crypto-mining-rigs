#!/usr/bin/env python3
"""Tables for R16 (Q223, Q546) from bench/mmvq_pascal's CSVs and test-backend-ops perf output.

Usage: python3 summarize.py DIR > tables.md
DIR holds q2_0.csv and bf16.csv (bench/mmvq_pascal --q2_0 / --bf16 --tokens 1,2,3,4 --csv) and the four
test-backend-ops perf logs perf_q2_0_{stock,pascal}.txt and perf_bf16_mmvf{0,1}.txt.
"""
import csv
import os
import re
import sys

Q2_0_BYTES_PER_WEIGHT = 18 / 64  # FP16 scale + 64 two-bit codes
BF16_BYTES_PER_WEIGHT = 2


def rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["shape"] != "TOKEN_TOTAL"]  # the bench's own totals


def q2_0_tables(d):
    data = rows(os.path.join(d, "q2_0.csv"))
    shapes = list(dict.fromkeys(r["shape"] for r in data))
    out = ["### Q2_0 experts, one token (bench/mmvq_pascal --q2_0)", "",
           "| Shape | Calls/token | Weights/call | llama µs | pascal µs | best | best µs | llama ps/w | pascal ps/w "
           "| best ps/w | pascal GB/s | speed-up | llama ms/token | pascal ms/token | best ms/token |",
           "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = {"llama": 0.0, "pascal": 0.0, "best": 0.0}
    for s in shapes:
        rs = [r for r in data if r["shape"] == s and r["check"] == "PASS"]
        by = {r["variant"]: r for r in rs}
        llama, pascal = by["llama"], by["pascal"]
        best = min((r for r in rs if r["variant"] != "llama"), key=lambda r: float(r["us_median"]))
        calls = int(llama["calls_per_token"])
        w = int(llama["bytes_per_call"]) / Q2_0_BYTES_PER_WEIGHT
        us = {k: float(v["us_median"]) for k, v in (("llama", llama), ("pascal", pascal), ("best", best))}
        for k in tot:
            tot[k] += calls * us[k] / 1000
        out.append(f"| `{s}` | {calls} | {w / 1e6:.2f} M | {us['llama']:.2f} | {us['pascal']:.2f} | `{best['variant']}` "
                   f"| {us['best']:.2f} | {us['llama'] * 1e6 / w:.2f} | {us['pascal'] * 1e6 / w:.2f} "
                   f"| {us['best'] * 1e6 / w:.2f} | {float(pascal['gbs']):.1f} | {us['llama'] / us['pascal']:.2f}× "
                   f"| {calls * us['llama'] / 1000:.2f} | {calls * us['pascal'] / 1000:.2f} "
                   f"| {calls * us['best'] / 1000:.2f} |")
    out.append(f"| **all four** | | | | | | | | | | | {tot['llama'] / tot['pascal']:.2f}× | **{tot['llama']:.2f}** "
               f"| **{tot['pascal']:.2f}** | **{tot['best']:.2f}** |")
    return out


def bf16_tables(d):
    data = rows(os.path.join(d, "bf16.csv"))
    out = ["### BF16 products, 1–4 tokens (bench/mmvq_pascal --bf16 --tokens 1,2,3,4)", "",
           "| Tokens | Shape | Calls/token | Weights/call | mmvf µs | pascal (G,R) µs | best | best µs | mmvf ps/w "
           "| pascal ps/w | pascal GB/s | speed-up | mmvf ms/step | pascal ms/step | best ms/step |",
           "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = []
    for t in sorted({int(r["tokens"]) for r in data}):
        tr = [r for r in data if int(r["tokens"]) == t and r["check"] == "PASS"]
        tot = {"mmvf": 0.0, "pascal": 0.0, "best": 0.0}
        for s in dict.fromkeys(r["shape"] for r in tr):
            rs = [r for r in tr if r["shape"] == s]
            by = {r["variant"]: r for r in rs}
            mmvf, pascal = by["mmvf"], by["pascal"]
            best = min((r for r in rs if r["variant"] != "mmvf"), key=lambda r: float(r["us_median"]))
            calls = int(mmvf["calls_per_token"])
            w = int(mmvf["bytes_per_call"]) / BF16_BYTES_PER_WEIGHT
            us = {k: float(v["us_median"]) for k, v in (("mmvf", mmvf), ("pascal", pascal), ("best", best))}
            for k in tot:
                tot[k] += calls * us[k] / 1000
            out.append(f"| {t} | `{s}` | {calls} | {w / 1e6:.3f} M | {us['mmvf']:.2f} "
                       f"| {us['pascal']:.2f} ({pascal['G']},{pascal['R']}) | `{best['variant']}` | {us['best']:.2f} "
                       f"| {us['mmvf'] * 1e6 / w:.2f} | {us['pascal'] * 1e6 / w:.2f} | {float(pascal['gbs']):.1f} "
                       f"| {us['mmvf'] / us['pascal']:.2f}× | {calls * us['mmvf'] / 1000:.2f} "
                       f"| {calls * us['pascal'] / 1000:.2f} | {calls * us['best'] / 1000:.2f} |")
        summary.append((t, tot))
    out += ["", "| Tokens per step | mmvf ms/step | pascal ms/step | best ms/step | speed-up |", "|---:|---:|---:|---:|---:|"]
    for t, tot in summary:
        out.append(f"| {t} | {tot['mmvf']:.2f} | {tot['pascal']:.2f} | {tot['best']:.2f} | {tot['mmvf'] / tot['pascal']:.2f}× |")
    return out


PERF = re.compile(r"^\s*(MUL_MAT(?:_ID)?)\((.*)\):\s+\d+ runs - +([\d.]+) us/run")


def perf(path):
    res = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = PERF.match(line)
            if m:
                res[f"{m.group(1)}({m.group(2)})"] = float(m.group(3))
    return res


def short(case):
    keep = ("n_mats", "n_used", "b", "m", "n", "k")
    op, args = case.split("(", 1)
    kv = dict(p.split("=", 1) for p in re.split(r",(?![^\[]*\])", args.rstrip(")")) if "=" in p)
    return op + " " + " ".join(f"{k}={kv[k]}" for k in keep if k in kv)


def perf_table(d, title, off, on, off_name, on_name):
    a, b = perf(os.path.join(d, off)), perf(os.path.join(d, on))
    out = [f"### {title}", "", f"| Case | {off_name} µs | {on_name} µs | speed-up |", "|---|---:|---:|---:|"]
    for case, us in a.items():
        if case in b:
            out.append(f"| `{short(case)}` | {us:.2f} | {b[case]:.2f} | {us / b[case]:.2f}× |")
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = q2_0_tables(d) + [""] + bf16_tables(d) + [""]
    out += perf_table(d, "Q2_0 inside ggml (test-backend-ops perf -o MUL_MAT,MUL_MAT_ID -p type_a=q2_0)",
                      "perf_q2_0_stock.txt", "perf_q2_0_pascal.txt",
                      "llama.cpp (`GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0`)", "Pascal (default)") + [""]
    out += perf_table(d, "BF16 inside ggml (test-backend-ops perf -o MUL_MAT -p type_a=bf16)",
                      "perf_bf16_mmvf0.txt", "perf_bf16_mmvf1.txt",
                      "llama.cpp (`GGML_CUDA_MMVF_PASCAL=0`)", "Pascal (default)")
    print("\n".join(out))


if __name__ == "__main__":
    main()
