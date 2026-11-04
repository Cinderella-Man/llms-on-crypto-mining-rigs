#!/usr/bin/env python3
"""Tables of bench/mmvq_pascal/iq_perf.sh: the grid types' products at Qwen3.8-Flash-Next's shapes, Pascal kernels
(on) against llama.cpp's (off).
  python3 -I iq_perf_table.py OUT_DIR [--layout LAYOUT.txt ...] > iq_perf.csv     (the CSV on stdout)
writes OUT_DIR/iq_perf.md: per type and case the microseconds per call off and on, the speed-up, and the weight bytes
read per microsecond (GB/s) at one token; then, per --layout file (the output of
docs/data/2026-11-04_iq2xs_kernel/run/iq_layout.py for a model file), the milliseconds a decoded token of that file
spends in the timed products of these types, one card's kernel after another, off and on.
Weight bytes per call: dense, every row; routed (10 of 128 random experts per token), the expected number of distinct
experts the tokens pick, 128 (1 - (1 - 10/128)^n), times one expert's bytes. Standard library only.
"""
import collections, os, re, sys

BS = {"iq2_xxs": 66, "iq2_xs": 74, "iq3_s": 110, "iq2_s": 82, "iq3_xxs": 98}  # bytes per 256 weights
PERF = re.compile(r"^\s*(MUL_MAT(?:_ID)?)\((.*?)\):\s+\d+ runs -\s+([\d.]+) us/run")


def kv(s):
    return dict(p.split("=", 1) for p in re.split(r",(?![^\[]*\])", s) if "=" in p)


def load(d, run):
    rows = []
    p = os.path.join(d, f"{run}.log")
    if os.path.exists(p):
        for line in open(p):
            m = PERF.match(line)
            if not m:
                continue
            op, v, us = m.group(1), kv(m.group(2)), float(m.group(3))
            t, n, mm, k = v["type_a"], int(v["n"]), int(v["m"]), int(v["k"])
            case = "moe_gate_or_up" if op == "MUL_MAT_ID" else f"dense_{mm}x{k}"
            rows.append((run, case, t, mm, k, n, us))
    p = os.path.join(d, f"{run}-fused.txt")
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if len(f) >= 3 and f[2] != "-":
                rows.append((run, "moe_gate_up_fused", f[0], 640, 2560, int(f[1]), float(f[2])))
    return rows


def wbytes(case, t, m, k, n):
    rb = k // 256 * BS[t]
    if case.startswith("dense"):
        return m * rb
    experts = 128 * (1 - (1 - 10 / 128) ** n)
    return experts * 640 * rb * (2 if case == "moe_gate_up_fused" else 1)


def layout_counts(path):
    """(case, type) -> calls per token, from an iq_layout.py output"""
    calls = collections.Counter()
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) != 5 or not f[0].startswith("blk.N."):
            continue
        kind, t, n, shape = f[0][6:], f[1].lower(), int(f[2]), f[3]
        if t not in BS:
            continue
        if kind == "ffn_gate_exps.weight" and shape == "2560x640x512":
            calls[("moe_gate_up_fused", t)] += n
        elif shape in ("2560x10240", "2560x12288", "2560x6144", "6144x2560"):
            k, m = shape.split("x")
            calls[(f"dense_{m}x{k}", t)] += n
    return calls


def main():
    d = sys.argv[1]
    layouts = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--layout"]
    rows = load(d, "off") + load(d, "on")
    print("run,case,type,m,k,tokens,us,weight_bytes,gbs")
    t_us = {}
    for run, case, t, m, k, n, us in rows:
        b = wbytes(case, t, m, k, n)
        t_us[(run, case, t, n)] = us
        print(f"{run},{case},{t},{m},{k},{n},{us:.2f},{b:.0f},{b / us / 1e3:.1f}")
    md = ["| Case | Type | Tokens | llama.cpp's, us | Pascal, us | Speed-up | Pascal, GB/s |", "|---|---|---|---|---|---|---|"]
    keys = sorted({(c, t, n) for (_, c, t, n) in t_us}, key=lambda x: (x[0], list(BS).index(x[1]), x[2]))
    for c, t, n in keys:
        off, on = t_us.get(("off", c, t, n)), t_us.get(("on", c, t, n))
        m, k = (640, 2560) if c.startswith("moe") else map(int, c[6:].split("x"))
        gbs = f"{wbytes(c, t, m, k, n) / on / 1e3:.1f}" if on and n == 1 else ""
        md.append(f"| {c} | {t} | {n} | {off or '':} | {on or '':} | {f'{off / on:.2f}x' if off and on else ''} | {gbs} |")
    for lp in layouts:
        calls = layout_counts(lp)
        md += ["", f"**{os.path.basename(lp)}**: a decoded token's timed products of these types, ms (one token)", "",
               "| Case | Type | Calls | llama.cpp's | Pascal |", "|---|---|---|---|---|"]
        so = sn = 0.0
        for (c, t), n in sorted(calls.items()):
            off, on = t_us.get(("off", c, t, 1)), t_us.get(("on", c, t, 1))
            so += n * (off or 0) / 1e3
            sn += n * (on or 0) / 1e3
            md.append(f"| {c} | {t} | {n} | {n * off / 1e3:.2f} | {n * on / 1e3:.2f} |" if off and on else
                      f"| {c} | {t} | {n} | (not timed) | |")
        md.append(f"| **all** | | | **{so:.2f}** | **{sn:.2f}** |")
    open(os.path.join(d, "iq_perf.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
