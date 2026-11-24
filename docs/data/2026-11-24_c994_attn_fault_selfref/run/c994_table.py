#!/usr/bin/env python3
"""c994_table.py: C994's tables (Q1727) from run.sh's llama-perplexity logs, and the check's rule applied to them.

    python3 -I c994_table.py <WORK> <C989 data folder> <dest folder>

Reads WORK/s/<arm>.8192.log (every arm scored against the Q8_0 cache's own run, q8) and C989's kld.tsv and chunks.tsv
(its pass g, against the F16 cache). Writes kld.tsv, chunks.tsv, paired.tsv, verdict.tsv and runs.tsv to the dest
folder. The rule is the one README.md fixed before the run: per stretch i, F_i (q8f1's KLD), T_i (its top-token
disagreement) and U_i, V_i (q8u's, the rounding yardstick); a form is seen when its mean over the 12 stretches is at least
3 standard errors above zero (KLD, top-token disagreement) or away from zero (ln PPL ratio). The smallest error seen,
est.: F_i scaled by (eps/0.05)^2 and T_i by eps/0.05 in each form against rounding, the smallest eps with t >= 3.
"""
import csv
import math
import os
import re
import sys

PAT = {
    "kld": r"^Mean    KLD:\s+(\S+) ±\s+(\S+)",
    "same_top": r"^Same top p:\s+(\S+) ±\s+(\S+) %",
    "kld_max": r"^Maximum KLD:\s+(\S+)",
    "kld_p999": r"^99\.9%   KLD:\s+(\S+)",
    "kld_p99": r"^99\.0%   KLD:\s+(\S+)",
    "kld_p90": r"^90\.0%   KLD:\s+(\S+)",
    "kld_median": r"^Median  KLD:\s+(\S+)",
    "rms_dp": r"^RMS Δp    :\s+(\S+) ±\s+(\S+) %",
    "ppl": r"^Mean PPL\(Q\)\s+:\s+(\S+) ±\s+(\S+)",
    "ppl_base": r"^Mean PPL\(base\)\s+:\s+(\S+) ±\s+(\S+)",
    "ln_ppl_ratio": r"^Mean ln\(PPL\(Q\)/PPL\(base\)\)\s+:\s+(\S+) ±\s+(\S+)",
}
COLS = ["arm", "against", "chunks", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se", "kld_max",
        "kld_p999", "kld_p99", "kld_p90", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio",
        "ln_ppl_ratio_se"]
DEPTH, CHUNKS = 8192, 12
SEEN_T = 3.0
FAULT1_SCALE = 0.05
ARMS = ("q8", "q8f1", "q8f3", "q8u", "q8r")
KERNEL = re.compile(r"compute-6\.1 prompt attention kernel: head (\d+), (\S+) K/V")
PREFIX = re.compile(r"^[0-9.]+ [IWED] ")


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = PREFIX.sub("", line.rstrip("\n"))
            for k, p in PAT.items():
                m = re.match(p, line)
                if m:
                    out[k] = m.groups()
    return out


def kernels(path):
    """The (head, K/V type) pairs the compute 6.1 prompt attention kernel logged taking."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return sorted({m.groups() for line in f for m in [KERNEL.search(line)] if m})


def chunks(path):
    """Each chunk's own (ln PPL ratio, KLD, same top %) from llama-perplexity --kl-divergence's running-mean lines
    (C976's reader: chunk k's own value is k*M_k - (k-1)*M_(k-1))."""
    cum = []
    if not os.path.exists(path):
        return cum
    pat = re.compile(r"^\s*(\d+)\s+\S+ ±\s+\S+\s+(\S+) ±\s+\S+\s+(\S+) ±\s+\S+\s+\S+ ±\s+\S+ %\s+(\S+) ±\s+\S+ %")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = re.sub(r"^.*seconds per pass - ETA(\s+[\d.]+ (minutes|seconds|hours))?", "", line.rstrip("\n"))
            line = PREFIX.sub("", line)
            m = pat.match(line)
            if m:
                cum.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    out = []
    for i, (k, ln, kl, top) in enumerate(cum):
        if i == 0:
            out.append((ln, kl, top))
        else:
            _, ln0, kl0, top0 = cum[i - 1]
            out.append((k * ln - (k - 1) * ln0, k * kl - (k - 1) * kl0, k * top - (k - 1) * top0))
    return out


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    return m, se, (m / se if se > 0 else (0.0 if m == 0 else float("inf")))


def corr(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def write(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def smallest_eps(form, power):
    """The smallest eps (0.01 % steps up to 20 %) at which form(scale) reaches t >= 3, scale = (eps/0.05)^power."""
    for k in range(1, 2001):
        eps = k / 10000
        _, _, t = stats(form((eps / FAULT1_SCALE) ** power))
        if t >= SEEN_T:
            return eps
    return None


def c989_g(folder):
    """C989's pass g per-stretch values: {arm: [(ln, kld, top), ...]} and its kld.tsv rows {arm: row}."""
    per, rows = {}, {}
    with open(os.path.join(folder, "chunks.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["pass"] == "g":
                per.setdefault(r["arm"], []).append((float(r["ln_ppl_ratio"]), float(r["kld"]), float(r["same_top_pct"])))
    with open(os.path.join(folder, "kld.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["pass"] == "g":
                rows[r["arm"]] = r
    return per, rows


def main():
    work, c989, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    summary, crow, per, kern, logs = [], [], {}, {}, {}
    for arm in ARMS:
        path = os.path.join(work, "s", f"{arm}.{DEPTH}.log")
        kern[arm] = kernels(path)
        if arm == "q8":
            continue  # the reference itself
        r = parse(path)
        logs[arm] = r
        if "kld" not in r:
            continue
        summary.append([arm, "q8 (the Q8_0 cache's own run)", CHUNKS, CHUNKS * (DEPTH - 1 - DEPTH // 2), r["kld"][0],
                        r["kld"][1], r["same_top"][0], r["same_top"][1], r["kld_max"][0], r["kld_p999"][0],
                        r["kld_p99"][0], r["kld_p90"][0], r["kld_median"][0], r["rms_dp"][0], r["ppl"][0],
                        r["ppl_base"][0], *r.get("ln_ppl_ratio", ("", ""))])
        # (ln PPL ratio, KLD, top-token disagreement in points) per stretch
        per[arm] = [(ln, kl, 100.0 - top) for ln, kl, top in chunks(path)]
        for k, (ln, kl, td) in enumerate(per[arm], 1):
            crow.append([arm, k, f"{ln:.5f}", f"{kl:.5f}", f"{100.0 - td:.3f}"])
    write(os.path.join(dest, "kld.tsv"), COLS, summary)
    write(os.path.join(dest, "chunks.tsv"), ["arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)

    prow, verdict, res = [], [], {}

    def add(form, arm, measure, xs, rule):
        m, se, t = stats(xs)
        res[(form, arm, measure)] = (m, se, t)
        prow.append([form, arm, measure, len(xs), f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"])
        if rule:
            ok = (abs(t) if measure == "ln_ppl_ratio" else t) >= SEEN_T
            verdict.append([f"{form}: {arm}", measure, f"{m:.6f} ± {se:.6f} (t {t:.2f})",
                            f"|t| >= {SEEN_T:g}" if measure == "ln_ppl_ratio" else f"t >= {SEEN_T:g}",
                            "seen" if ok else "not seen"])

    names = {0: "ln_ppl_ratio", 1: "kld", 2: "top_differs_pct"}
    for arm in ("q8f1", "q8f3", "q8u", "q8r"):
        if arm in per:
            for idx in (1, 2, 0):
                add("alone", arm, names[idx], [x[idx] for x in per[arm]], False)
    u = per.get("q8u")
    for arm in ("q8f1", "q8f3"):
        a = per.get(arm)
        if not a or not u or len(a) != len(u):
            prow.append(["against rounding", f"{arm} - q8u", "missing", "", "", "", ""])
            continue
        for idx in (1, 2, 0):
            add("against rounding", f"{arm} - q8u", names[idx], [x[idx] - y[idx] for x, y in zip(a, u)], True)
        for idx in (1, 2):
            if all(y[idx] > 0 for y in u):
                add("against rounding, divided by rounding", f"({arm} - q8u) / q8u", names[idx],
                    [(x[idx] - y[idx]) / y[idx] for x, y in zip(a, u)], True)
    write(os.path.join(dest, "paired.tsv"), ["form", "arm or pair", "measure", "stretches", "mean", "se", "t"], prow)

    # the smallest error seen (est.) and the error as large as rounding
    f1 = per.get("q8f1")
    if f1 and u:
        F = [x[1] for x in f1]
        T = [x[2] for x in f1]
        U = [x[1] for x in u]
        V = [x[2] for x in u]
        ests = [
            ("against rounding", "kld", lambda s: [s * f - v for f, v in zip(F, U)], 2),
            ("against rounding", "top_differs_pct", lambda s: [s * f - v for f, v in zip(T, V)], 1),
            ("against rounding, divided by rounding", "kld", lambda s: [(s * f - v) / v for f, v in zip(F, U)], 2),
            ("against rounding, divided by rounding", "top_differs_pct",
             lambda s: [(s * f - v) / v for f, v in zip(T, V)], 1),
        ]
        for form, measure, fn, power in ests:
            eps = smallest_eps(fn, power)
            verdict.append([f"{form}: smallest output scale error seen, est. (q8f1 scaled by (eps/0.05)^{power})",
                            measure, "none up to 20 %" if eps is None else f"{100 * eps:.2f} %", "", "est."])
        mf, mu = sum(F) / len(F), sum(U) / len(U)
        verdict.append(["output scale error whose KLD equals rounding's (q8u), est. 0.05 * sqrt(mean U / mean F)", "kld",
                        f"{100 * FAULT1_SCALE * math.sqrt(mu / mf):.2f} %", "", "est."])
        mt, mv = sum(T) / len(T), sum(V) / len(V)
        verdict.append(["output scale error whose top-token disagreement equals rounding's, est. 0.05 * mean V / mean T",
                        "top_differs_pct", f"{100 * FAULT1_SCALE * mv / mt:.2f} %", "", "est."])
        verdict.append(["correlation over the stretches of q8f1's KLD with q8u's (C989: 0.86 with f16u's)", "kld",
                        f"{corr(F, U):.2f}", "", "context"])
        verdict.append(["correlation over the stretches of q8f1's top-token disagreement with q8u's", "top_differs_pct",
                        f"{corr(T, V):.2f}", "", "context"])
        for name, xs in (("q8f1's KLD", F), ("q8f1's KLD / q8u's", [f / v for f, v in zip(F, U)]),
                         ("q8f1's top-token disagreement", T), ("q8f1's top-token disagreement / q8u's",
                                                                 [f / v for f, v in zip(T, V)])):
            m, se, _ = stats(xs)
            verdict.append([f"spread across stretches of {name}: sd / mean", "spread",
                            f"{se * math.sqrt(len(xs)) / m:.3f}", "", "context"])

    # C989's fault against the F16 cache, per stretch, minus this run's fault alone: the cross term with the F16
    # cache's own error
    old, oldrows = c989_g(c989)
    if f1 and old.get("q8f1") and old.get("q8"):
        cross = [(a[1] - b[1]) - f[1] for a, b, f in zip(old["q8f1"], old["q8"], f1)]
        m, se, t = stats(cross)
        verdict.append(["C989's g q8f1 - q8 (against the F16 cache) minus this run's q8f1 alone, per stretch", "kld",
                        f"{m:.6f} ± {se:.6f} (t {t:.2f})", "", "context"])
        topx = [((100 - a[2]) - (100 - b[2])) - f[2] for a, b, f in zip(old["q8f1"], old["q8"], f1)]
        m, se, t = stats(topx)
        verdict.append(["C989's g q8f1 - q8 (against the F16 cache) minus this run's q8f1 alone, per stretch",
                        "top_differs_pct", f"{m:.6f} ± {se:.6f} (t {t:.2f})", "", "context"])

    # validity: the repeat, the reference against C989's, the kernel lines
    r = per.get("q8r")
    if r:
        mk = max(abs(x[1]) for x in r)
        mt = max(abs(x[2]) for x in r)
        ok = mk < 0.000005 and mt < 0.0005
        lr = logs["q8r"].get("ln_ppl_ratio", ("", ""))
        verdict.append(["validity: q8r repeats q8 exactly (every stretch's KLD 0.00000, same top 100.000 %)", "kld, top",
                        f"largest stretch KLD {mk:.5f}, largest top differs {mt:.3f}; over all tokens mean KLD "
                        f"{logs['q8r']['kld'][0]} and ln PPL ratio {lr[0]} ± {lr[1]} (the reference file's 16-bit "
                        f"storage)", "0 on every stretch", "pass" if ok else "FAIL"])
    # q8r's own perplexity is q8's run again, computed afresh (PPL(base) is read back from the reference file, which
    # stores each token's log-probabilities in 16 bits)
    if "q8" in oldrows and logs.get("q8r", {}).get("ppl"):
        a, b = logs["q8r"]["ppl"][0], oldrows["q8"]["ppl"]
        verdict.append(["validity: q8's run (repeated as q8r) has the perplexity of C989's pass g q8", "ppl",
                        f"{a} against {b}", "equal at 6 decimals", "pass" if a == b else "FAIL"])
    for arm, ks in kern.items():
        ok = ("256", "q8_0") in ks
        verdict.append([f"validity: {arm} logs the compute 6.1 prompt attention kernel", "kernel",
                        "; ".join(f"head {h}, {t} K/V" for h, t in ks) or "none", "head 256, q8_0 K/V",
                        "pass" if ok else "FAIL"])
    write(os.path.join(dest, "verdict.tsv"), ["check", "measure", "value", "rule", "result"], verdict)

    path = os.path.join(work, "s", "runs.tsv")
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            runs = [line.split("\t") for line in fh.read().splitlines()[1:]]
    write(os.path.join(dest, "runs.tsv"), ["arm", "depth", "ub", "exit", "start", "end"], runs)


if __name__ == "__main__":
    main()
