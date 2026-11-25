#!/usr/bin/env python3
"""c1032_table.py: C1032's tables (Q1765) from run.sh's llama-perplexity logs, and the rule applied to them.

    python3 -I c1032_table.py <WORK> <C994 data folder> <dest folder>

Reads WORK/s/<arm>.8192.log (q8g and q8f1 scored against the Q8_0 cache's own run, q8; q8's log for its kernel line
and buffers) and C994's kld.tsv and chunks.tsv (its q8u, the Q8_0 cache read in micro-batches of 128, and its q8f1,
scored against the same reference). Writes kld.tsv, chunks.tsv, paired.tsv, verdict.tsv and runs.tsv to the dest
folder. The rule is the one README.md fixed before the run: per stretch i, G_i and H_i (q8g's KLD and top-token
disagreement: the attention-only yardstick), F_i and T_i (q8f1's), U_i and V_i (C994's q8u); a form is seen when its
mean over the 12 stretches is at least 3 standard errors above zero (KLD, top-token disagreement) or away from zero
(ln PPL ratio). The smallest error seen, est.: F_i scaled by (eps/0.05)^2 and T_i by eps/0.05 in each form against a
yardstick, the smallest eps with t >= 3; and the error as large as the yardstick, 0.05 * sqrt(yardstick KLD / F).
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
C994_Q8U_KLD = 0.007963   # C994's yardstick over all tokens
C994_Q8F1_KLD = 0.011030  # C994's 5 % error over all tokens
ARMS = ("q8", "q8g", "q8f1")
KERNEL = re.compile(r"compute-6\.1 prompt attention kernel: head (\d+), (\S+) K/V")
BUFFER = re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s+(\S+) MiB")
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


def scan(path, pat):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return [m.groups() for line in f for m in [pat.search(line)] if m]


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


def c994(folder):
    """C994's per-stretch values {arm: [(ln, kld, top differs), ...]} and its kld.tsv rows {arm: row}."""
    per, rows = {}, {}
    with open(os.path.join(folder, "chunks.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            per.setdefault(r["arm"], []).append(
                (float(r["ln_ppl_ratio"]), float(r["kld"]), 100.0 - float(r["same_top_pct"])))
    with open(os.path.join(folder, "kld.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["arm"]] = r
    return per, rows


def main():
    work, c994_dir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    summary, crow, per, kern, bufs, logs = [], [], {}, {}, {}, {}
    for arm in ARMS:
        path = os.path.join(work, "s", f"{arm}.{DEPTH}.log")
        kern[arm] = sorted(set(scan(path, KERNEL)))
        bufs[arm] = scan(path, BUFFER)
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
        per[arm] = [(ln, kl, 100.0 - top) for ln, kl, top in chunks(path)]
        for k, (ln, kl, td) in enumerate(per[arm], 1):
            crow.append([arm, k, f"{ln:.5f}", f"{kl:.5f}", f"{100.0 - td:.3f}"])
    write(os.path.join(dest, "kld.tsv"), COLS, summary)
    write(os.path.join(dest, "chunks.tsv"), ["arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)

    old, oldrows = c994(c994_dir)
    u = old.get("q8u")
    g = per.get("q8g")
    f1 = per.get("q8f1")
    prow, verdict = [], []

    def add(form, arm, measure, xs, rule, two_sided=False):
        m, se, t = stats(xs)
        prow.append([form, arm, measure, len(xs), f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"])
        if rule:
            ok = (abs(t) if two_sided else t) >= SEEN_T
            verdict.append([f"{form}: {arm}", measure, f"{m:.6f} ± {se:.6f} (t {t:.2f})",
                            f"|t| >= {SEEN_T:g}" if two_sided else f"t >= {SEEN_T:g}", "seen" if ok else "not seen"])
        return m, se, t

    names = {0: "ln_ppl_ratio", 1: "kld", 2: "top_differs_pct"}
    for arm, xs in (("q8g", g), ("q8f1", f1), ("C994 q8u", u)):
        if xs:
            for idx in (1, 2, 0):
                add("alone", arm, names[idx], [x[idx] for x in xs], False)

    # the attention-only yardstick against C994's whole-model one, per stretch
    if g and u and len(g) == len(u):
        for idx in (1, 2, 0):
            add("yardstick against C994's", "q8g - q8u", names[idx], [x[idx] - y[idx] for x, y in zip(g, u)], True,
                two_sided=True)
        for idx in (1, 2):
            if all(y[idx] > 0 for y in u):
                add("yardstick against C994's", "q8g / q8u - 1", names[idx],
                    [x[idx] / y[idx] - 1 for x, y in zip(g, u)], False)

    # the 5 % error against each yardstick: the new-kernel check
    for yname, y in (("q8g", g), ("C994 q8u", u)):
        if not f1 or not y or len(f1) != len(y):
            continue
        for idx in (1, 2, 0):
            add(f"against {yname}", f"q8f1 - {yname}", names[idx], [a[idx] - b[idx] for a, b in zip(f1, y)],
                yname == "q8g", two_sided=idx == 0)
        for idx in (1, 2):
            if all(b[idx] > 0 for b in y):
                add(f"against {yname}, divided by it", f"(q8f1 - {yname}) / {yname}", names[idx],
                    [(a[idx] - b[idx]) / b[idx] for a, b in zip(f1, y)], yname == "q8g")
            else:
                prow.append([f"against {yname}, divided by it", f"(q8f1 - {yname}) / {yname}", names[idx], "",
                             "a stretch has 0", "", ""])
    write(os.path.join(dest, "paired.tsv"), ["form", "arm or pair", "measure", "stretches", "mean", "se", "t"], prow)

    # the smallest error seen (est.), against each yardstick
    if f1:
        F = [x[1] for x in f1]
        T = [x[2] for x in f1]
        for yname, y in (("q8g", g), ("C994 q8u", u)):
            if not y:
                continue
            Y = [x[1] for x in y]
            Z = [x[2] for x in y]
            ests = [
                (f"against {yname}", "kld", lambda s, Y=Y: [s * a - b for a, b in zip(F, Y)], 2, True),
                (f"against {yname}", "top_differs_pct", lambda s, Z=Z: [s * a - b for a, b in zip(T, Z)], 1, True),
                (f"against {yname}, divided by it", "kld", lambda s, Y=Y: [(s * a - b) / b for a, b in zip(F, Y)], 2,
                 all(b > 0 for b in Y)),
                (f"against {yname}, divided by it", "top_differs_pct",
                 lambda s, Z=Z: [(s * a - b) / b for a, b in zip(T, Z)], 1, all(b > 0 for b in Z)),
            ]
            for form, measure, fn, power, ok in ests:
                if not ok:
                    continue
                eps = smallest_eps(fn, power)
                verdict.append([f"{form}: smallest output scale error seen, est. (q8f1 scaled by (eps/0.05)^{power})",
                                measure, "none up to 20 %" if eps is None else f"{100 * eps:.2f} %", "", "est."])
            mf, my = sum(F) / len(F), sum(Y) / len(Y)
            verdict.append([f"output scale error whose KLD equals {yname}'s, est. 0.05 * sqrt(mean {yname} / mean q8f1)"
                            " per stretch", "kld", f"{100 * FAULT1_SCALE * math.sqrt(max(my, 0) / mf):.2f} %", "",
                            "est."])
            mt, mz = sum(T) / len(T), sum(Z) / len(Z)
            verdict.append([f"output scale error whose top-token disagreement equals {yname}'s, est. 0.05 * mean "
                            f"{yname} / mean q8f1", "top_differs_pct", f"{100 * FAULT1_SCALE * mz / mt:.2f} %", "",
                            "est."])
            verdict.append([f"correlation over the stretches of q8f1's KLD with {yname}'s", "kld", f"{corr(F, Y):.2f}",
                            "", "context"])
    if "q8g" in logs and "kld" in logs["q8g"]:
        kg = float(logs["q8g"]["kld"][0])
        verdict.append(["the attention-only yardstick's KLD over all tokens, against C994's q8u (0.007963)", "kld",
                        f"{kg:.6f} against {C994_Q8U_KLD:.6f}: {kg / C994_Q8U_KLD:.3f} of it", "", "result"])
        verdict.append(["the smallest attention error seen, est. 0.05 * sqrt(q8g's KLD / 0.011030) over all tokens",
                        "kld", f"{100 * FAULT1_SCALE * math.sqrt(max(kg, 0) / C994_Q8F1_KLD):.2f} % (C994's yardstick: "
                        f"{100 * FAULT1_SCALE * math.sqrt(C994_Q8U_KLD / C994_Q8F1_KLD):.2f} %)", "", "est."])
    if g and u:
        verdict.append(["correlation over the stretches of q8g's KLD with C994's q8u's", "kld",
                        f"{corr([x[1] for x in g], [x[1] for x in u]):.2f}", "", "context"])

    # validity: the same reference as C994, q8f1 repeated exactly, the kernel lines
    for arm in ("q8g", "q8f1"):
        pb = logs.get(arm, {}).get("ppl_base")
        if pb and "q8f1" in oldrows:
            verdict.append([f"validity: {arm}'s reference perplexity (PPL(base), read from q8's file) equals C994's",
                            "ppl", f"{pb[0]} against {oldrows['q8f1']['ppl_base']}", "equal at 6 decimals",
                            "pass" if pb[0] == oldrows["q8f1"]["ppl_base"] else "FAIL"])
    if f1 and old.get("q8f1") and "q8f1" in logs:
        same = all(abs(a[1] - b[1]) < 0.000005 and abs(a[2] - b[2]) < 0.0005 for a, b in zip(f1, old["q8f1"]))
        same = same and len(f1) == len(old["q8f1"])
        verdict.append(["validity: q8f1 repeats C994's q8f1 (every stretch's KLD and same top, and the mean KLD)",
                        "kld, top", f"mean KLD {logs['q8f1']['kld'][0]} against {oldrows['q8f1']['kld']}; same top "
                        f"{logs['q8f1']['same_top'][0]} against {oldrows['q8f1']['same_top_pct']}",
                        "equal on every stretch", "pass" if same and logs['q8f1']['kld'][0] == oldrows['q8f1']['kld']
                        else "FAIL"])
    for arm in ARMS:
        ks = kern.get(arm, [])
        want = arm != "q8g"
        ok = (("256", "q8_0") in ks) if want else not ks
        verdict.append([f"validity: {arm} " + ("logs the compute 6.1 prompt attention kernel" if want else
                                              "logs no compute 6.1 prompt attention kernel line (the generic one ran)"),
                        "kernel", "; ".join(f"head {h}, {t} K/V" for h, t in ks) or "none",
                        "head 256, q8_0 K/V" if want else "none", "pass" if ok else "FAIL"])
    for arm in ARMS:
        if bufs.get(arm):
            verdict.append([f"context: {arm}'s compute buffers (the generic kernel's F16 copies of K/V make them larger)",
                            "MiB", ", ".join(f"{d} {s}" for d, s in bufs[arm]), "", "context"])
    write(os.path.join(dest, "verdict.tsv"), ["check", "measure", "value", "rule", "result"], verdict)

    path = os.path.join(work, "s", "runs.tsv")
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            runs = [line.split("\t") for line in fh.read().splitlines()[1:]]
    write(os.path.join(dest, "runs.tsv"), ["arm", "depth", "ub", "exit", "start", "end"], runs)


if __name__ == "__main__":
    main()
