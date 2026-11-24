#!/usr/bin/env python3
"""c989_table.py: C989's tables (Q1722) from run.sh's llama-perplexity logs, and the check's rule applied to them.

    python3 -I c989_table.py <WORK> <C976 data folder> <dest folder>

Reads WORK/g/<arm>.8192.log (against the F16 cache) and WORK/h/<arm>.8192.log (against Ornith-1.5-122B Q8_0's
reference), and C976's chunks.tsv and kld.tsv (its pass a's per-stretch values and pass b's q8 at 8,192). Writes
kld.tsv, chunks.tsv, paired.tsv, verdict.tsv and runs.tsv to the dest folder. The rule is the one README.md fixed
before the run: a pair's mean per-stretch difference is seen when it is at least 3 standard errors above zero (KLD) or
away from zero (ln PPL ratio); the smallest output scale error seen is est. 0.05 * sqrt(3 SE / mean) from q8f1 - q8.
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
COLS = ["pass", "arm", "against", "chunks", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se", "kld_max",
        "kld_p999", "kld_p99", "kld_p90", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio",
        "ln_ppl_ratio_se"]
DEPTH, CHUNKS = 8192, 12
SEEN_T = 3.0
FAULT1_SCALE = 0.05
KERNEL = re.compile(r"compute-6\.1 prompt attention kernel: head (\d+), (\S+) K/V")
PREFIX = re.compile(r"^[0-9.]+ [IWED] ")
AGAINST = {"g": "the F16 cache", "h": "Ornith-1.5-122B Q8_0"}


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


def write(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def c976_chunks(folder):
    """C976's pass a per-stretch values: {arm: [(ln, kld, top), ...]}."""
    out = {}
    with open(os.path.join(folder, "chunks.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["pass"] == "a":
                out.setdefault(r["arm"], []).append((float(r["ln_ppl_ratio"]), float(r["kld"]), float(r["same_top_pct"])))
    return out


def c976_b_q8(folder):
    with open(os.path.join(folder, "kld.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["arm"] == "q8" and r["depth"] == "8192":
                return float(r["kld"])
    return None


def main():
    work, c976, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    arms = {"g": ("f16", "q8", "q8f1", "q8f3", "f16u"), "h": ("q8", "q8f1", "q8f3")}
    summary, crow, per, kern = [], [], {}, {}
    for p, names in arms.items():
        for arm in names:
            path = os.path.join(work, p, f"{arm}.{DEPTH}.log")
            kern[(p, arm)] = kernels(path)
            if p == "g" and arm == "f16":
                continue  # the reference itself
            r = parse(path)
            if "kld" not in r:
                continue
            summary.append([p, arm, AGAINST[p], CHUNKS, CHUNKS * (DEPTH - 1 - DEPTH // 2), r["kld"][0], r["kld"][1],
                            r["same_top"][0], r["same_top"][1], r["kld_max"][0], r["kld_p999"][0], r["kld_p99"][0],
                            r["kld_p90"][0], r["kld_median"][0], r["rms_dp"][0], r["ppl"][0], r["ppl_base"][0],
                            *r.get("ln_ppl_ratio", ("", ""))])
            per[(p, arm)] = chunks(path)
            for k, (ln, kl, top) in enumerate(per[(p, arm)], 1):
                crow.append([p, arm, k, f"{ln:.5f}", f"{kl:.5f}", f"{top:.3f}"])
    old = c976_chunks(c976)
    if "f16" in old:
        per[("a", "f16")] = old["f16"]
        per[("a", "q8")] = old["q8"]
    write(os.path.join(dest, "kld.tsv"), COLS, summary)
    write(os.path.join(dest, "chunks.tsv"), ["pass", "arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)

    # the pairs: (pass, arm, pass of the paired arm, paired arm, what it measures)
    pairs = [("g", "q8f1", "g", "q8", "fault =1 against its own cache"),
             ("g", "q8f3", "g", "q8", "fault =3 against its own cache"),
             ("g", "q8f1", "g", "f16u", "fault =1 against rounding"),
             ("g", "q8f3", "g", "f16u", "fault =3 against rounding"),
             ("g", "q8", "g", "f16u", "the Q8_0 cache against rounding"),
             ("h", "q8f1", "h", "q8", "fault =1 against its own cache"),
             ("h", "q8f3", "h", "q8", "fault =3 against its own cache"),
             ("h", "q8f1", "a", "f16", "fault =1 against C976's F16 cache"),
             ("h", "q8f3", "a", "f16", "fault =3 against C976's F16 cache"),
             ("h", "q8", "a", "f16", "the Q8_0 cache against C976's F16 cache (C976's (A) again)")]
    prow, verdict, res = [], [], {}
    for p, arm, p2, arm2, what in pairs:
        a, b = per.get((p, arm)), per.get((p2, arm2))
        if not a or not b or len(a) != len(b):
            prow.append([p, f"{arm} - {arm2}", what, "missing"] + [""] * 9)
            continue
        out = [p, f"{arm} - {arm2}", what, len(a)]
        for idx, name, sign in ((1, "kld", 1), (2, "top_differs_pct", -1), (0, "ln_ppl_ratio", 1)):
            d = [sign * (x[idx] - y[idx]) for x, y in zip(a, b)]
            m, se, t = stats(d)
            res[(p, arm, arm2, name)] = (m, se, t)
            out += [f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"]
        prow.append(out)
        mk, sk, tk = res[(p, arm, arm2, "kld")]
        ml, sl, tl = res[(p, arm, arm2, "ln_ppl_ratio")]
        verdict.append([f"{p}: {arm} - {arm2} ({what})", "kld", f"{mk:.6f} ± {sk:.6f} (t {tk:.2f})",
                        f"t >= {SEEN_T:g}", "seen" if tk >= SEEN_T else "not seen"])
        verdict.append([f"{p}: {arm} - {arm2} ({what})", "ln_ppl_ratio", f"{ml:.6f} ± {sl:.6f} (t {tl:.2f})",
                        f"|t| >= {SEEN_T:g}", "seen" if abs(tl) >= SEEN_T else "not seen"])
    write(os.path.join(dest, "paired.tsv"), ["pass", "pair", "what", "stretches", "kld_diff", "kld_diff_se", "kld_t",
                                             "top_differs_diff_pct", "top_differs_diff_se", "top_differs_t",
                                             "ln_ppl_ratio_diff", "ln_ppl_ratio_diff_se", "ln_ppl_ratio_t"], prow)

    # the smallest output scale error seen (est.), from fault =1 against its own cache; and the error as large as rounding
    f16u = parse(os.path.join(work, "g", f"f16u.{DEPTH}.log"))
    for p in ("g", "h"):
        key = (p, "q8f1", "q8", "kld")
        if key in res and res[key][0] > 0:
            m, se, _ = res[key]
            eps = FAULT1_SCALE * math.sqrt(SEEN_T * se / m)
            verdict.append([f"{p}: smallest output scale error seen, est. 0.05 * sqrt(3 SE / mean)", "kld",
                            f"{100 * eps:.2f} %", "", "est."])
            if p == "g" and "kld" in f16u:
                er = FAULT1_SCALE * math.sqrt(float(f16u["kld"][0]) / m)
                verdict.append(["g: output scale error whose KLD equals rounding's (f16u), est. 0.05 * sqrt(KLD_f16u / mean)",
                                "kld", f"{100 * er:.2f} %", "", "est."])

    # validity: the repeats and the kernel lines
    b_q8 = c976_b_q8(c976)
    if per.get(("g", "q8")) and b_q8 is not None:
        v = per[("g", "q8")][0][1]
        verdict.append(["validity: g q8 stretch 1 repeats C976's pass b q8 at 8,192", "kld",
                        f"{v:.5f} against {b_q8:.6f}", "equal at 5 decimals",
                        "pass" if abs(v - b_q8) <= 0.000006 else "FAIL"])
    if per.get(("h", "q8")) and old.get("q8"):
        dmax = max(abs(x[i] - y[i]) for x, y in zip(per[("h", "q8")], old["q8"]) for i in range(3))
        verdict.append(["validity: h q8 repeats C976's pass a q8 on every stretch", "ln, kld, top",
                        f"largest difference {dmax:.5f}", "<= 0.00002", "pass" if dmax <= 0.00002 else "FAIL"])
    for (p, arm), ks in sorted(kern.items()):
        want = "f16" if arm.startswith("f16") else "q8_0"
        ok = ("256", want) in ks
        verdict.append([f"validity: {p} {arm} logs the compute 6.1 prompt attention kernel", "kernel",
                        "; ".join(f"head {h}, {t} K/V" for h, t in ks) or "none", f"head 256, {want} K/V",
                        "pass" if ok else "FAIL"])
    write(os.path.join(dest, "verdict.tsv"), ["check", "measure", "value", "rule", "result"], verdict)

    runs = []
    for p in ("g", "h"):
        path = os.path.join(work, p, "runs.tsv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                runs += [[p] + line.split("\t") for line in fh.read().splitlines()[1:]]
    write(os.path.join(dest, "runs.tsv"), ["pass", "arm", "depth", "ub", "exit", "start", "end"], runs)


if __name__ == "__main__":
    main()
