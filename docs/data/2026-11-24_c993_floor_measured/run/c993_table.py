#!/usr/bin/env python3
"""c993_table.py: C993's tables (Q1726) from run.sh's llama-perplexity logs, and the rule applied to them.

    python3 -I c993_table.py <WORK> <C989 data folder> <dest folder>

Reads WORK/g/<arm>.8192.log (every arm against the F16 cache's reference, written by the f16 arm) and WORK/g/runs.tsv,
and C989's kld.tsv and chunks.tsv (its pass g, for the repeat). Writes kld.tsv, chunks.tsv, paired.tsv, verdict.tsv
and runs.tsv to the dest folder. The rule is the one README.md fixed before the run: for each factor f, the fault arm's
per-stretch value minus the unfaulted Q8_0 arm's (q8) on the same stretch, over the 12 stretches; a factor is seen when
the mean difference is at least 3 standard errors above zero in KLD or in top-token disagreement (one-sided); the
answer is the smallest factor seen. The ln PPL ratio difference is reported two-sided and is not part of the answer.
"""
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
COLS = ["arm", "factor", "against", "chunks", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se",
        "kld_max", "kld_p999", "kld_p99", "kld_p90", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio",
        "ln_ppl_ratio_se"]
DEPTH, CHUNKS = 8192, 12
SEEN_T = 3.0
ARMS = ["q8", "q8f101", "q8f102", "q8f103", "q8f105"]
FACTOR = {"q8": None, "q8f101": 1.01, "q8f102": 1.02, "q8f103": 1.03, "q8f105": 1.05}
C989_EST = {"kld": 3.43, "top_differs_pct": 1.37}  # C989's est. smallest error seen, % (against the F16 cache)
KERNEL = re.compile(r"compute-6\.1 prompt attention kernel: head (\d+), (\S+) K/V")
FAULT_LINE = re.compile(r"FATTN_TILE_PASCAL_FAULT=1 \(tests of the tests\)\S*\s*output x\s*([0-9.]+)")
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


def grep_all(path, rx):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return sorted({m.groups() for line in f for m in [rx.search(line)] if m})


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


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def main():
    work, c989, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    g = os.path.join(work, "g")
    summary, crow, per, kern, faults = [], [], {}, {}, {}
    for arm in ["f16"] + ARMS:
        path = os.path.join(g, f"{arm}.{DEPTH}.log")
        kern[arm] = grep_all(path, KERNEL)
        faults[arm] = grep_all(path, FAULT_LINE)
        if arm == "f16":
            continue  # the reference itself
        r = parse(path)
        if "kld" not in r:
            continue
        f = FACTOR[arm]
        summary.append([arm, f"{f:.2f}" if f else "", "the F16 cache", CHUNKS, CHUNKS * (DEPTH - 1 - DEPTH // 2),
                        r["kld"][0], r["kld"][1], r["same_top"][0], r["same_top"][1], r["kld_max"][0],
                        r["kld_p999"][0], r["kld_p99"][0], r["kld_p90"][0], r["kld_median"][0], r["rms_dp"][0],
                        r["ppl"][0], r["ppl_base"][0], *r.get("ln_ppl_ratio", ("", ""))])
        per[arm] = chunks(path)
        for k, (ln, kl, top) in enumerate(per[arm], 1):
            crow.append([arm, f"{f:.2f}" if f else "", k, f"{ln:.5f}", f"{kl:.5f}", f"{top:.3f}"])
    write(os.path.join(dest, "kld.tsv"), COLS, summary)
    write(os.path.join(dest, "chunks.tsv"), ["arm", "factor", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)

    # the pairs: each fault arm minus q8, stretch by stretch
    prow, verdict, res = [], [], {}
    base = per.get("q8")
    for arm in ARMS[1:]:
        a, f = per.get(arm), FACTOR[arm]
        eps = f - 1
        if not a or not base or len(a) != len(base):
            prow.append([f"{arm} - q8", f"{f:.2f}", "missing"] + [""] * 14)
            continue
        out = [f"{arm} - q8", f"{f:.2f}", len(a)]
        for idx, name, sign in ((1, "kld", 1), (2, "top_differs_pct", -1), (0, "ln_ppl_ratio", 1)):
            d = [sign * (x[idx] - y[idx]) for x, y in zip(a, base)]
            m, se, t = stats(d)
            res[(arm, name)] = (m, se, t, sum(1 for v in d if v > 0))
            out += [f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"]
        mk, sk, tk, nk = res[(arm, "kld")]
        mt, st, tt, nt = res[(arm, "top_differs_pct")]
        # est. smallest error seen if the difference grew as eps^2 (KLD) or eps (top token) at a fixed spread
        fk = 100 * eps * math.sqrt(SEEN_T * sk / mk) if mk > 0 else float("nan")
        ft = 100 * eps * SEEN_T * st / mt if mt > 0 else float("nan")
        out += [nk, nt, f"{mk / eps ** 2:.4f}", f"{mt / eps:.3f}", f"{fk:.2f}", f"{ft:.2f}"]
        prow.append(out)
        for name, m, se, t in (("kld", mk, sk, tk), ("top_differs_pct", mt, st, tt)):
            verdict.append([f"x {f:.2f}: {arm} - q8", name, f"{m:.6f} ± {se:.6f} (t {t:.2f})",
                            f"t >= {SEEN_T:g}", "seen" if t >= SEEN_T else "not seen"])
        ml, sl, tl, _ = res[(arm, "ln_ppl_ratio")]
        verdict.append([f"x {f:.2f}: {arm} - q8", "ln_ppl_ratio", f"{ml:.6f} ± {sl:.6f} (t {tl:.2f})",
                        f"|t| >= {SEEN_T:g} (reported, not in the answer)",
                        "seen" if abs(tl) >= SEEN_T else "not seen"])
    # context, not in the rule: each arm against C989's rounding yardstick (its f16u arm, the F16 cache read in
    # micro-batches of 128, run on C989's build), stretch by stretch; valid as a pair only where q8 repeats C989's q8
    old_f16u = None
    for r in read_tsv(os.path.join(c989, "chunks.tsv")):
        if r["pass"] == "g" and r["arm"] == "f16u":
            old_f16u = (old_f16u or []) + [(float(r["ln_ppl_ratio"]), float(r["kld"]), float(r["same_top_pct"]))]
    for arm in ARMS:
        a, f = per.get(arm), FACTOR[arm]
        if not a or not old_f16u or len(a) != len(old_f16u):
            continue
        out = [f"{arm} - f16u (C989's arm, context)", f"{f:.2f}" if f else "1.00", len(a)]
        ups = []
        for idx, sign in ((1, 1), (2, -1), (0, 1)):
            d = [sign * (x[idx] - y[idx]) for x, y in zip(a, old_f16u)]
            m, se, t = stats(d)
            out += [f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"]
            ups.append(sum(1 for v in d if v > 0))
        prow.append(out + [ups[0], ups[1], "", "", "", ""])
    write(os.path.join(dest, "paired.tsv"),
          ["pair", "factor", "stretches", "kld_diff", "kld_diff_se", "kld_t", "top_differs_diff_pct",
           "top_differs_diff_se", "top_differs_t", "ln_ppl_ratio_diff", "ln_ppl_ratio_diff_se", "ln_ppl_ratio_t",
           "stretches_kld_up", "stretches_top_differs_up", "kld_diff_per_eps2", "top_differs_diff_per_eps",
           "est_floor_kld_pct", "est_floor_top_pct"], prow)

    # the answer: the smallest factor seen, by each measure and by either
    seen = {}
    for name in ("kld", "top_differs_pct"):
        fs = [FACTOR[a] for a in ARMS[1:] if (a, name) in res and res[(a, name)][2] >= SEEN_T]
        seen[name] = min(fs) if fs else None
        verdict.append([f"answer: the smallest factor seen in {name} (C989's est.: {C989_EST[name]:.2f} %)", name,
                        f"x {seen[name]:.2f} ({100 * (seen[name] - 1):.0f} %)" if seen[name] else "none of x 1.01-1.05",
                        f"t >= {SEEN_T:g}", "measured"])
    both = [v for v in seen.values() if v]
    verdict.append(["answer: the smallest factor seen in KLD or top-token disagreement", "kld, top_differs_pct",
                    f"x {min(both):.2f} ({100 * (min(both) - 1):.0f} %)" if both else "none of x 1.01-1.05",
                    f"t >= {SEEN_T:g}", "measured"])

    # validity: the kernel lines, the factor lines, and the repeat of C989's pass g
    for arm in ["f16"] + ARMS:
        want = "f16" if arm.startswith("f16") else "q8_0"
        ok = ("256", want) in kern[arm]
        verdict.append([f"validity: {arm} logs the compute 6.1 prompt attention kernel", "kernel",
                        "; ".join(f"head {h}, {t} K/V" for h, t in kern[arm]) or "none", f"head 256, {want} K/V",
                        "pass" if ok else "FAIL"])
        got = [float(x[0]) for x in faults[arm]]
        f = FACTOR.get(arm)
        if f:
            ok = len(got) == 1 and abs(got[0] - f) < 1e-6
            verdict.append([f"validity: {arm} logs the fault at its factor", "fault line",
                            ", ".join(x[0] for x in faults[arm]) or "none", f"output x {f:.2f} (as a float)",
                            "pass" if ok else "FAIL"])
        else:
            verdict.append([f"validity: {arm} runs without the fault", "fault line",
                            ", ".join(x[0] for x in faults[arm]) or "none", "none", "pass" if not got else "FAIL"])
    old_k = {r["arm"]: r for r in read_tsv(os.path.join(c989, "kld.tsv")) if r["pass"] == "g"}
    old_c = {}
    for r in read_tsv(os.path.join(c989, "chunks.tsv")):
        if r["pass"] == "g":
            old_c.setdefault(r["arm"], []).append((float(r["ln_ppl_ratio"]), float(r["kld"]), float(r["same_top_pct"])))
    for arm, old in (("q8", "q8"), ("q8f105", "q8f1")):
        if per.get(arm) and old_c.get(old):
            dmax = max(abs(x[i] - y[i]) for x, y in zip(per[arm], old_c[old]) for i in range(3))
            verdict.append([f"repeat: {arm} against C989's pass g {old}, every stretch", "ln, kld, top",
                            f"largest difference {dmax:.5f}", "<= 0.00002 (expected, not required)",
                            "same" if dmax <= 0.00002 else "differs"])
    r = parse(os.path.join(g, f"q8.{DEPTH}.log"))
    if "ppl_base" in r and "q8" in old_k:
        verdict.append(["repeat: the F16 reference's perplexity against C989's", "ppl_base",
                        f"{r['ppl_base'][0]} against {old_k['q8']['ppl_base']}", "equal (expected, not required)",
                        "same" if r["ppl_base"][0] == old_k["q8"]["ppl_base"] else "differs"])
    write(os.path.join(dest, "verdict.tsv"), ["check", "measure", "value", "rule", "result"], verdict)

    runs = []
    path = os.path.join(g, "runs.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            runs = [line.split("\t") for line in fh.read().splitlines()[1:]]
    write(os.path.join(dest, "runs.tsv"), ["arm", "depth", "ub", "exit", "start", "end"], runs)


if __name__ == "__main__":
    main()
