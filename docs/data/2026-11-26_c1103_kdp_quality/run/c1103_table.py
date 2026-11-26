#!/usr/bin/env python3
"""c1103_table.py: C1103's tables (Q1836) from run.sh's llama-perplexity logs, and the rule applied to them.

    python3 -I c1103_table.py <WORK> <dest folder>

Reads WORK/s/<arm>.8192.log (12 stretches of 8,192 tokens: kdp, q8g, q8u, q8f1, q8t and q8r scored against the
preset's own run, q8; q8's log for its kernel lines) and WORK/w/<arm>.49152.log (one window of 49,152 tokens: kdp,
q8g and q8u against q8). Writes kld.tsv, chunks.tsv, paired.tsv, window.tsv, verdict.tsv and runs.tsv to the dest
folder. The rule is the one README.md fixed before the run:
- per stretch i, K_i (kdp's KLD against q8), G_i (q8g's: the attention-only yardstick, C1032), U_i (q8u's: C994's
  yardstick, -ub 128), F_i (q8f1's: a 5 % error); kdp is within a yardstick when the mean of K_i - Y_i over the 12
  stretches is not 3 standard errors above zero (t < 3); the same for top-token disagreement (reported);
- perplexity: the mean of kdp's per-stretch ln PPL ratio within +-0.01, and the window's within +-0.01;
- the window: kdp's KLD at most the larger of twice each yardstick's and 0.003, its top-token disagreement at most the
  larger of twice each yardstick's and 0.3 points (C976's band B);
- validity: q8f1 - q8g seen (t >= 3) in KLD; q8r repeats q8; the kernel lines; one reference perplexity for every arm.
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
COLS = ["pass", "arm", "against", "chunks", "ctx", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se",
        "kld_max", "kld_p999", "kld_p99", "kld_p90", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio",
        "ln_ppl_ratio_se"]
SEEN_T = 3.0
FAULT1_SCALE = 0.05
PPL_BAND = 0.01
WIN_KLD_FLOOR, WIN_TOP_FLOOR = 0.003, 0.3
# earlier runs with TF32 on and cuBLAS in F16 (pascal-p106-v5 3dfad876f): C994's yardstick q8u and 5 % error q8f1,
# C1032's yardstick q8g, and C994's reference perplexity over the 12 stretches
EARLIER = {"q8u": 0.007963, "q8f1": 0.011030, "q8g": 0.006319}
EARLIER_PPL = "1.577859"
PASSES = {"s": (8192, 12, ("q8", "kdp", "q8g", "q8u", "q8f1", "q8t", "q8r")), "w": (49152, 1, ("q8", "kdp", "q8g", "q8u"))}
KERNEL = re.compile(r"compute-6\.1 prompt attention kernel: head (\d+), (\S+) K/V")
KDP = re.compile(r"compute-6\.1 prompt attention: q8_0 K scored in int8 with dp4a")
FAULT = re.compile(r"GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 \(tests of the tests\): output x (\S+)")
PREFIX = re.compile(r"^[0-9.]+ [IWED] ")
NAMES = {0: "ln_ppl_ratio", 1: "kld", 2: "top_differs_pct"}


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


def main():
    work, dest = sys.argv[1], sys.argv[2]
    summary, crow, per, logs, kern, kdpl, faultl, exists = [], [], {}, {}, {}, {}, {}, {}
    for p, (depth, nch, arms) in PASSES.items():
        for arm in arms:
            path = os.path.join(work, p, f"{arm}.{depth}.log")
            key = (p, arm)
            exists[key] = os.path.exists(path)
            kern[key] = sorted(set(scan(path, KERNEL)))
            kdpl[key] = len(scan(path, KDP))
            faultl[key] = sorted(set(x[0] for x in scan(path, FAULT)))
            if arm == "q8":
                continue  # the reference itself
            r = parse(path)
            logs[key] = r
            if "kld" not in r:
                continue
            summary.append([p, arm, "q8 (the preset's own run)", nch, depth, nch * (depth - 1 - depth // 2),
                            r["kld"][0], r["kld"][1], r["same_top"][0], r["same_top"][1], r["kld_max"][0],
                            r["kld_p999"][0], r["kld_p99"][0], r["kld_p90"][0], r["kld_median"][0], r["rms_dp"][0],
                            r["ppl"][0], r["ppl_base"][0], *r.get("ln_ppl_ratio", ("", ""))])
            if p == "s":
                per[arm] = [(ln, kl, 100.0 - top) for ln, kl, top in chunks(path)]
                for k, (ln, kl, td) in enumerate(per[arm], 1):
                    crow.append([arm, k, f"{ln:.5f}", f"{kl:.5f}", f"{100.0 - td:.3f}"])
    write(os.path.join(dest, "kld.tsv"), COLS, summary)
    write(os.path.join(dest, "chunks.tsv"), ["arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)

    prow, verdict = [], []

    def add(form, pair, idx, xs, judge=None):
        """judge: None (reported only), 'within' (pass when t < 3), 'seen' (pass when t >= 3)."""
        m, se, t = stats(xs)
        prow.append([form, pair, NAMES[idx], len(xs), f"{m:.6f}", f"{se:.6f}", f"{t:.2f}"])
        if judge == "within":
            verdict.append([f"{form}: {pair}", NAMES[idx], f"{m:.6f} ± {se:.6f} (t {t:.2f})", "t < 3 (not seen above "
                            "the yardstick)", "pass" if t < SEEN_T else "FAIL"])
        elif judge == "seen":
            verdict.append([f"{form}: {pair}", NAMES[idx], f"{m:.6f} ± {se:.6f} (t {t:.2f})", "t >= 3",
                            "pass" if t >= SEEN_T else "FAIL"])
        return m, se, t

    K, G, U, F, T, R = (per.get(a) for a in ("kdp", "q8g", "q8u", "q8f1", "q8t", "q8r"))
    for arm, xs in (("kdp", K), ("q8g", G), ("q8u", U), ("q8f1", F), ("q8t", T), ("q8r", R)):
        if xs:
            for idx in (1, 2, 0):
                add("alone", arm, idx, [x[idx] for x in xs])

    def pairs(a_name, a, y_name, y, judge_kld, judge_top):
        if not a or not y or len(a) != len(y):
            return
        add(f"against {y_name}", f"{a_name} - {y_name}", 1, [p[1] - q[1] for p, q in zip(a, y)], judge_kld)
        add(f"against {y_name}", f"{a_name} - {y_name}", 2, [p[2] - q[2] for p, q in zip(a, y)], judge_top)
        add(f"against {y_name}", f"{a_name} - {y_name}", 0, [p[0] - q[0] for p, q in zip(a, y)])
        for idx in (1, 2):
            if all(q[idx] > 0 for q in y):
                add(f"against {y_name}, divided by it", f"({a_name} - {y_name}) / {y_name}", idx,
                    [(p[idx] - q[idx]) / q[idx] for p, q in zip(a, y)])

    # the check: kdp against each yardstick (KLD decides, top-token disagreement reported by the same rule)
    pairs("kdp", K, "q8g", G, "within", None)
    pairs("kdp", K, "q8u", U, "within", None)
    # validity: the 5 % error against the attention-only yardstick, and against C994's
    pairs("q8f1", F, "q8g", G, "seen", None)
    pairs("q8f1", F, "q8u", U, None, None)
    # the kdp arm against the 5 % error, and the TF32 arm against the yardsticks (context)
    pairs("kdp", K, "q8f1", F, None, None)
    pairs("q8t", T, "q8g", G, None, None)
    pairs("q8t", T, "q8u", U, None, None)
    write(os.path.join(dest, "paired.tsv"), ["form", "arm or pair", "measure", "stretches", "mean", "se", "t"], prow)

    # top-token disagreement by the same rule, reported
    for y_name, y in (("q8g", G), ("q8u", U)):
        if K and y:
            m, se, t = stats([p[2] - q[2] for p, q in zip(K, y)])
            verdict.append([f"reported: kdp - {y_name}", "top_differs_pct", f"{m:.6f} ± {se:.6f} (t {t:.2f})",
                            "t < 3 (not seen above the yardstick)", "within" if t < SEEN_T else "above"])
    # perplexity over the 12 stretches
    if K:
        m, se, t = stats([x[0] for x in K])
        verdict.append(["kdp's per-stretch ln PPL ratio against q8, mean over the 12 stretches", "ln_ppl_ratio",
                        f"{m:+.6f} ± {se:.6f}", f"within ±{PPL_BAND:g}", "pass" if abs(m) <= PPL_BAND else "FAIL"])

    # the window of 49,152
    wrow = []
    wk = logs.get(("w", "kdp"), {})
    for arm in ("kdp", "q8g", "q8u"):
        r = logs.get(("w", arm), {})
        if "kld" in r:
            wrow.append([arm, r["kld"][0], r["kld"][1], f"{100 - float(r['same_top'][0]):.3f}", r["same_top"][1],
                         *r.get("ln_ppl_ratio", ("", "")), r["ppl"][0], r["ppl_base"][0]])
    write(os.path.join(dest, "window.tsv"), ["arm", "kld", "kld_se", "top_differs_pct", "top_se", "ln_ppl_ratio",
                                             "ln_ppl_ratio_se", "ppl", "ppl_base"], wrow)
    if "kld" in wk:
        kk, kt = float(wk["kld"][0]), 100 - float(wk["same_top"][0])
        for y in ("q8g", "q8u"):
            r = logs.get(("w", y), {})
            if "kld" not in r:
                verdict.append([f"window 49,152: kdp against {y}", "kld", "no yardstick run", "", "FAIL"])
                continue
            yk, yt = float(r["kld"][0]), 100 - float(r["same_top"][0])
            bk, bt = max(2 * yk, WIN_KLD_FLOOR), max(2 * yt, WIN_TOP_FLOOR)
            verdict.append([f"window 49,152: kdp's KLD against {y}'s", "kld", f"{kk:.6f} against {yk:.6f} "
                            f"({kk / yk:.3f} of it)", f"<= {bk:.6f} (max of 2 x {y}, {WIN_KLD_FLOOR})",
                            "pass" if kk <= bk else "FAIL"])
            verdict.append([f"window 49,152: kdp's top-token disagreement against {y}'s", "top_differs_pct",
                            f"{kt:.3f} against {yt:.3f}", f"<= {bt:.3f} (max of 2 x {y}, {WIN_TOP_FLOOR})",
                            "pass" if kt <= bt else "FAIL"])
        if "ln_ppl_ratio" in wk:
            ln = float(wk["ln_ppl_ratio"][0])
            verdict.append(["window 49,152: kdp's ln PPL ratio against q8", "ln_ppl_ratio",
                            f"{ln:+.6f} ± {wk['ln_ppl_ratio'][1]}", f"within ±{PPL_BAND:g}",
                            "pass" if abs(ln) <= PPL_BAND else "FAIL"])

    # sizes: kdp and the yardsticks as uniform attention output errors (est.), the smallest error seen (est.)
    if F:
        Fk = [x[1] for x in F]
        Ft = [x[2] for x in F]
        mf = sum(Fk) / len(Fk)
        for name, xs in (("kdp", K), ("q8g", G), ("q8u", U), ("q8t", T)):
            if xs:
                mx = sum(x[1] for x in xs) / len(xs)
                verdict.append([f"size: {name}'s KLD as a uniform attention output error, est. 0.05 * sqrt(mean "
                                f"{name} / mean q8f1)", "kld", f"{100 * FAULT1_SCALE * math.sqrt(max(mx, 0) / mf):.2f} %",
                                "", "est."])
        for y_name, y in (("q8g", G), ("q8u", U)):
            if not y:
                continue
            Y = [x[1] for x in y]
            Z = [x[2] for x in y]
            ests = [
                ("kld", lambda s, Y=Y: [s * a - b for a, b in zip(Fk, Y)], 2),
                ("top_differs_pct", lambda s, Z=Z: [s * a - b for a, b in zip(Ft, Z)], 1),
            ]
            if all(b > 0 for b in Y):
                ests.append(("kld, divided by the yardstick", lambda s, Y=Y: [(s * a - b) / b for a, b in zip(Fk, Y)], 2))
            for measure, fn, power in ests:
                eps = smallest_eps(fn, power)
                verdict.append([f"against {y_name}: smallest uniform attention error seen, est. (q8f1 scaled by "
                                f"(eps/0.05)^{power})", measure, "none up to 20 %" if eps is None else
                                f"{100 * eps:.2f} %", "", "est."])
        if K and G:
            verdict.append(["context: correlation over the stretches of kdp's KLD with q8g's", "kld",
                            f"{corr([x[1] for x in K], [x[1] for x in G]):.2f}", "", "context"])

    # TF32 and cuBLAS in F16 (q8t), and the yardsticks against the earlier runs' (TF32 on)
    if T:
        m, se, t = stats([x[1] for x in T])
        verdict.append(["tf32: q8t (TF32 on, cuBLAS in F16) against q8, per stretch", "kld",
                        f"{m:.6f} ± {se:.6f} (t {t:.2f})", "", "context"])
    for arm in ("q8g", "q8u", "q8f1"):
        r = logs.get(("s", arm), {})
        if "kld" in r:
            v = float(r["kld"][0])
            verdict.append([f"tf32: {arm}'s KLD over all tokens now, against the earlier run's with TF32 on "
                            f"({EARLIER[arm]:.6f})", "kld", f"{v:.6f} ({v / EARLIER[arm]:.3f} of it)", "", "context"])
    pb = logs.get(("s", "kdp"), {}).get("ppl_base")
    if pb:
        verdict.append(["tf32: the reference's perplexity over the 12 stretches, against C994's with TF32 on",
                        "ppl", f"{pb[0]} against {EARLIER_PPL}", "", "context"])

    # validity
    if R:
        okr = all(abs(x[1]) < 0.00001 and abs(x[2]) < 0.0005 for x in R)
        verdict.append(["validity: q8r (q8 loaded again) repeats q8 on every stretch", "kld, top",
                        f"largest stretch KLD {max(abs(x[1]) for x in R):.5f}, largest top differs "
                        f"{max(x[2] for x in R):.3f} %", "KLD 0.00000 and same top 100.000 % on every stretch",
                        "pass" if okr else "FAIL"])
    for p, (depth, nch, arms) in PASSES.items():
        bases = {a: logs.get((p, a), {}).get("ppl_base", ("",))[0] for a in arms if a != "q8"}
        vals = set(v for v in bases.values() if v)
        verdict.append([f"validity: pass {p}: every arm read the same reference (PPL(base))", "ppl",
                        "; ".join(f"{a} {v}" for a, v in bases.items()), "one value",
                        "pass" if len(vals) == 1 and all(bases.values()) else "FAIL"])
        for arm in arms:
            key = (p, arm)
            ks = kern.get(key, [])
            if arm == "q8g":
                ok = not ks and kdpl[key] == 0
                want = "no prompt kernel line"
            else:
                ok = ("256", "q8_0") in ks and (kdpl[key] > 0) == (arm == "kdp")
                want = "head 256, q8_0 K/V" + ("; the int8 line" if arm == "kdp" else "; no int8 line")
            got = ("; ".join(f"head {h}, {t} K/V" for h, t in ks) or "no prompt kernel line") + \
                  (f"; int8 line x{kdpl[key]}" if kdpl[key] else "; no int8 line")
            if arm == "q8f1":
                got += "; fault " + (", ".join(f"x {s}" for s in faultl[key]) or "not logged")
                ok = ok and len(faultl[key]) == 1 and abs(float(faultl[key][0]) - 1.05) < 1e-6
                want += "; fault x 1.05"
            if not exists[key]:
                ok, got = False, "no log"
            verdict.append([f"validity: pass {p}, {arm}'s log", "kernel", got, want, "pass" if ok else "FAIL"])
    write(os.path.join(dest, "verdict.tsv"), ["check", "measure", "value", "rule", "result"], verdict)

    runs = []
    for p in PASSES:
        path = os.path.join(work, p, "runs.tsv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                runs += [[p] + line.split("\t") for line in fh.read().splitlines()[1:]]
    write(os.path.join(dest, "runs.tsv"), ["pass", "arm", "depth", "ub", "exit", "start", "end"], runs)


if __name__ == "__main__":
    main()
