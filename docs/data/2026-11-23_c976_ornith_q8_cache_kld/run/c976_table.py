#!/usr/bin/env python3
"""c976_table.py: C976's tables (Q1707) from run.sh's llama-perplexity logs, and the pass band applied to them.

    python3 -I c976_table.py <WORK> <dest folder>

Reads WORK/a/<arm>.8192.log (both caches against Ornith-1.5-122B Q8_0's reference), WORK/b/<arm>.<depth>.log (against
the F16 cache, prompt micro-batches) and WORK/c/<arm>.8192.log (against the F16 cache, one token a pass). Writes
against_q8_model.tsv, kld.tsv, decode.tsv and verdict.tsv to the dest folder. The band is the one README.md fixed
before the run.
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
    "kld_median": r"^Median  KLD:\s+(\S+)",
    "rms_dp": r"^RMS Δp    :\s+(\S+) ±\s+(\S+) %",
    "ppl": r"^Mean PPL\(Q\)\s+:\s+(\S+) ±\s+(\S+)",
    "ppl_base": r"^Mean PPL\(base\)\s+:\s+(\S+) ±\s+(\S+)",
    "ln_ppl_ratio": r"^Mean ln\(PPL\(Q\)/PPL\(base\)\)\s+:\s+(\S+) ±\s+(\S+)",
    "final": r"^Final estimate: PPL = (\S+) \+/- (\S+)",
}
COLS = ["arm", "depth", "chunks", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se", "kld_max",
        "kld_p999", "kld_p99", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio", "ln_ppl_ratio_se"]

# the band (README.md, "Pass band", fixed before the run)
A_KLD, A_TOP = 0.003, 0.3          # (A): Q8_0 cache's KLD rise and same-top fall against the F16 cache's
B_FLOOR_KLD, B_FLOOR_TOP = 0.003, 0.3
B_FACTOR, B_LNPPL = 2.0, 0.01
FLOOR_KLD = 0.0001


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = re.sub(r"^[0-9.]+ [IWE] ", "", line.rstrip("\n"))
            for k, p in PAT.items():
                m = re.match(p, line)
                if m:
                    out[k] = m.groups()
    return out


def row(arm, depth, chunks, r):
    scored = chunks * (depth - 1 - depth // 2)
    return [arm, str(depth), str(chunks), str(scored), r["kld"][0], r["kld"][1], r["same_top"][0], r["same_top"][1],
            r["kld_max"][0], r["kld_p999"][0], r["kld_p99"][0], r["kld_median"][0], r["rms_dp"][0], r["ppl"][0],
            r["ppl_base"][0], *r.get("ln_ppl_ratio", ("", ""))]


def write(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def f(r, k, i=0):
    return float(r[k][i])


def check_b(name, depth, r, y, verdict):
    """(B)'s rule: arm r against the reference, yardstick y on the same tokens."""
    if not r or not y:
        verdict.append([name, str(depth), "missing", "", "", ""])
        return
    kb = max(B_FACTOR * f(y, "kld"), B_FLOOR_KLD)
    tb = max(B_FACTOR * (100 - f(y, "same_top")), B_FLOOR_TOP)
    ln = f(r, "ln_ppl_ratio") if "ln_ppl_ratio" in r else float("nan")
    for what, val, band, ok in [
            ("kld", f(r, "kld"), kb, f(r, "kld") <= kb),
            ("top_differs_pct", 100 - f(r, "same_top"), tb, 100 - f(r, "same_top") <= tb),
            ("ln_ppl_ratio", ln, B_LNPPL, abs(ln) <= B_LNPPL)]:
        verdict.append([name, str(depth), what, f"{val:.6f}", f"{band:.6f}", "pass" if ok else "FAIL"])


def main():
    work, dest = sys.argv[1], sys.argv[2]
    verdict = []

    # (A) against Ornith-1.5-122B Q8_0's reference, 12 chunks of 8,192
    a = {arm: parse(os.path.join(work, "a", f"{arm}.8192.log")) for arm in ("f16", "q8")}
    rows = [row(arm, 8192, 12, r) for arm, r in a.items() if "kld" in r]
    if all("kld" in r for r in a.values()):
        dk = f(a["q8"], "kld") - f(a["f16"], "kld")
        dt = f(a["q8"], "same_top") - f(a["f16"], "same_top")
        sk = math.hypot(f(a["q8"], "kld", 1), f(a["f16"], "kld", 1))
        st = math.hypot(f(a["q8"], "same_top", 1), f(a["f16"], "same_top", 1))
        dl = math.log(f(a["q8"], "ppl")) - math.log(f(a["f16"], "ppl"))
        rows.append(["q8-f16", "8192", "12", "", f"{dk:.6f}", f"{sk:.6f}", f"{dt:.3f}", f"{st:.3f}",
                     "", "", "", "", "", "", "", f"{dl:.6f}", ""])
        verdict.append(["A q8 vs f16 against the Q8_0 model", "8192x12", "kld_rise", f"{dk:.6f}", f"{A_KLD:.6f}",
                        "pass" if dk <= A_KLD else "FAIL"])
        verdict.append(["A q8 vs f16 against the Q8_0 model", "8192x12", "same_top_fall_pct", f"{-dt:.3f}",
                        f"{A_TOP:.3f}", "pass" if -dt <= A_TOP else "FAIL"])
    write(os.path.join(dest, "against_q8_model.tsv"), COLS, rows)

    # (B) against the F16 cache, prompt micro-batches of 512 (the yardstick f16u: micro-batches of 128)
    bdir = os.path.join(work, "b")
    depths = sorted({int(m.group(1)) for x in os.listdir(bdir) for m in [re.match(r"f16\.(\d+)\.log$", x)] if m})
    rows = []
    for d in depths:
        r = {arm: parse(os.path.join(bdir, f"{arm}.{d}.log")) for arm in ("f16b", "q8", "f16u", "q8f")}
        rows += [row(arm, d, 1, x) for arm, x in r.items() if "kld" in x]
        if "kld" in r["f16b"]:
            ok = f(r["f16b"], "kld") <= FLOOR_KLD and f(r["f16b"], "same_top") >= 100.0
            verdict.append(["validity: f16 second load repeats", str(d), "kld / same_top",
                            f"{f(r['f16b'], 'kld'):.6f} / {f(r['f16b'], 'same_top'):.3f}",
                            f"{FLOOR_KLD} / 100", "pass" if ok else "FAIL"])
        check_b("B q8 vs f16 (prompt, -ub 512)", d, r["q8"] if "kld" in r["q8"] else None,
                r["f16u"] if "kld" in r["f16u"] else None, verdict)
        if "kld" in r["q8f"] and "kld" in r["f16u"]:
            kb = max(B_FACTOR * f(r["f16u"], "kld"), B_FLOOR_KLD)
            verdict.append(["validity: fault x1.05 seen", str(d), "kld", f"{f(r['q8f'], 'kld'):.6f}",
                            f"> {kb:.6f}", "pass" if f(r["q8f"], "kld") > kb else "FAIL"])
    write(os.path.join(dest, "kld.tsv"), COLS, rows)

    # (C) the decode path: one token a pass, against the F16 cache in one-token passes (yardstick f16p: -ub 512)
    c = {arm: parse(os.path.join(work, "c", f"{arm}.8192.log")) for arm in ("q8d", "f16p", "q8k")}
    v = {arm: parse(os.path.join(work, "v", f"{arm}.512.log")) for arm in ("q8kv", "q8vd", "f16vp")}
    write(os.path.join(dest, "decode.tsv"), COLS, [row(arm, 8192, 1, x) for arm, x in c.items() if "kld" in x]
          + [row(arm, 512, 1, x) for arm, x in v.items() if "kld" in x])
    for arm, name in (("q8d", "C q8 vs f16 (decode, base kernels)"), ("q8k", "C q8 vs f16 (decode, PR 9 kernel)")):
        check_b(name, 8192, c[arm] if "kld" in c[arm] else None, c["f16p"] if "kld" in c["f16p"] else None, verdict)

    # per chunk (llama-perplexity prints running means over equal chunks, so chunk k's own mean is k*M_k - (k-1)*M_k-1):
    # pass a (12 chunks of 8,192 against the Q8_0 model), e (6 of 16,384) and f (4 of 32,768) against the F16 cache, each
    # chunk a repeat on its own stretch of text, so the spread across chunks is an error that holds for correlated tokens
    crow, srow = [], []
    for p, d, arms, ref in (("a", 8192, ("f16", "q8"), "f16"), ("e", 16384, ("q8", "f16u"), None), ("f", 32768, ("q8",), None)):
        per = {arm: chunks(os.path.join(work, p, f"{arm}.{d}.log")) for arm in arms}
        if not all(per.values()):
            continue
        for arm, rows_ in per.items():
            for k, (ln, kl, top) in enumerate(rows_, 1):
                crow.append([p, str(d), arm, str(k), f"{ln:.5f}", f"{kl:.5f}", f"{top:.3f}"])
            srow.append([p, str(d), arm, str(len(rows_))] + summary([x[0] for x in rows_]) + summary([x[1] for x in rows_]))
        if ref:
            other = [a for a in arms if a != ref][0]
            n = min(len(per[ref]), len(per[other]))
            dln = [per[other][i][0] - per[ref][i][0] for i in range(n)]
            dkl = [per[other][i][1] - per[ref][i][1] for i in range(n)]
            srow.append([p, str(d), f"{other}-{ref}", str(n)] + summary(dln) + summary(dkl))
        if p in ("e", "f"):
            m, se = summary([x[0] for x in per["q8"]])[:2]
            verdict.append([f"added: q8 vs f16 on {len(per['q8'])} chunks of {d:,} (mean of chunks)", f"{d}x{len(per['q8'])}", "ln_ppl_ratio",
                            f"{float(m):.6f} ± {float(se):.6f}", f"{B_LNPPL:.6f}",
                            "pass" if abs(float(m)) <= B_LNPPL else "FAIL"])
    write(os.path.join(dest, "chunks.tsv"), ["pass", "depth", "arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct"], crow)
    write(os.path.join(dest, "chunks_summary.tsv"), ["pass", "depth", "arm", "chunks", "ln_ppl_ratio_mean",
                                                     "ln_ppl_ratio_se_across_chunks", "ln_ppl_ratio_t", "kld_mean",
                                                     "kld_se_across_chunks", "kld_t"], srow)

    write(os.path.join(dest, "verdict.tsv"), ["check", "depth", "measure", "value", "band", "result"], verdict)

    # every run, one table (folder b holds passes b, b2 and d)
    runs = []
    for p in ("a", "b", "c", "e", "v", "f"):
        path = os.path.join(work, p, "runs.tsv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh.read().splitlines()[1:]:
                arm, depth = line.split("\t")[:2]
                label = p if p != "b" else ("b2" if arm == "q8f" else ("d" if depth in ("16384", "32768") else "b"))
                runs.append([label] + line.split("\t"))
    write(os.path.join(dest, "runs.tsv"), ["pass", "arm", "depth", "ub", "exit", "start", "end"], runs)


def chunks(path):
    """Each chunk's own (ln PPL ratio, KLD, same top %) from llama-perplexity --kl-divergence's running-mean lines."""
    cum = []
    if not os.path.exists(path):
        return cum
    pat = re.compile(r"^\s*(\d+)\s+\S+ ±\s+\S+\s+(\S+) ±\s+\S+\s+(\S+) ±\s+\S+\s+\S+ ±\s+\S+ %\s+(\S+) ±\s+\S+ %")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # chunk 1's line can follow the timing message on the same line ("... seconds per pass - ETA    1 ...")
            line = re.sub(r"^.*seconds per pass - ETA(\s+[\d.]+ (minutes|seconds|hours))?", "", line.rstrip("\n"))
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


def summary(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    return [f"{m:.6f}", f"{se:.6f}", f"{m / se:.2f}" if se > 0 else ""]


if __name__ == "__main__":
    main()
