#!/usr/bin/env python3
"""c990_table.py: C990's tables (Q1723) from run.sh's llama-perplexity logs, and the rule README.md fixed before the run.

    python3 -I c990_table.py <WORK> <dest folder>

Reads WORK/p/<arm>.<depth>.log (f16 the reference; q8, q8k and f16d scored against it), WORK/p/runs.tsv and
WORK/mem.tsv. Writes kld.tsv (each arm's whole-run figures), chunks.tsv (each stretch), summary.tsv (means, standard
errors across stretches and t, for each arm and the rule's differences), validity.tsv, fit.tsv, runs.tsv and
verdict.tsv to the dest folder.
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
    "over": r"^\S+: (?:computing|calculating perplexity) over (\d+) chunks, n_ctx=(\d+)",
    "cache": r"^llama_kv_cache: size = .* K \((\S+)\):.* V \((\S+)\):",
    "prompt_kernel": r"^ggml_cuda_flash_attn_ext_tile_pascal: compute-6.1 prompt attention kernel: head (\d+), (\S+) K/V",
    "decode_kernel": r"^ggml_cuda_flash_attn_ext_tile_pascal: compute-6.1 decode attention kernel: head (\d+), (\S+) K/V, up to (\d+) rows",
}
COLS = ["arm", "depth", "chunks", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se", "kld_max",
        "kld_p999", "kld_p99", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio", "ln_ppl_ratio_se"]
ARMS = ("q8", "q8k", "f16d")
CACHE = {"f16": "f16", "q8": "q8_0", "q8k": "q8_0", "f16d": "f16"}
CHUNKS = 3
# the rule (README.md, "Rule", fixed before the run)
C976_32K, C976_32K_SE = 0.003490, 0.002556   # C976's q8 ln PPL ratio over four stretches of 32,768
T_LINE, BOUND = 3.0, 0.01


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = re.sub(r"^[0-9.]+ [IWED] ", "", line.rstrip("\n"))
            for k, p in PAT.items():
                # the reference's "Final estimate" can share a line with its progress output
                m = re.search(p.lstrip("^"), line) if k == "final" else re.match(p, line)
                if m and k not in out:
                    out[k] = m.groups()
    return out


def chunks(path):
    """Each chunk's own (ln PPL ratio, KLD, same top %) from llama-perplexity --kl-divergence's running-mean lines."""
    cum = []
    if not os.path.exists(path):
        return cum
    pat = re.compile(r"^\s*(\d+)\s+\S+ ±\s+\S+\s+(\S+) ±\s+\S+\s+(\S+) ±\s+\S+\s+\S+ ±\s+\S+ %\s+(\S+) ±\s+\S+ %")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = re.sub(r"^[0-9.]+ [IWED] ", "", line.rstrip("\n"))
            # chunk 1's line can follow the timing message on the same line ("... seconds per pass - ETA ...")
            line = re.sub(r"^.*seconds per pass - ETA(\s+\d+ hours)?(\s+[\d.]+ (minutes|seconds))?", "", line)
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
    return m, se, (m / se if se > 0 else float("nan"))


def fmt(m, se, t, d=6):
    return [f"{m:.{d}f}", f"{se:.{d}f}", f"{t:.2f}"]


def write(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def main():
    work, dest = sys.argv[1], sys.argv[2]
    p = os.path.join(work, "p")
    depths = sorted({int(m.group(2)) for x in os.listdir(p) for m in [re.match(r"(f16|q8|q8k|f16d)\.(\d+)\.log$", x)] if m},
                    reverse=True)
    runs = []
    if os.path.exists(os.path.join(p, "runs.tsv")):
        with open(os.path.join(p, "runs.tsv"), encoding="utf-8") as fh:
            runs = [line.split("\t") for line in fh.read().splitlines()[1:] if line]
    exits = {(r[0], int(r[1])): r[3] for r in runs}

    krows, crows, srows, vrows, verdict = [], [], [], [], []
    for d in depths:
        logs = {arm: parse(os.path.join(p, f"{arm}.{d}.log")) for arm in ("f16",) + ARMS}
        per = {arm: chunks(os.path.join(p, f"{arm}.{d}.log")) for arm in ARMS}

        # validity
        ok_all = True
        for arm, r in logs.items():
            if not r:
                vrows.append([d, arm, "missing", "", "", "", "", "FAIL"])
                ok_all = False
                continue
            over = r.get("over", ("", ""))
            cache = r.get("cache", ("", ""))
            pk = ",".join(r["prompt_kernel"]) if "prompt_kernel" in r else ""
            dk = ",".join(r["decode_kernel"]) if "decode_kernel" in r else ""
            ok = exits.get((arm, d)) == "0" and over == (str(CHUNKS), str(d)) and cache == (CACHE[arm], CACHE[arm])
            ok = ok and (("final" in r) if arm == "f16" else ("kld" in r and len(per[arm]) == CHUNKS))
            if arm == "q8":
                ok = ok and pk == "256,q8_0"
            if arm == "q8k":
                ok = ok and dk == "256,q8_0,4"
            if arm == "f16d":
                ok = ok and dk == ""
            ok_all = ok_all and ok
            vrows.append([d, arm, exits.get((arm, d), ""), "/".join(over), "/".join(cache), pk, dk, "ok" if ok else "FAIL"])

        for arm in ARMS:
            r = logs[arm]
            if "kld" in r:
                scored = CHUNKS * (d - 1 - d // 2)
                krows.append([arm, d, CHUNKS, scored, r["kld"][0], r["kld"][1], r["same_top"][0], r["same_top"][1],
                              r["kld_max"][0], r["kld_p999"][0], r["kld_p99"][0], r["kld_median"][0], r["rms_dp"][0],
                              r["ppl"][0], r["ppl_base"][0], *r.get("ln_ppl_ratio", ("", ""))])
            for k, (ln, kl, top) in enumerate(per[arm], 1):
                crows.append([d, arm, k, f"{ln:.5f}", f"{kl:.5f}", f"{top:.3f}", f"{100 - top:.3f}"])

        def series(arm):
            return [(ln, kl, 100 - top) for ln, kl, top in per[arm]] if len(per[arm]) == CHUNKS else None

        s = {arm: series(arm) for arm in ARMS}
        for arm in ARMS:
            if s[arm]:
                srows.append([d, arm, CHUNKS] + sum((fmt(*stats([x[i] for x in s[arm]]), 6 if i < 2 else 3)
                                                     for i in range(3)), []))
        if s["q8k"] and s["f16d"]:
            dd = [tuple(a[i] - b[i] for i in range(3)) for a, b in zip(s["q8k"], s["f16d"])]
            srows.append([d, "q8k-f16d", CHUNKS] + sum((fmt(*stats([x[i] for x in dd]), 6 if i < 2 else 3)
                                                         for i in range(3)), []))
            if s["q8"]:
                dl = [tuple(x[i] - q[i] for i in range(3)) for x, q in zip(dd, s["q8"])]
                srows.append([d, "(q8k-f16d)-q8", CHUNKS] + sum((fmt(*stats([x[i] for x in dl]), 6 if i < 2 else 3)
                                                                  for i in range(3)), []))

        tag = "" if ok_all else " (validity failed)"
        if s["q8"]:
            m, se, t = stats([x[0] for x in s["q8"]])
            tg = (m - C976_32K) / math.hypot(se, C976_32K_SE)
            verdict.append([f"1 prompt kernel: q8 ln PPL ratio against C976's 32,768{tag}", d,
                            f"{m:.6f} ± {se:.6f}", f"t {tg:.2f}", "grows with the context" if tg >= T_LINE
                            else "not measured growing"])
            verdict.append([f"1 prompt kernel: q8 within 1 %{tag}", d, f"{m:.6f} + 3 × {se:.6f} = {m + 3 * se:.6f}",
                            f"<= {BOUND}", "within 1 %" if m + 3 * se <= BOUND else "not shown within 1 %"])
        if s["q8"] and s["q8k"] and s["f16d"]:
            dd = [tuple(a[i] - b[i] - q[i] for i in range(3)) for a, b, q in zip(s["q8k"], s["f16d"], s["q8"])]
            for i, name, two in ((0, "ln PPL ratio", True), (1, "KLD", False), (2, "top-token disagreement", False)):
                m, se, t = stats([x[i] for x in dd])
                seen = abs(t) >= T_LINE if two else t >= T_LINE
                verdict.append([f"2 decode kernel: (q8k - f16d) - q8, {name}{tag}", d, f"{m:.6f} ± {se:.6f}",
                                f"t {t:.2f}", "adds an error" if seen else "no added error seen"])
            m, se, t = stats([a[0] - b[0] for a, b in zip(s["q8k"], s["f16d"])])
            verdict.append([f"2 decode path: q8k - f16d ln PPL ratio within 1 %{tag}", d,
                            f"{m:.6f} + 3 × {se:.6f} = {m + 3 * se:.6f}", f"<= {BOUND}",
                            "within 1 %" if m + 3 * se <= BOUND else "not shown within 1 %"])
            # added after the run: rule 2's KLD and top-token parts subtract q8's divergence (from an unperturbed
            # reference) from a difference taken on top of the decode path's own divergence; divergences on the same
            # tokens do not add, so those parts judge nothing. q8k - f16d itself has the form of C993's floor (an arm
            # on top of a perturbed run, both against F16): a 1 % attention error there was +0.000500 KLD per stretch.
            for i, name in ((1, "KLD"), (2, "top-token disagreement")):
                m, se, t = stats([a[i] - b[i] for a, b in zip(s["q8k"], s["f16d"])])
                verdict.append([f"added after the run: q8k - f16d, {name} (C993's floor: +0.000500 KLD, t 3.18, for a "
                                f"1 % attention error){tag}", d, f"{m:.6f} ± {se:.6f}", f"t {t:.2f}", "reported"])

    write(os.path.join(dest, "kld.tsv"), COLS, krows)
    write(os.path.join(dest, "chunks.tsv"), ["depth", "arm", "chunk", "ln_ppl_ratio", "kld", "same_top_pct",
                                             "top_differs_pct"], crows)
    write(os.path.join(dest, "summary.tsv"), ["depth", "arm_or_difference", "chunks", "ln_ppl_ratio_mean", "ln_ppl_ratio_se",
                                              "ln_ppl_ratio_t", "kld_mean", "kld_se", "kld_t", "top_differs_mean",
                                              "top_differs_se", "top_differs_t"], srows)
    write(os.path.join(dest, "validity.tsv"), ["depth", "arm", "exit", "chunks/n_ctx", "cache K/V", "prompt kernel",
                                               "decode kernel", "result"], vrows)
    write(os.path.join(dest, "verdict.tsv"), ["check", "depth", "value", "line", "result"], verdict)
    write(os.path.join(dest, "runs.tsv"), ["arm", "depth", "ub", "exit", "start", "end"], runs)

    # the fit: lowest host memory available and each card's highest use while the run lasted
    mem = os.path.join(work, "mem.tsv")
    if os.path.exists(mem):
        with open(mem, encoding="utf-8") as fh:
            rows = [line.split("\t") for line in fh.read().splitlines()[1:] if line.count("\t") == 3]
        avail = [int(r[1]) for r in rows if r[1].isdigit()]
        swap = [int(r[2]) for r in rows if r[2].isdigit()]
        gpus = [[int(x) for x in r[3].split(",") if x.isdigit()] for r in rows]
        gpus = [g for g in gpus if g]
        fit = [["whole run", "", len(rows), f"{min(avail) / 2**20:.2f}" if avail else "",
                f"{min(swap) / 2**20:.2f}" if swap else "",
                ",".join(str(max(g[i] for g in gpus if len(g) > i)) for i in range(len(gpus[0]))) if gpus else ""]]
        # per run: the samples between its start and end (both ISO times in the host's zone, so they sort as text)
        for r in runs:
            if len(r) < 6:
                continue
            sel = [x for x in rows if r[4] <= x[0] <= r[5]]
            a = [int(x[1]) for x in sel if x[1].isdigit()]
            sw = [int(x[2]) for x in sel if x[2].isdigit()]
            g = [[int(v) for v in x[3].split(",") if v.isdigit()] for x in sel]
            g = [v for v in g if v]
            fit.append([r[0], r[1], len(sel), f"{min(a) / 2**20:.2f}" if a else "", f"{min(sw) / 2**20:.2f}" if sw else "",
                        ",".join(str(max(v[i] for v in g if len(v) > i)) for i in range(len(g[0]))) if g else ""])
        write(os.path.join(dest, "fit.tsv"), ["arm", "depth", "samples", "lowest_mem_available_gib",
                                              "lowest_swap_free_gib", "highest_used_mib_per_card"], fit)


if __name__ == "__main__":
    main()
