#!/usr/bin/env python3
"""Per-chunk table and final statistics of llama-perplexity --kl-divergence logs (scripts/quality_kld.sh).

usage: kld_chunks.py OUT_DIR LOG...
Writes OUT_DIR/kld_by_chunk.tsv (one row per file and chunk, the values llama-perplexity prints after each chunk:
running means over chunks 1..N with their standard errors) and OUT_DIR/kld_summary.csv (one row per file).
"""
import csv
import os
import re
import sys

NUM = r"(-?[0-9.]+)"
ROW = re.compile(r"^\s*(\d+)\s+" + r"\s+".join([NUM + r"\s+±\s+" + NUM] * 3) + r"\s+" + NUM + r"\s+±\s+" + NUM +
                 r"\s+%\s+" + NUM + r"\s+±\s+" + NUM + r"\s+%\s*$")
STATS = [("Mean PPL(Q)", "ppl_q"), ("Mean PPL(base)", "ppl_base"), ("Mean PPL(Q)/PPL(base)", "ppl_ratio"),
         ("Mean ln(PPL(Q)/PPL(base))", "ln_ppl_ratio"), ("Mean    KLD", "kld_mean"), ("Maximum KLD", "kld_max"),
         ("99.9%   KLD", "kld_p99_9"), ("99.0%   KLD", "kld_p99"), ("95.0%   KLD", "kld_p95"),
         ("90.0%   KLD", "kld_p90"), ("Median  KLD", "kld_median"), ("10.0%   KLD", "kld_p10"),
         ("RMS Δp", "dp_rms_pct"), ("Same top p", "same_top_pct")]


def main():
    out = sys.argv[1]
    chunks, summary = [], []
    for p in sys.argv[2:]:
        name = os.path.basename(p).replace(".kld.log", "")
        stats = {"file": name}
        for line in open(p, encoding="utf-8"):
            m = ROW.match(line)
            if m:
                chunks.append([name] + list(m.groups()))
                continue
            for key, col in STATS:
                if ":" in line and line.split(":", 1)[0].strip() == key:
                    vals = re.findall(r"-?[0-9]+\.[0-9]+", line.split(":", 1)[1])
                    stats[col] = vals[0]
                    if len(vals) > 1 and col in ("ppl_q", "ppl_base", "ppl_ratio", "kld_mean", "same_top_pct",
                                                 "dp_rms_pct", "ln_ppl_ratio"):
                        stats[col + "_se"] = vals[1]
        stats["chunks"] = sum(1 for c in chunks if c[0] == name)
        summary.append(stats)
    with open(os.path.join(out, "kld_by_chunk.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["file", "chunk", "ppl", "ppl_se", "ln_ppl_ratio", "ln_ppl_ratio_se", "kld", "kld_se",
                    "dp_rms_pct", "dp_rms_se", "same_top_pct", "same_top_se"])
        w.writerows(chunks)
    cols = ["file", "chunks"]
    for _, col in STATS:
        cols.append(col)
        if col in ("ppl_q", "ppl_base", "ppl_ratio", "kld_mean", "same_top_pct", "dp_rms_pct", "ln_ppl_ratio"):
            cols.append(col + "_se")
    with open(os.path.join(out, "kld_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(len(chunks), "chunk rows,", len(summary), "files")


if __name__ == "__main__":
    main()
