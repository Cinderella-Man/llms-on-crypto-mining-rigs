#!/usr/bin/env python3
"""kld_table.py: the table of kld_depth.sh's runs (Q1248).

    python3 -I kld_table.py <OUT> <base arm> <floor arm> <arm> [<arm> ...] > kld.tsv

Reads OUT/<arm>.<depth>.log (llama-perplexity --kl-divergence) and OUT/<base arm>.<depth>.log (the reference run),
for every depth that has logs. Writes one TSV line per arm and depth: the scored tokens, mean KLD and its standard
error, same top token (%) and its standard error, the KLD's maximum, 99.9th and 99th percentiles and median, RMS of
the probability difference, the perplexity of the arm and of the reference, the mean log-ratio of the two perplexities
and its standard error, and the arm's KLD and same-top minus the
floor arm's (the reference's own second load) with that difference in units of the two standard errors combined.
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
}


def parse(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = re.sub(r"^[0-9.]+ [IWE] ", "", line.rstrip("\n"))
            for k, p in PAT.items():
                m = re.match(p, line)
                if m:
                    out[k] = m.groups()
    return out


def main():
    out_dir, base_arm, floor_arm, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    depths = sorted({int(m.group(1)) for f in os.listdir(out_dir)
                     for m in [re.match(re.escape(base_arm) + r"\.(\d+)\.log$", f)] if m})
    cols = ["arm", "depth", "scored_positions", "scored_tokens", "kld", "kld_se", "same_top_pct", "same_top_se",
            "kld_max", "kld_p999", "kld_p99", "kld_median", "rms_dp_pct", "ppl", "ppl_base", "ln_ppl_ratio",
            "ln_ppl_ratio_se", "kld_minus_floor", "kld_minus_floor_in_se", "same_top_minus_floor", "same_top_minus_floor_in_se"]
    print("\t".join(cols))
    for d in depths:
        rows = {}
        for arm in [floor_arm] + [a for a in arms if a != floor_arm]:
            p = os.path.join(out_dir, f"{arm}.{d}.log")
            if os.path.exists(p):
                r = parse(p)
                if "kld" in r:
                    rows[arm] = r
        fl = rows.get(floor_arm)
        for arm in arms:
            r = rows.get(arm)
            if r is None:
                continue
            first = d // 2
            line = [arm, str(d), f"{first}-{d - 1}", str(d - 1 - first), r["kld"][0], r["kld"][1],
                    r["same_top"][0], r["same_top"][1], r["kld_max"][0], r["kld_p999"][0], r["kld_p99"][0],
                    r["kld_median"][0], r["rms_dp"][0], r["ppl"][0], r["ppl_base"][0],
                    *r.get("ln_ppl_ratio", ("", ""))]
            if fl is not None and arm != floor_arm:
                dk = float(r["kld"][0]) - float(fl["kld"][0])
                sk = math.hypot(float(r["kld"][1]), float(fl["kld"][1]))
                dt = float(r["same_top"][0]) - float(fl["same_top"][0])
                st = math.hypot(float(r["same_top"][1]), float(fl["same_top"][1]))
                line += [f"{dk:.6f}", f"{dk / sk:.2f}" if sk > 0 else "",
                         f"{dt:.3f}", f"{dt / st:.2f}" if st > 0 else ""]
            else:
                line += ["", "", "", ""]
            print("\t".join(line))


if __name__ == "__main__":
    main()
