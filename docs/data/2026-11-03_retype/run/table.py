#!/usr/bin/env python3
# table.py: kld.sh's logs to one TSV row per file and text. Standard library only.
#   python3 -I table.py <label>=<log prefix> ... > kld.tsv
# A log prefix is <kld.sh OUT folder>/<variant name>; the logs are <prefix>.<text>.log. Columns: label, text, the mean
# KLD and its standard error, the 99th and 99.9th percentile KLD, the same-top-token share and its standard error, the
# RMS of the change in the top token's probability, and the perplexity of the file and of the unmodified file.
import os, re, sys

NUM = r"(-?[\d.]+(?:e[-+]?\d+)?)"
PATS = {
    "kld": rf"^Mean    KLD:\s*{NUM}\s*±\s*{NUM}",
    "kld99": rf"^99\.0%   KLD:\s*{NUM}",
    "kld999": rf"^99\.9%   KLD:\s*{NUM}",
    "top": rf"^Same top p:\s*{NUM}\s*±\s*{NUM}\s*%",
    "rmsdp": rf"^RMS Δp\s*:\s*{NUM}\s*±\s*{NUM}\s*%",
    "ppl": rf"^Mean PPL\(Q\)\s*:\s*{NUM}\s*±\s*{NUM}",
    "ppl_base": rf"^Mean PPL\(base\)\s*:\s*{NUM}\s*±\s*{NUM}",
}


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        for k, p in PATS.items():
            m = re.match(p, line)
            if m:
                out[k] = m.groups()
    return out


print("\t".join(["file", "text", "mean_kld", "kld_se", "kld_p99", "kld_p999", "same_top_pct", "same_top_se",
                 "rms_dp_pct", "ppl", "ppl_se", "ppl_base"]))
for arg in sys.argv[1:]:
    label, prefix = arg.split("=", 1)
    for t in ("wiki", "agent", "code"):
        r = parse(f"{prefix}.{t}.log")
        if "kld" not in r:
            continue
        print("\t".join([label, t, *r["kld"], r["kld99"][0], r["kld999"][0], *r["top"], r["rmsdp"][0], *r["ppl"],
                         r["ppl_base"][0]]))
