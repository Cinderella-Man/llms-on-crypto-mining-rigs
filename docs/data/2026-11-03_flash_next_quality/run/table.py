#!/usr/bin/env python3
"""The KL divergence logs of kld.sh to one TSV: per scored name and text, the mean KLD and its standard error, the
99th and 99.9th percentile KLD, the share of tokens with the same top token and its error, the RMS change of the top
token's probability, and the perplexity of the file and of the reference. Names in the order given (default: every
<name>.<text>.log in the folder that holds a result, base logs aside).
  python3 -I table.py <kld folder> [NAME...] > kld.tsv
"""
import glob, os, re, sys

TEXTS = ("wiki", "agent", "code")
FIELDS = [("kld", r"^Mean\s+KLD:\s+(\S+)\s+±\s+(\S+)"), ("kld_p99", r"^99\.0%\s+KLD:\s+(\S+)"),
          ("kld_p999", r"^99\.9%\s+KLD:\s+(\S+)"), ("same_top", r"^Same top p:\s+(\S+)\s+±\s+(\S+)"),
          ("rms_dp", r"^RMS Δp\s*:\s+(\S+)"), ("ppl", r"^Mean\s+PPL\(Q\)\s*:\s+(\S+)"),
          ("ppl_ref", r"^Mean\s+PPL\(base\)\s*:\s+(\S+)")]


def parse(path):
    text = open(path, errors="replace").read()
    row = {}
    for key, pat in FIELDS:
        m = re.search(pat, text, re.M)
        if m:
            row[key] = m.group(1)
            if m.lastindex > 1:
                row[key + "_se"] = m.group(2)
    return row


def main():
    folder, names = sys.argv[1], sys.argv[2:]
    if not names:
        seen = []
        for p in sorted(glob.glob(os.path.join(folder, "*.log"))):
            name, _, text = os.path.basename(p)[:-4].rpartition(".")
            if text in TEXTS and name not in ("base", "fit") and name not in seen and "kld" in parse(p):
                seen.append(name)
        names = seen
    cols = ["kld", "kld_se", "kld_p99", "kld_p999", "same_top", "same_top_se", "rms_dp", "ppl", "ppl_ref"]
    print("\t".join(["name", "text"] + cols))
    for name in names:
        for t in TEXTS:
            p = os.path.join(folder, f"{name}.{t}.log")
            if os.path.exists(p):
                row = parse(p)
                print("\t".join([name, t] + [row.get(c, "") for c in cols]))


if __name__ == "__main__":
    main()
