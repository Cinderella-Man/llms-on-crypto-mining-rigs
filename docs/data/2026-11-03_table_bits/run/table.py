#!/usr/bin/env python3
# table.py: kld.sh's logs and the round trips' error tables to one TSV row per variant and text. Standard library only.
#   python3 -I table.py <kld.sh OUT folder> <variants folder> FORMAT... > kld.tsv
# Columns: format, the type the table was written back in, bits per weight and table size of the format's own row
# layout (the 16-bit row scale included; the per-head parameters, at most 0.66 MB, not), the round trip's relative
# squared error (format alone, and after the re-encode: what the model reads), then per text the mean KLD and its standard error, the 99th and 99.9th
# percentile KLD, the same-top-token share and its standard error, RMS of the change in the top token's probability,
# and the perplexity of the variant and of the unmodified file.
import os, re, sys

ROWS = 320001536  # the table's rows, padding included
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


def row_bits(fmt):
    fmt = fmt.removesuffix("-q8")  # written back as Q8_0: the format's own size is reported
    if fmt in ("iq4nl", "self", "ub256"):  # the IQ4_NL table: re-encoded, or unmodified
        return 90 * 8
    if fmt == "zero":
        return 0
    if fmt.startswith("bf"):
        return int(fmt[2:]) + 16
    if fmt.startswith("q"):
        return 160 * int(fmt[1:]) + 16
    raise ValueError(fmt)


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


def rt_errors(path):
    if not os.path.exists(path):
        return "", ""
    for line in open(path):
        f = line.split("\t")
        if f[0] == "all":
            return f[2], f[4]
    return "", ""


kld_dir, var_dir, fmts = sys.argv[1], sys.argv[2], sys.argv[3:]
print("\t".join(["format", "written_as", "bits_per_weight", "table_GB", "format_rel_err", "written_rel_err", "text",
                 "mean_kld", "kld_se", "kld_p99", "kld_p999", "same_top_pct", "same_top_se", "rms_dp_pct", "ppl",
                 "ppl_base"]))
for f in fmts:
    b = row_bits(f)
    fe, te = rt_errors(os.path.join(var_dir, f, "roundtrip.tsv"))
    if f in ("self", "ub256"):
        fe = te = "0"
    w = "Q8_0" if f.endswith("-q8") else "IQ4_NL"
    for t in ("wiki", "agent", "code"):
        r = parse(os.path.join(kld_dir, f"{f}.{t}.log"))
        if "kld" not in r:
            continue
        print("\t".join([f, w, f"{b / 160:.3f}", f"{ROWS * b / 8 / 1e9:.2f}", fe, te, t, r["kld"][0], r["kld"][1],
                         r["kld99"][0], r["kld999"][0], r["top"][0], r["top"][1], r["rmsdp"][0], r["ppl"][0],
                         r["ppl_base"][0]]))
