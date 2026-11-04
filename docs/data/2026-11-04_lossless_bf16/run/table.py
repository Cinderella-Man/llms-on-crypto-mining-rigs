#!/usr/bin/env python3
# table.py: the entropy table of Q621 from lossless_entropy's per-tensor TSVs (one per file, named tensors_<file>.tsv):
# per file and shape class, the weights, the entropies in bits a weight, the codes' bits a weight and escapes, and the
# megabytes a token reads as stored (BF16, or F32), in x12, x11 and at the 16-bit entropy (every class is read once per token per tensor;
# calls per token are the tensor counts). Standard library only.
#   python3 -I table.py tensors_<file>.tsv ... > entropy.md
import os, re, sys
from collections import OrderedDict


def cls(name):
    n = re.sub(r"^blk\.\d+\.", "", name)
    for pat, c in [(r"hc_(attn|ffn)_up|output_hc_up", "hc_up"), (r"hc_(attn|ffn)_down|output_hc_down", "hc_down"),
                   (r"hc_(attn|ffn)_inject", "hc_inject"), (r"indexer\.q_proj", "indexer_q"),
                   (r"indexer\.k_proj", "indexer_k"), (r"ffn_gate_inp\.weight", "router"),
                   (r"ssm_(alpha|beta)", "ssm_ab"), (r"ple_value", "ple_value")]:
        if re.fullmatch(r"(?:" + pat + r")(?:\.weight)?", n):
            return c
    return n


COLS = ["H_hi", "H_lo", "H_exp", "H_signmant", "H_u16", "H_exp_row", "H_hi_row", "huff_hi", "x12_bits", "x11_bits"]
print("| File | Class | Type | Tensors | Weights (M) | H hi | H lo | H exp | H sign+mant | H 16 bits | H exp per row | "
      "H hi per row | Huffman hi | x12 bits | x12 escapes | x11 bits | x11 escapes | MB as stored | MB x12 | MB x11 | MB at H 16 | Round trip |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
hashes = {}
for path in sys.argv[1:]:
    fname = re.sub(r"^tensors_|\.tsv$", "", os.path.basename(path))
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    head, rows = rows[0], rows[1:]
    ix = {c: i for i, c in enumerate(head)}
    groups = OrderedDict()
    for r in rows:
        hashes.setdefault(r[ix["tensor"]], set()).add(r[ix["fnv1a"]])
        groups.setdefault((cls(r[ix["tensor"]]), r[ix["type"]]), []).append(r)
    tot = {"n": 0, "bf16": 0.0, "x12": 0.0, "x11": 0.0, "h": 0.0}
    for (c, typ), rs in sorted(groups.items(), key=lambda kv: -sum(int(r[ix["weights"]]) for r in kv[1])):
        n = sum(int(r[ix["weights"]]) for r in rs)
        mean = {k: sum(float(r[ix[k]]) * int(r[ix["weights"]]) for r in rs) / n for k in COLS}
        e12 = sum(int(r[ix["x12_esc"]]) for r in rs) / n
        e11 = sum(int(r[ix["x11_esc"]]) for r in rs) / n
        checks = sorted(set(r[ix["check"]] for r in rs))
        mb = lambda bits: n * bits / 8e6
        tot["n"] += n
        stored = mb(32 if typ == "F32" else 16)
        tot["bf16"] += stored
        tot["x12"] += mb(mean["x12_bits"])
        tot["x11"] += mb(mean["x11_bits"])
        tot["h"] += mb(mean["H_u16"])
        print(f"| {fname} | {c} | {typ} | {len(rs)} | {n / 1e6:.2f} | {mean['H_hi']:.3f} | {mean['H_lo']:.3f} | "
              f"{mean['H_exp']:.3f} | {mean['H_signmant']:.3f} | {mean['H_u16']:.3f} | {mean['H_exp_row']:.3f} | "
              f"{mean['H_hi_row']:.3f} | {mean['huff_hi'] + 8:.3f} | {mean['x12_bits']:.3f} | {100 * e12:.4f} % | "
              f"{mean['x11_bits']:.3f} | {100 * e11:.3f} % | {stored:.1f} | {mb(mean['x12_bits']):.1f} | "
              f"{mb(mean['x11_bits']):.1f} | {mb(mean['H_u16']):.1f} | {', '.join(checks)} |")
    print(f"| {fname} | **all** | | | {tot['n'] / 1e6:.2f} | | | | | | | | | | | | | **{tot['bf16']:.1f}** | "
          f"**{tot['x12']:.1f}** | **{tot['x11']:.1f}** | **{tot['h']:.1f}** | |")
if len(sys.argv) > 2:
    same = sum(1 for h in hashes.values() if len(h) == 1)
    print(f"\nTensors with the same name in the files: {same} of {len(hashes)} hold the same BF16 values "
          "(F32 tensors compared after conversion to BF16).")
