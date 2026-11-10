#!/usr/bin/env python3
"""output_head_check.py: are a checkpoint's output head and token embeddings another model file's, value for value?

    PYTHONPATH=<llama.cpp>/gguf-py python3 output_head_check.py NEW_BF16 BASE_GGUF [--out TSV]

NEW_BF16 holds `output.weight` and `token_embd.weight` in BF16 (the checkpoint's own values, written by
mtp_head_from_safetensors.py); BASE_GGUF is a file of the base model that holds the two in Q8_0. Q8_0 is rounded
without a search (per 32 values: the largest magnitude over 127 as the scale, each value to the nearest step), so
rounding the same values gives the same bytes, and gguf-py's Q8_0 is ggml's reference rounding bit for bit. The
script rounds NEW_BF16's tensors to Q8_0 and counts the 32-value blocks whose bytes equal BASE_GGUF's; for the
blocks that differ it prints how far the decoded values are apart (|a - b| / |a| over the whole tensor).
"""
import argparse

import numpy as np

import gguf
from gguf import quants

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("new_bf16")
ap.add_argument("base")
ap.add_argument("--out")
A = ap.parse_args()
Q8_0, BF16 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.BF16
new = {t.name: t for t in gguf.GGUFReader(A.new_bf16).tensors}
base = {t.name: t for t in gguf.GGUFReader(A.base).tensors}
lines = ["tensor\tshape\tbase_type\tblocks\tsame_blocks\tvalues\tsame_values\trel"]
for name in ("token_embd.weight", "output.weight"):
    n, b = new[name], base[name]
    if n.tensor_type != BF16 or b.tensor_type != Q8_0 or list(n.shape) != list(b.shape):
        raise SystemExit(f"{name}: {n.tensor_type.name} {list(n.shape)} against {b.tensor_type.name} {list(b.shape)}")
    rows = n.data.shape[0]
    blocks = same = values = same_values = 0
    dd = aa = 0.0
    for lo in range(0, rows, 16384):
        x = quants.dequantize(np.asarray(n.data[lo:lo + 16384]), BF16)
        q = quants.quantize(x, Q8_0).reshape(-1, 34)
        r = np.asarray(b.data[lo:lo + 16384]).reshape(-1, 34)
        eq = (q == r).all(axis=1)
        blocks += len(eq)
        same += int(eq.sum())
        xq = quants.dequantize(q.reshape(-1, b.data.shape[1]), Q8_0).astype(np.float64)
        xr = quants.dequantize(np.asarray(b.data[lo:lo + 16384]), Q8_0).astype(np.float64)
        values += xq.size
        same_values += int((xq == xr).sum())
        dd += float(((xq - xr) ** 2).sum())
        aa += float((xr ** 2).sum())
    lines.append(f"{name}\t{'x'.join(str(int(v)) for v in n.shape)}\t{b.tensor_type.name}\t{blocks}\t{same}\t{values}\t"
                 f"{same_values}\t{(dd / aa) ** 0.5:.6f}")
    print(lines[-1], flush=True)
if A.out:
    with open(A.out, "w") as f:
        f.write("\n".join(lines) + "\n")
