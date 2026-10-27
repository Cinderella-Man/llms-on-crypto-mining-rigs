# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Compare the unquantized (BF16) tensors of flint-int4 and the BF16 repo bit for bit.

usage: python -I floats.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import load_st  # noqa: E402

for name, n in [("model.layers.24.input_layernorm.weight", 2048), ("model.layers.24.post_attention_layernorm.weight", 2048),
                ("model.layers.24.self_attn.q_norm.weight", 128), ("model.layers.24.self_attn.k_norm.weight", 128),
                ("model.layers.24.self_attn.softmax_offset", 32), ("model.layers.24.mlp.gate.weight", 256),
                ("model.norm.weight", 2048)]:
    a = load_st("int4", f"{name}@0-{n}")
    b = load_st("bf16", f"{name}@0-{n}")
    d = np.abs(a - b)
    print(f"{name:50s} n={a.size:7d} identical {np.mean(a == b):.4%}  max|diff| {d.max():.3e}  "
          f"rel {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}")
