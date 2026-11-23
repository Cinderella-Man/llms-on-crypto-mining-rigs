#!/usr/bin/env python3
"""C953 (Q1687 point 7): a test-backend-ops test file (--test-file) of Ornith-1.5-122B Pascal N2P GGUF's dense Q5_0 and
Q8_0 products and of the bounds of GGML_CUDA_MMVQ_PASCAL_MT=2's rule (fork file ggml/src/ggml-cuda/mmvq-pascal-pick.h,
C899), each at 1 to 5 tokens. One MUL_MAT per line in test-backend-ops' file format: op, output type, output ne[4],
op-parameter count, source count, then per source its type, ne[4] and nb[4] (0 0 0 0: contiguous), then a name.

  ggml enum values of the fork at 37c05b76a: GGML_OP_MUL_MAT 28; GGML_TYPE_F32 0, Q5_0 6, Q8_0 8.

Usage: python3 -I shapes.py > shapes.txt
"""
OP_MUL_MAT, F32, Q5_0, Q8_0 = 28, 0, 6, 8

# (name, type, values in (K), rows (M))
SHAPES = [
    # Ornith's dense products (R197's table, docs/data/2026-11-21_c899_kcol_ornith/rule.md)
    ("ornith_ssm_out_attn_output", Q5_0, 8192, 3072),
    ("ornith_output", Q5_0, 3072, 248320),
    ("ornith_attn_q", Q5_0, 3072, 16384),
    ("ornith_attn_qkv", Q5_0, 3072, 12288),
    ("ornith_attn_gate", Q5_0, 3072, 8192),
    ("ornith_ffn_gate_up_shexp", Q5_0, 3072, 1024),
    ("ornith_ssm_alpha_beta", Q5_0, 3072, 64),
    ("ornith_ffn_down_shexp", Q5_0, 1024, 3072),
    ("ornith_attn_k_v", Q8_0, 3072, 512),
    # the rule's bounds
    ("bound_q5_0_k4096_r4096", Q5_0, 4096, 4096),
    ("bound_q5_0_k4096_r4097", Q5_0, 4096, 4097),
    ("bound_q5_0_k4128_r64", Q5_0, 4128, 64),
    ("bound_q5_0_k2048_r1024", Q5_0, 2048, 1024),
    ("bound_q5_0_k2080_r1024", Q5_0, 2080, 1024),
    ("bound_q8_0_k4096_r512", Q8_0, 4096, 512),
    ("bound_q8_0_k4128_r512", Q8_0, 4128, 512),
]

for name, t, k, m in SHAPES:
    for n in range(1, 6):
        print(f"{OP_MUL_MAT} {F32} {m} {n} 1 1 0 2 {t} {k} {m} 1 1 0 0 0 0 {F32} {k} {n} 1 1 0 0 0 0 {name}_n{n}")
