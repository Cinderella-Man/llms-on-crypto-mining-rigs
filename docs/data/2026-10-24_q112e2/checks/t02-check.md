layer 9: 2044 rows, 6 experts per token, 256 experts; dense rows in the file 2044
rows in ffn_norm .h64 vs dump: 0.00e+00 (tolerance 0e+00) ok
ffn_norm H vs X^T X (max abs / max): 1.40e-07 (tolerance 1e-05) ok
gate/up row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
down row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
distinct experts per token: 0.00e+00 (tolerance 0e+00) ok
per-expert gate/up H vs recomputed (worst expert): 1.35e-07 (tolerance 1e-05) ok
per-expert down H vs recomputed (worst expert): 1.45e-07 (tolerance 1e-05) ok
sum of expert gate/up H vs n_used x all-token H: 1.43e-07 (tolerance 1e-05) ok
ALL OK
layer 47: 2044 rows, 6 experts per token, 256 experts; dense rows in the file 2044
rows in ffn_norm .h64 vs dump: 0.00e+00 (tolerance 0e+00) ok
ffn_norm H vs X^T X (max abs / max): 1.53e-07 (tolerance 1e-05) ok
gate/up row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
down row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
distinct experts per token: 0.00e+00 (tolerance 0e+00) ok
per-expert gate/up H vs recomputed (worst expert): 1.89e-07 (tolerance 1e-05) ok
per-expert down H vs recomputed (worst expert): 2.55e-07 (tolerance 1e-05) ok
sum of expert gate/up H vs n_used x all-token H: 1.56e-07 (tolerance 1e-05) ok
ALL OK
attn_norm-0.h64: 2.35e-07
experts-47-down.e64: 5.03e-08
experts-47-gate_up.e64: 6.00e-08
experts-9-down.e64: 4.17e-08
experts-9-gate_up.e64: 2.44e-08
ffn_norm-47.h64: 2.00e-07
ffn_norm-9.h64: 1.29e-07
ffn_swiglu-0.h64: 2.12e-07
ffn_swiglu-20.h64: 8.40e-08
kqv_out-33.h64: 1.04e-07
10 files, worst 2.35e-07 (tolerance 1e-05): ALL OK
