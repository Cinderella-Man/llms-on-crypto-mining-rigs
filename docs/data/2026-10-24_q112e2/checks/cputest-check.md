layer 9: 2044 rows, 6 experts per token, 256 experts; dense rows in the file 2044
rows in ffn_norm .h64 vs dump: 0.00e+00 (tolerance 0e+00) ok
ffn_norm H vs X^T X (max abs / max): 6.56e-08 (tolerance 1e-05) ok
gate/up row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
down row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
distinct experts per token: 0.00e+00 (tolerance 0e+00) ok
per-expert gate/up H vs recomputed (worst expert): 1.36e-07 (tolerance 1e-05) ok
per-expert down H vs recomputed (worst expert): 1.82e-07 (tolerance 1e-05) ok
sum of expert gate/up H vs n_used x all-token H: 6.35e-08 (tolerance 1e-05) ok
ALL OK
layer 47: 2044 rows, 6 experts per token, 256 experts; dense rows in the file 2044
rows in ffn_norm .h64 vs dump: 0.00e+00 (tolerance 0e+00) ok
ffn_norm H vs X^T X (max abs / max): 4.96e-08 (tolerance 1e-05) ok
gate/up row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
down row counts vs ids: 0.00e+00 (tolerance 0e+00) ok
distinct experts per token: 0.00e+00 (tolerance 0e+00) ok
per-expert gate/up H vs recomputed (worst expert): 1.22e-07 (tolerance 1e-05) ok
per-expert down H vs recomputed (worst expert): 1.25e-07 (tolerance 1e-05) ok
sum of expert gate/up H vs n_used x all-token H: 5.09e-08 (tolerance 1e-05) ok
ALL OK
Experts per layer: 6, spread over the ranking by routed rows; errors are sums over each expert's rows (the layer ratios weight experts by use)

| Layer | Tensor | mix type | experts (rows: min / median / max) | mix / RTN | RTN = mix (rows) | GPTQ / RTN | per expert: median, 10th-90th pct | relative RTN error (median) |
|---|---|---|---|---|---|---|---|---|
| 9 | gate | Q5_0 | 6 (1 / 40 / 540) | 0.999 | 82.4 % | **0.173** | 0.176, 0.170-0.185 | 1.37e-03 |
| 9 | up | Q5_0 | 6 (1 / 40 / 540) | 1.000 | 83.9 % | **0.172** | 0.179, 0.169-0.189 | 1.51e-03 |
| 9 | down | Q5_0 | 6 (1 / 40 / 540) | 1.000 | 95.0 % | **0.326** | 0.041, 0.011-0.213 | 1.57e-03 |
| 47 | gate | Q5_0 | 6 (0 / 35 / 289) | 1.000 | 68.3 % | **0.197** | 0.190, 0.183-0.202 | 1.32e-03 |
| 47 | up | Q5_0 | 6 (0 / 35 / 289) | 1.000 | 68.7 % | **0.192** | 0.188, 0.188-0.198 | 1.43e-03 |
| 47 | down | Q8_0 | 6 (0 / 35 / 289) | 0.018 | nan % | **0.208** | 0.058, 0.023-0.184 | 1.71e-03 |

All sampled experts: GPTQ / RTN error 0.193 (kill above 0.85)
Layers/tensors above the kill line: none
