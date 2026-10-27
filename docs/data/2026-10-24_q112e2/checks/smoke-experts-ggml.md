Experts per layer: 6, spread over the ranking by routed rows; errors are sums over each expert's rows (the layer ratios weight experts by use)

| Layer | Tensor | mix type | experts (rows: min / median / max) | mix / RTN | RTN = mix (rows) | GPTQ / RTN | per expert: median, 10th-90th pct | relative RTN error (median) |
|---|---|---|---|---|---|---|---|---|
| 9 | gate | Q5_0 | 6 (1 / 40 / 540) | 1.000 | 100.0 % | **0.173** | 0.176, 0.171-0.185 | 1.37e-03 |
| 9 | up | Q5_0 | 6 (1 / 40 / 540) | 1.000 | 100.0 % | **0.172** | 0.179, 0.170-0.189 | 1.51e-03 |
| 9 | down | Q5_0 | 6 (1 / 40 / 540) | 1.000 | 100.0 % | **0.326** | 0.041, 0.011-0.213 | 1.57e-03 |
| 47 | gate | Q5_0 | 6 (0 / 35 / 289) | 1.000 | 100.0 % | **0.197** | 0.190, 0.183-0.201 | 1.32e-03 |
| 47 | up | Q5_0 | 6 (0 / 35 / 289) | 1.000 | 100.0 % | **0.192** | 0.188, 0.188-0.198 | 1.43e-03 |
| 47 | down | Q8_0 | 6 (0 / 35 / 289) | 0.018 | nan % | **0.208** | 0.058, 0.023-0.184 | 1.71e-03 |

All sampled experts: GPTQ / RTN error 0.193 (kill above 0.85)
Layers/tensors above the kill line: none
