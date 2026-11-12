f-2: 3122 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1626 evaluations and 98 main passes
evaluations between main passes, by card: card 4: 711, card 8: 711; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 98 | 6 | 91 | 4: 90, 8: 1 | 88 |

**f-2, all requests: 168 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.99 | 115.43 | 106.73 | 120.08 |
| verify_span | 95.66 | 97.66 | 90.85 | 99.59 |
|   verify_gpu (the nine cards) | 92.03 | 93.94 | 87.43 | 95.84 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.67 | 3.43 | 3.74 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.63 | 3.79 | 3.29 | 3.92 |
| pass 0: turn | -1.94 | -2.27 | -2.54 | 0.29 |
| pass 1 (draft 1): lead | 2.52 | 2.76 | 0.51 | 3.03 |
| pass 1: gpu | 3.18 | 2.74 | 2.49 | 6.26 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.21 | 2.76 | 2.49 | 6.24 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.15 | 2.78 | 2.49 | 6.23 |
| pass 3: turn (the server before the next verify) | 1.91 | 1.81 | 1.06 | 2.45 |
| the draft's span (all passes) | 18.33 | 17.01 | 15.36 | 27.08 |
|   the draft's GPU time on its last card | 13.16 | 12.15 | 10.79 | 22.07 |
|   the leads (crossings) | 4.04 | 4.30 | 2.06 | 4.56 |
|   the turns of the passes before the last | -0.78 | -1.14 | -1.40 | 1.48 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.73 | 3.19 | 3.12 | 3.28 |

verify_gpu per card, median ms: 11.32, 11.50, 11.11, 11.70, 9.26, 10.69, 9.64, 10.49, 7.75

**f-2, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.20 | 107.92 | 106.30 | 119.55 |
| verify_span | 91.94 | 91.74 | 90.53 | 93.01 |
|   verify_gpu (the nine cards) | 88.42 | 88.22 | 87.10 | 89.54 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.52 | 3.45 | 3.42 | 3.83 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.34 | 3.31 | 3.26 | 3.37 |
| pass 0: turn | -1.68 | -2.21 | -2.27 | 0.30 |
| pass 1 (draft 1): lead | 2.29 | 2.70 | 0.51 | 2.76 |
| pass 1: gpu | 3.23 | 2.50 | 2.47 | 6.27 |
| pass 1: turn | 0.60 | 0.58 | 0.53 | 0.69 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.26 | 2.51 | 2.48 | 6.25 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.51 | 2.48 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.98 | 1.83 | 1.11 | 2.44 |
| the draft's span (all passes) | 18.27 | 16.00 | 15.16 | 27.27 |
|   the draft's GPU time on its last card | 12.99 | 10.83 | 10.75 | 22.10 |
|   the leads (crossings) | 3.82 | 4.21 | 2.04 | 4.28 |
|   the turns of the passes before the last | -0.53 | -1.07 | -1.14 | 1.53 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.73 | 3.16 | 3.10 | 3.28 |

verify_gpu per card, median ms: 11.01, 10.63, 10.73, 10.81, 8.85, 10.25, 8.66, 10.00, 7.24

**f-2, long: 88 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.43 | 116.06 | 114.89 | 124.86 |
| verify_span | 99.05 | 98.82 | 97.85 | 100.01 |
|   verify_gpu (the nine cards) | 95.31 | 95.05 | 94.13 | 96.24 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.70 | 3.67 | 3.74 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.89 | 3.88 | 3.81 | 3.93 |
| pass 0: turn | -2.17 | -2.47 | -2.57 | -2.31 |
| pass 1 (draft 1): lead | 2.72 | 2.96 | 2.81 | 3.06 |
| pass 1: gpu | 3.14 | 2.78 | 2.73 | 3.35 |
| pass 1: turn | 0.59 | 0.59 | 0.54 | 0.62 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.16 | 2.79 | 2.73 | 3.48 |
| pass 2: turn | 0.57 | 0.57 | 0.53 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.12 | 2.80 | 2.74 | 2.83 |
| pass 3: turn (the server before the next verify) | 1.84 | 1.80 | 1.06 | 2.49 |
| the draft's span (all passes) | 18.38 | 17.24 | 16.43 | 21.35 |
|   the draft's GPU time on its last card | 13.31 | 12.22 | 12.13 | 13.53 |
|   the leads (crossings) | 4.24 | 4.48 | 4.31 | 4.59 |
|   the turns of the passes before the last | -1.01 | -1.31 | -1.42 | -0.68 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.72 | 3.20 | 3.17 | 3.28 |

verify_gpu per card, median ms: 11.55, 11.73, 11.26, 11.92, 9.43, 10.85, 9.82, 10.71, 7.83

