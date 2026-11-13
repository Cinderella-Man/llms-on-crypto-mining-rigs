f1ka250-2: 3054 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1558 evaluations and 94 main passes
evaluations between main passes, by card: card 4: 695, card 8: 695; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 94 | 7 | 86 | 4: 86 | 82 |

**f1ka250-2, all requests: 160 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.70 | 114.75 | 106.14 | 121.08 |
| verify_span | 95.01 | 96.67 | 90.46 | 99.18 |
|   verify_gpu (the nine cards) | 91.38 | 92.81 | 87.03 | 95.43 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.67 | 3.44 | 3.76 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.62 | 3.74 | 3.28 | 3.91 |
| pass 0: turn | -1.85 | -2.25 | -2.49 | 0.28 |
| pass 1 (draft 1): lead | 2.45 | 2.74 | 0.51 | 2.99 |
| pass 1: gpu | 3.51 | 2.75 | 2.48 | 6.30 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.31 | 2.75 | 2.48 | 6.26 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.21 | 2.76 | 2.47 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.75 | 1.74 | 1.04 | 2.16 |
| the draft's span (all passes) | 18.70 | 16.83 | 15.35 | 27.72 |
|   the draft's GPU time on its last card | 13.65 | 12.14 | 10.73 | 22.09 |
|   the leads (crossings) | 3.99 | 4.29 | 2.06 | 4.54 |
|   the turns of the passes before the last | -0.70 | -1.14 | -1.37 | 1.46 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.61 | 3.23 | 3.15 | 3.34 |

verify_gpu per card, median ms: 11.26, 11.35, 11.05, 11.51, 9.14, 10.65, 9.50, 10.36, 7.72

**f1ka250-2, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.84 | 107.31 | 105.72 | 118.72 |
| verify_span | 91.36 | 91.31 | 90.21 | 92.44 |
|   verify_gpu (the nine cards) | 87.86 | 87.82 | 86.77 | 88.87 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.50 | 3.46 | 3.43 | 3.73 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.35 | 3.30 | 3.25 | 3.36 |
| pass 0: turn | -1.65 | -2.19 | -2.25 | 0.28 |
| pass 1 (draft 1): lead | 2.27 | 2.69 | 0.51 | 2.75 |
| pass 1: gpu | 3.54 | 2.50 | 2.48 | 6.25 |
| pass 1: turn | 0.59 | 0.58 | 0.54 | 0.64 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.29 | 2.50 | 2.47 | 6.24 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.57 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.17 | 2.49 | 2.45 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.83 | 1.77 | 1.11 | 2.17 |
| the draft's span (all passes) | 18.48 | 15.83 | 15.03 | 27.45 |
|   the draft's GPU time on its last card | 13.35 | 10.78 | 10.71 | 22.08 |
|   the leads (crossings) | 3.81 | 4.22 | 2.05 | 4.29 |
|   the turns of the passes before the last | -0.51 | -1.06 | -1.15 | 1.44 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.64 | 3.19 | 3.14 | 3.33 |

verify_gpu per card, median ms: 10.92, 10.57, 10.68, 10.74, 8.82, 10.23, 8.65, 9.97, 7.19

**f1ka250-2, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.37 | 115.64 | 114.28 | 126.68 |
| verify_span | 98.47 | 98.49 | 97.24 | 99.37 |
|   verify_gpu (the nine cards) | 94.73 | 94.77 | 93.54 | 95.68 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.71 | 3.67 | 3.76 |
| pass 0 (catch-up): lead | 0.52 | 0.52 | 0.51 | 0.53 |
| pass 0: gpu | 3.87 | 3.86 | 3.76 | 3.93 |
| pass 0: turn | -2.05 | -2.44 | -2.53 | 0.28 |
| pass 1 (draft 1): lead | 2.63 | 2.94 | 0.52 | 3.03 |
| pass 1: gpu | 3.49 | 2.77 | 2.73 | 6.44 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.32 | 2.78 | 2.73 | 6.42 |
| pass 2: turn | 0.58 | 0.57 | 0.52 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.26 | 2.78 | 2.74 | 6.40 |
| pass 3: turn (the server before the next verify) | 1.67 | 1.72 | 0.99 | 1.99 |
| the draft's span (all passes) | 18.90 | 17.13 | 16.20 | 28.33 |
|   the draft's GPU time on its last card | 13.93 | 12.17 | 12.06 | 23.19 |
|   the leads (crossings) | 4.17 | 4.48 | 2.07 | 4.57 |
|   the turns of the passes before the last | -0.87 | -1.27 | -1.39 | 1.46 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.25 | 3.21 | 3.34 |

verify_gpu per card, median ms: 11.47, 11.60, 11.27, 11.86, 9.36, 10.87, 9.78, 10.63, 7.88

