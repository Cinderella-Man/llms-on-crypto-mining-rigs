f1-2: 3060 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1564 evaluations and 94 main passes
evaluations between main passes, by card: card 4: 695, card 8: 695; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 94 | 7 | 86 | 4: 86 | 82 |

**f1-2, all requests: 160 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.67 | 114.94 | 106.08 | 120.83 |
| verify_span | 95.03 | 96.28 | 90.46 | 99.23 |
|   verify_gpu (the nine cards) | 91.42 | 92.59 | 87.00 | 95.49 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.66 | 3.43 | 3.75 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.60 | 3.68 | 3.28 | 3.90 |
| pass 0: turn | -1.87 | -2.25 | -2.53 | 0.32 |
| pass 1 (draft 1): lead | 2.47 | 2.75 | 0.52 | 3.03 |
| pass 1: gpu | 3.49 | 2.75 | 2.48 | 6.27 |
| pass 1: turn | 0.59 | 0.58 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.28 | 2.76 | 2.48 | 6.24 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.51 |
| pass 3: gpu | 3.21 | 2.74 | 2.48 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.77 | 1.72 | 1.08 | 2.08 |
| the draft's span (all passes) | 18.63 | 16.77 | 15.32 | 27.45 |
|   the draft's GPU time on its last card | 13.58 | 12.12 | 10.74 | 22.09 |
|   the leads (crossings) | 4.01 | 4.29 | 2.07 | 4.55 |
|   the turns of the passes before the last | -0.72 | -1.13 | -1.40 | 1.49 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.62 | 3.25 | 3.16 | 3.34 |

verify_gpu per card, median ms: 11.21, 11.37, 11.01, 11.45, 9.13, 10.60, 9.38, 10.33, 7.72

**f1-2, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.78 | 107.38 | 105.84 | 118.66 |
| verify_span | 91.37 | 91.37 | 90.27 | 92.39 |
|   verify_gpu (the nine cards) | 87.87 | 87.83 | 86.83 | 88.91 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.49 | 3.45 | 3.42 | 3.51 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.26 | 3.35 |
| pass 0: turn | -1.65 | -2.20 | -2.26 | 0.32 |
| pass 1 (draft 1): lead | 2.27 | 2.69 | 0.51 | 2.75 |
| pass 1: gpu | 3.54 | 2.49 | 2.47 | 6.25 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.51 | 0.54 |
| pass 2: gpu | 3.29 | 2.50 | 2.48 | 6.24 |
| pass 2: turn | 0.55 | 0.55 | 0.51 | 0.57 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.49 | 2.47 | 6.23 |
| pass 3: turn (the server before the next verify) | 1.78 | 1.74 | 1.11 | 2.08 |
| the draft's span (all passes) | 18.41 | 15.78 | 15.09 | 27.04 |
|   the draft's GPU time on its last card | 13.33 | 10.79 | 10.71 | 22.06 |
|   the leads (crossings) | 3.81 | 4.23 | 2.06 | 4.29 |
|   the turns of the passes before the last | -0.51 | -1.07 | -1.14 | 1.47 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.63 | 3.19 | 3.15 | 3.30 |

verify_gpu per card, median ms: 10.93, 10.55, 10.69, 10.73, 8.80, 10.23, 8.64, 9.96, 7.21

**f1-2, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.37 | 115.73 | 114.16 | 125.68 |
| verify_span | 98.52 | 98.61 | 97.25 | 99.47 |
|   verify_gpu (the nine cards) | 94.79 | 94.91 | 93.32 | 95.64 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.73 | 3.70 | 3.65 | 3.75 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.86 | 3.86 | 3.77 | 3.91 |
| pass 0: turn | -2.09 | -2.45 | -2.56 | 0.32 |
| pass 1 (draft 1): lead | 2.66 | 2.95 | 0.53 | 3.06 |
| pass 1: gpu | 3.44 | 2.78 | 2.72 | 6.41 |
| pass 1: turn | 0.58 | 0.59 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.28 | 2.79 | 2.73 | 6.37 |
| pass 2: turn | 0.58 | 0.57 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.51 |
| pass 3: gpu | 3.25 | 2.78 | 2.73 | 6.37 |
| pass 3: turn (the server before the next verify) | 1.76 | 1.68 | 0.99 | 2.08 |
| the draft's span (all passes) | 18.85 | 17.11 | 16.34 | 28.18 |
|   the draft's GPU time on its last card | 13.82 | 12.18 | 12.02 | 23.08 |
|   the leads (crossings) | 4.20 | 4.49 | 2.11 | 4.59 |
|   the turns of the passes before the last | -0.93 | -1.29 | -1.43 | 1.49 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.61 | 3.26 | 3.24 | 3.35 |

verify_gpu per card, median ms: 11.47, 11.68, 11.31, 11.90, 9.36, 10.87, 9.81, 10.65, 7.81

