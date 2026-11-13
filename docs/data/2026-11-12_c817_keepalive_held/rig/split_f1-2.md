f1-2: 2992 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1496 evaluations and 90 main passes
evaluations between main passes, by card: card 4: 679, card 8: 679; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 90 | 7 | 82 | 4: 82 | 77 |

**f1-2, all requests: 155 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.07 | 114.62 | 106.07 | 124.35 |
| verify_span | 94.80 | 95.58 | 90.33 | 99.01 |
|   verify_gpu (the nine cards) | 91.17 | 91.64 | 86.90 | 95.27 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.68 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.52 | 0.52 | 0.51 | 0.53 |
| pass 0: gpu | 3.59 | 3.68 | 3.27 | 3.89 |
| pass 0: turn | -1.74 | -2.24 | -2.48 | 0.32 |
| pass 1 (draft 1): lead | 2.37 | 2.74 | 0.51 | 2.98 |
| pass 1: gpu | 3.70 | 2.75 | 2.48 | 6.41 |
| pass 1: turn | 0.60 | 0.60 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.43 | 2.76 | 2.48 | 6.41 |
| pass 2: turn | 0.57 | 0.56 | 0.52 | 0.62 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.32 | 2.74 | 2.47 | 6.26 |
| pass 3: turn (the server before the next verify) | 1.89 | 1.71 | 1.04 | 2.42 |
| the draft's span (all passes) | 19.27 | 16.88 | 15.36 | 28.13 |
|   the draft's GPU time on its last card | 14.04 | 12.11 | 10.73 | 23.19 |
|   the leads (crossings) | 3.91 | 4.27 | 2.06 | 4.53 |
|   the turns of the passes before the last | -0.58 | -1.07 | -1.34 | 1.78 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.25 | 3.16 | 3.36 |

verify_gpu per card, median ms: 11.15, 11.13, 10.94, 11.33, 9.09, 10.63, 9.27, 10.30, 7.70

**f1-2, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.82 | 107.34 | 105.81 | 118.67 |
| verify_span | 91.32 | 91.29 | 90.10 | 92.40 |
|   verify_gpu (the nine cards) | 87.80 | 87.72 | 86.63 | 88.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.52 | 3.45 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.32 | 3.31 | 3.26 | 3.36 |
| pass 0: turn | -1.65 | -2.18 | -2.25 | 0.29 |
| pass 1 (draft 1): lead | 2.27 | 2.67 | 0.51 | 2.74 |
| pass 1: gpu | 3.55 | 2.49 | 2.47 | 6.25 |
| pass 1: turn | 0.61 | 0.61 | 0.53 | 0.66 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.29 | 2.50 | 2.47 | 6.23 |
| pass 2: turn | 0.56 | 0.56 | 0.53 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.17 | 2.49 | 2.46 | 6.23 |
| pass 3: turn (the server before the next verify) | 1.83 | 1.75 | 1.11 | 2.35 |
| the draft's span (all passes) | 18.50 | 15.85 | 15.12 | 27.18 |
|   the draft's GPU time on its last card | 13.33 | 10.79 | 10.71 | 22.05 |
|   the leads (crossings) | 3.81 | 4.21 | 2.05 | 4.28 |
|   the turns of the passes before the last | -0.48 | -1.02 | -1.11 | 1.48 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.56 | 3.21 | 3.16 | 3.32 |

verify_gpu per card, median ms: 10.89, 10.56, 10.70, 10.74, 8.80, 10.23, 8.62, 9.97, 7.18

**f1-2, long: 77 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.38 | 116.16 | 114.25 | 126.83 |
| verify_span | 98.33 | 98.53 | 96.91 | 99.23 |
|   verify_gpu (the nine cards) | 94.58 | 94.73 | 93.25 | 95.52 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.75 | 3.71 | 3.68 | 3.78 |
| pass 0 (catch-up): lead | 0.52 | 0.52 | 0.51 | 0.54 |
| pass 0: gpu | 3.86 | 3.85 | 3.76 | 3.91 |
| pass 0: turn | -1.84 | -2.42 | -2.50 | 0.29 |
| pass 1 (draft 1): lead | 2.46 | 2.92 | 0.51 | 3.00 |
| pass 1: gpu | 3.85 | 2.79 | 2.73 | 6.46 |
| pass 1: turn | 0.59 | 0.59 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.58 | 2.79 | 2.73 | 6.45 |
| pass 2: turn | 0.57 | 0.57 | 0.52 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.48 | 2.77 | 2.73 | 6.45 |
| pass 3: turn (the server before the next verify) | 1.95 | 1.65 | 0.96 | 2.42 |
| the draft's span (all passes) | 20.05 | 17.15 | 16.32 | 28.30 |
|   the draft's GPU time on its last card | 14.77 | 12.16 | 12.04 | 23.24 |
|   the leads (crossings) | 4.01 | 4.45 | 2.06 | 4.55 |
|   the turns of the passes before the last | -0.68 | -1.26 | -1.37 | 1.51 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.62 | 3.27 | 3.24 | 3.36 |

verify_gpu per card, median ms: 11.48, 11.63, 11.24, 11.87, 9.36, 10.85, 9.79, 10.58, 7.87

