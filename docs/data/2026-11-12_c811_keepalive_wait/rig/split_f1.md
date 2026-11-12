f1: 2969 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1473 evaluations and 89 main passes
evaluations between main passes, by card: card 4: 675, card 8: 675; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 89 | 7 | 81 | 4: 81 | 77 |

**f1, all requests: 155 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.01 | 115.29 | 106.59 | 120.40 |
| verify_span | 95.41 | 96.39 | 90.80 | 99.92 |
|   verify_gpu (the nine cards) | 91.82 | 92.50 | 87.35 | 96.23 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.66 | 3.43 | 3.72 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.61 | 3.69 | 3.28 | 3.92 |
| pass 0: turn | -1.97 | -2.29 | -2.55 | 0.28 |
| pass 1 (draft 1): lead | 2.56 | 2.78 | 0.52 | 3.04 |
| pass 1: gpu | 3.38 | 2.79 | 2.49 | 6.27 |
| pass 1: turn | 0.59 | 0.59 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.22 | 2.79 | 2.49 | 6.25 |
| pass 2: turn | 0.57 | 0.56 | 0.52 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.16 | 2.79 | 2.49 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.95 | 1.77 | 1.02 | 2.45 |
| the draft's span (all passes) | 18.59 | 16.79 | 15.40 | 27.70 |
|   the draft's GPU time on its last card | 13.36 | 12.25 | 10.77 | 22.08 |
|   the leads (crossings) | 4.09 | 4.32 | 2.07 | 4.58 |
|   the turns of the passes before the last | -0.81 | -1.16 | -1.40 | 1.42 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.57 | 3.23 | 3.14 | 3.35 |

verify_gpu per card, median ms: 11.23, 11.25, 11.00, 11.45, 9.12, 10.66, 9.31, 10.35, 7.81

**f1, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.27 | 107.90 | 106.26 | 119.39 |
| verify_span | 91.72 | 91.73 | 90.64 | 92.77 |
|   verify_gpu (the nine cards) | 88.23 | 88.24 | 87.18 | 89.30 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.49 | 3.45 | 3.42 | 3.50 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.33 | 3.32 | 3.26 | 3.37 |
| pass 0: turn | -1.67 | -2.20 | -2.27 | 0.29 |
| pass 1 (draft 1): lead | 2.29 | 2.70 | 0.51 | 2.77 |
| pass 1: gpu | 3.55 | 2.50 | 2.48 | 6.28 |
| pass 1: turn | 0.59 | 0.58 | 0.54 | 0.64 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.31 | 2.51 | 2.49 | 6.25 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.18 | 2.50 | 2.48 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.87 | 1.80 | 1.12 | 2.32 |
| the draft's span (all passes) | 18.55 | 15.99 | 15.13 | 27.70 |
|   the draft's GPU time on its last card | 13.37 | 10.84 | 10.74 | 22.12 |
|   the leads (crossings) | 3.83 | 4.23 | 2.05 | 4.32 |
|   the turns of the passes before the last | -0.52 | -1.06 | -1.16 | 1.47 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.54 | 3.17 | 3.12 | 3.32 |

verify_gpu per card, median ms: 10.97, 10.59, 10.76, 10.79, 8.80, 10.27, 8.68, 10.02, 7.24

**f1, long: 77 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.79 | 116.36 | 114.93 | 121.70 |
| verify_span | 99.16 | 99.17 | 97.52 | 100.25 |
|   verify_gpu (the nine cards) | 95.45 | 95.45 | 93.83 | 96.55 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.71 | 3.69 | 3.66 | 3.72 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.89 | 3.88 | 3.83 | 3.94 |
| pass 0: turn | -2.28 | -2.49 | -2.57 | -2.36 |
| pass 1 (draft 1): lead | 2.83 | 2.99 | 2.85 | 3.07 |
| pass 1: gpu | 3.20 | 2.80 | 2.76 | 2.84 |
| pass 1: turn | 0.59 | 0.59 | 0.54 | 0.62 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.12 | 2.81 | 2.77 | 2.84 |
| pass 2: turn | 0.58 | 0.57 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.13 | 2.81 | 2.76 | 2.84 |
| pass 3: turn (the server before the next verify) | 2.03 | 1.74 | 0.99 | 2.51 |
| the draft's span (all passes) | 18.64 | 17.25 | 16.45 | 23.26 |
|   the draft's GPU time on its last card | 13.35 | 12.29 | 12.22 | 15.34 |
|   the leads (crossings) | 4.36 | 4.51 | 4.41 | 4.61 |
|   the turns of the passes before the last | -1.11 | -1.33 | -1.43 | -1.16 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.60 | 3.24 | 3.21 | 3.35 |

verify_gpu per card, median ms: 11.55, 11.71, 11.39, 11.99, 9.43, 10.93, 9.81, 10.70, 7.98

