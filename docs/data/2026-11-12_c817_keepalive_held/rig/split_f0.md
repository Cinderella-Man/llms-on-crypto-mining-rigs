f0: 1938 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 442 evaluations and 28 main passes
evaluations between main passes, by card: card 4: 431, card 8: 431; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 28 | 6 | 21 | 4: 20, 8: 1 | 18 |

**f0, all requests: 98 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 111.61 | 108.45 | 106.07 | 119.47 |
| verify_span | 93.00 | 91.60 | 90.36 | 98.32 |
|   verify_gpu (the nine cards) | 89.44 | 88.07 | 86.92 | 94.65 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.56 | 3.46 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.43 | 3.32 | 3.26 | 3.82 |
| pass 0: turn | -1.74 | -2.21 | -2.29 | 0.29 |
| pass 1 (draft 1): lead | 2.34 | 2.70 | 0.51 | 2.78 |
| pass 1: gpu | 3.23 | 2.50 | 2.47 | 6.27 |
| pass 1: turn | 0.61 | 0.58 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.26 | 2.51 | 2.47 | 6.24 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.57 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.50 | 2.47 | 6.21 |
| pass 3: turn (the server before the next verify) | 2.22 | 1.87 | 1.12 | 2.95 |
| the draft's span (all passes) | 18.61 | 16.75 | 15.11 | 27.49 |
|   the draft's GPU time on its last card | 13.09 | 10.81 | 10.71 | 22.08 |
|   the leads (crossings) | 3.88 | 4.23 | 2.05 | 4.32 |
|   the turns of the passes before the last | -0.58 | -1.09 | -1.18 | 1.50 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 4.15 | 3.19 | 3.13 | 3.33 |

verify_gpu per card, median ms: 10.92, 10.59, 10.72, 10.80, 8.85, 10.35, 8.70, 10.02, 7.21

**f0, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.86 | 107.72 | 106.02 | 119.06 |
| verify_span | 91.51 | 91.34 | 90.26 | 92.52 |
|   verify_gpu (the nine cards) | 87.99 | 87.78 | 86.80 | 89.05 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.45 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.26 | 3.36 |
| pass 0: turn | -1.68 | -2.21 | -2.27 | 0.29 |
| pass 1 (draft 1): lead | 2.29 | 2.70 | 0.51 | 2.76 |
| pass 1: gpu | 3.22 | 2.49 | 2.47 | 6.27 |
| pass 1: turn | 0.60 | 0.58 | 0.54 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.25 | 2.49 | 2.47 | 6.24 |
| pass 2: turn | 0.54 | 0.55 | 0.52 | 0.56 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.15 | 2.49 | 2.47 | 6.21 |
| pass 3: turn (the server before the next verify) | 2.13 | 1.89 | 1.11 | 2.95 |
| the draft's span (all passes) | 18.36 | 16.03 | 15.08 | 27.49 |
|   the draft's GPU time on its last card | 12.94 | 10.78 | 10.69 | 22.08 |
|   the leads (crossings) | 3.83 | 4.22 | 2.04 | 4.29 |
|   the turns of the passes before the last | -0.54 | -1.09 | -1.16 | 1.50 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.78 | 3.18 | 3.13 | 3.31 |

verify_gpu per card, median ms: 10.89, 10.56, 10.70, 10.75, 8.80, 10.28, 8.65, 9.96, 7.18

**f0, long: 18 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 119.37 | 116.18 | 114.31 | 126.73 |
| verify_span | 99.64 | 98.47 | 96.58 | 103.96 |
|   verify_gpu (the nine cards) | 95.84 | 94.75 | 92.89 | 99.82 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.80 | 3.70 | 3.68 | 3.75 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.51 |
| pass 0: gpu | 3.90 | 3.82 | 3.74 | 3.90 |
| pass 0: turn | -1.99 | -2.27 | -2.49 | -2.02 |
| pass 1 (draft 1): lead | 2.56 | 2.76 | 2.55 | 2.98 |
| pass 1: gpu | 3.30 | 2.79 | 2.78 | 3.53 |
| pass 1: turn | 0.64 | 0.59 | 0.57 | 0.72 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.35 | 2.81 | 2.78 | 3.49 |
| pass 2: turn | 0.60 | 0.56 | 0.54 | 0.62 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.22 | 2.78 | 2.77 | 3.34 |
| pass 3: turn (the server before the next verify) | 2.63 | 1.83 | 1.69 | 2.03 |
| the draft's span (all passes) | 19.73 | 17.15 | 17.04 | 28.54 |
|   the draft's GPU time on its last card | 13.76 | 12.20 | 12.13 | 13.46 |
|   the leads (crossings) | 4.10 | 4.29 | 4.10 | 4.52 |
|   the turns of the passes before the last | -0.75 | -1.12 | -1.36 | 0.23 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.09 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 5.78 | 3.25 | 3.19 | 3.43 |

verify_gpu per card, median ms: 11.43, 11.65, 11.34, 11.87, 9.31, 10.88, 9.71, 10.51, 7.84

