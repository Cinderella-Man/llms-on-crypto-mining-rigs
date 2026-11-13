f0ka250-2: 3020 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1524 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 687, card 8: 687; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 92 | 6 | 85 | 4: 84, 8: 1 | 82 |

**f0ka250-2, all requests: 162 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.19 | 115.26 | 106.20 | 125.70 |
| verify_span | 95.18 | 97.09 | 90.48 | 99.44 |
|   verify_gpu (the nine cards) | 91.55 | 93.40 | 87.04 | 95.66 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.67 | 3.43 | 3.86 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.60 | 3.75 | 3.27 | 3.91 |
| pass 0: turn | -1.78 | -2.25 | -2.49 | 0.29 |
| pass 1 (draft 1): lead | 2.40 | 2.74 | 0.51 | 2.99 |
| pass 1: gpu | 3.36 | 2.74 | 2.48 | 6.40 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.38 | 2.75 | 2.48 | 6.41 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.30 | 2.74 | 2.47 | 6.26 |
| pass 3: turn (the server before the next verify) | 2.06 | 1.76 | 1.04 | 2.35 |
| the draft's span (all passes) | 19.01 | 16.97 | 15.32 | 28.21 |
|   the draft's GPU time on its last card | 13.65 | 12.10 | 10.72 | 23.08 |
|   the leads (crossings) | 3.93 | 4.26 | 2.04 | 4.52 |
|   the turns of the passes before the last | -0.62 | -1.12 | -1.36 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.80 | 3.23 | 3.15 | 3.34 |

verify_gpu per card, median ms: 11.27, 11.37, 11.11, 11.55, 9.21, 10.66, 9.49, 10.34, 7.73

**f0ka250-2, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.68 | 107.41 | 105.69 | 119.01 |
| verify_span | 91.50 | 91.36 | 90.15 | 92.45 |
|   verify_gpu (the nine cards) | 88.00 | 87.84 | 86.72 | 88.97 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.45 | 3.42 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.26 | 3.37 |
| pass 0: turn | -1.67 | -2.19 | -2.26 | 0.28 |
| pass 1 (draft 1): lead | 2.28 | 2.68 | 0.51 | 2.75 |
| pass 1: gpu | 3.22 | 2.49 | 2.47 | 6.25 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.24 | 2.49 | 2.47 | 6.23 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.15 | 2.49 | 2.46 | 6.23 |
| pass 3: turn (the server before the next verify) | 1.95 | 1.76 | 1.12 | 2.18 |
| the draft's span (all passes) | 18.17 | 15.83 | 15.09 | 27.00 |
|   the draft's GPU time on its last card | 12.93 | 10.77 | 10.71 | 22.06 |
|   the leads (crossings) | 3.81 | 4.20 | 2.05 | 4.28 |
|   the turns of the passes before the last | -0.52 | -1.03 | -1.16 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.77 | 3.18 | 3.13 | 3.32 |

verify_gpu per card, median ms: 10.93, 10.56, 10.69, 10.74, 8.82, 10.24, 8.64, 9.95, 7.22

**f0ka250-2, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.59 | 115.95 | 114.37 | 127.36 |
| verify_span | 98.77 | 98.67 | 97.70 | 99.74 |
|   verify_gpu (the nine cards) | 95.01 | 94.89 | 93.99 | 95.98 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.76 | 3.70 | 3.67 | 4.09 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.54 |
| pass 0: gpu | 3.87 | 3.86 | 3.78 | 3.94 |
| pass 0: turn | -1.90 | -2.44 | -2.55 | 0.30 |
| pass 1 (draft 1): lead | 2.51 | 2.93 | 0.51 | 3.04 |
| pass 1: gpu | 3.51 | 2.79 | 2.73 | 6.46 |
| pass 1: turn | 0.60 | 0.60 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.52 | 2.78 | 2.73 | 6.45 |
| pass 2: turn | 0.58 | 0.57 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.45 | 2.77 | 2.73 | 6.45 |
| pass 3: turn (the server before the next verify) | 2.16 | 1.76 | 1.00 | 2.54 |
| the draft's span (all passes) | 19.82 | 17.26 | 16.28 | 28.96 |
|   the draft's GPU time on its last card | 14.34 | 12.17 | 12.06 | 23.26 |
|   the leads (crossings) | 4.04 | 4.45 | 2.04 | 4.56 |
|   the turns of the passes before the last | -0.72 | -1.28 | -1.41 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.83 | 3.24 | 3.21 | 3.37 |

verify_gpu per card, median ms: 11.49, 11.65, 11.29, 11.90, 9.39, 10.89, 9.79, 10.67, 7.87

