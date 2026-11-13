f1: 2935 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1439 evaluations and 87 main passes
evaluations between main passes, by card: card 4: 667, card 8: 667; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 87 | 7 | 79 | 4: 79 | 75 |

**f1, all requests: 153 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.41 | 115.01 | 106.05 | 119.55 |
| verify_span | 94.92 | 93.66 | 90.42 | 99.30 |
|   verify_gpu (the nine cards) | 91.30 | 90.22 | 86.99 | 95.57 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.67 | 3.43 | 3.75 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.59 | 3.42 | 3.27 | 3.91 |
| pass 0: turn | -1.93 | -2.26 | -2.52 | 0.28 |
| pass 1 (draft 1): lead | 2.52 | 2.76 | 0.51 | 3.01 |
| pass 1: gpu | 3.42 | 2.78 | 2.48 | 6.25 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.21 | 2.77 | 2.47 | 6.22 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.14 | 2.77 | 2.47 | 6.21 |
| pass 3: turn (the server before the next verify) | 1.84 | 1.73 | 1.04 | 2.34 |
| the draft's span (all passes) | 18.49 | 16.80 | 15.32 | 27.20 |
|   the draft's GPU time on its last card | 13.38 | 12.18 | 10.72 | 22.00 |
|   the leads (crossings) | 4.06 | 4.31 | 2.06 | 4.55 |
|   the turns of the passes before the last | -0.78 | -1.15 | -1.37 | 1.44 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.23 | 3.14 | 3.34 |

verify_gpu per card, median ms: 11.13, 10.81, 10.99, 11.01, 9.10, 10.63, 8.98, 10.31, 7.35

**f1, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.81 | 107.47 | 105.89 | 118.87 |
| verify_span | 91.35 | 91.41 | 90.10 | 92.47 |
|   verify_gpu (the nine cards) | 87.84 | 87.85 | 86.65 | 88.98 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.46 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.31 | 3.25 | 3.36 |
| pass 0: turn | -1.66 | -2.20 | -2.26 | 0.29 |
| pass 1 (draft 1): lead | 2.28 | 2.70 | 0.51 | 2.76 |
| pass 1: gpu | 3.55 | 2.49 | 2.47 | 6.25 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.29 | 2.49 | 2.47 | 6.23 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.49 | 2.46 | 6.22 |
| pass 3: turn (the server before the next verify) | 1.83 | 1.73 | 1.11 | 2.37 |
| the draft's span (all passes) | 18.46 | 15.84 | 15.08 | 27.20 |
|   the draft's GPU time on its last card | 13.33 | 10.80 | 10.68 | 22.01 |
|   the leads (crossings) | 3.82 | 4.22 | 2.05 | 4.31 |
|   the turns of the passes before the last | -0.52 | -1.06 | -1.15 | 1.46 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.57 | 3.19 | 3.14 | 3.31 |

verify_gpu per card, median ms: 10.92, 10.56, 10.69, 10.74, 8.79, 10.26, 8.66, 9.96, 7.17

**f1, long: 75 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.15 | 115.80 | 114.82 | 122.32 |
| verify_span | 98.62 | 98.67 | 97.34 | 99.58 |
|   verify_gpu (the nine cards) | 94.89 | 94.94 | 93.69 | 95.80 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.73 | 3.71 | 3.67 | 3.75 |
| pass 0 (catch-up): lead | 0.52 | 0.52 | 0.51 | 0.53 |
| pass 0: gpu | 3.88 | 3.86 | 3.79 | 3.94 |
| pass 0: turn | -2.22 | -2.47 | -2.54 | -2.30 |
| pass 1 (draft 1): lead | 2.77 | 2.96 | 2.80 | 3.04 |
| pass 1: gpu | 3.30 | 2.79 | 2.74 | 6.11 |
| pass 1: turn | 0.59 | 0.59 | 0.53 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.13 | 2.79 | 2.75 | 3.49 |
| pass 2: turn | 0.58 | 0.58 | 0.54 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.12 | 2.79 | 2.75 | 2.83 |
| pass 3: turn (the server before the next verify) | 1.85 | 1.73 | 1.03 | 2.27 |
| the draft's span (all passes) | 18.53 | 17.21 | 16.43 | 24.46 |
|   the draft's GPU time on its last card | 13.43 | 12.24 | 12.13 | 16.27 |
|   the leads (crossings) | 4.31 | 4.50 | 4.33 | 4.57 |
|   the turns of the passes before the last | -1.05 | -1.30 | -1.40 | -1.10 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.62 | 3.25 | 3.21 | 3.38 |

verify_gpu per card, median ms: 11.48, 11.64, 11.23, 11.93, 9.38, 10.88, 9.85, 10.66, 7.94

