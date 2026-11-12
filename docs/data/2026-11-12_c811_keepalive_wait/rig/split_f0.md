f0: 3020 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1524 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 687, card 8: 687; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 92 | 6 | 85 | 4: 84, 8: 1 | 82 |

**f0, all requests: 162 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.42 | 115.72 | 106.48 | 121.77 |
| verify_span | 95.71 | 97.87 | 90.84 | 99.91 |
|   verify_gpu (the nine cards) | 92.07 | 94.18 | 87.29 | 96.12 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.68 | 3.43 | 3.82 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.63 | 3.79 | 3.28 | 3.94 |
| pass 0: turn | -1.91 | -2.26 | -2.53 | 0.29 |
| pass 1 (draft 1): lead | 2.50 | 2.76 | 0.51 | 3.02 |
| pass 1: gpu | 3.26 | 2.78 | 2.49 | 6.29 |
| pass 1: turn | 0.60 | 0.58 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.53 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.28 | 2.80 | 2.49 | 6.24 |
| pass 2: turn | 0.57 | 0.56 | 0.52 | 0.60 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.21 | 2.79 | 2.48 | 6.26 |
| pass 3: turn (the server before the next verify) | 2.02 | 1.86 | 1.10 | 2.53 |
| the draft's span (all passes) | 18.71 | 17.05 | 15.38 | 27.80 |
|   the draft's GPU time on its last card | 13.38 | 12.24 | 10.75 | 22.16 |
|   the leads (crossings) | 4.04 | 4.29 | 2.05 | 4.55 |
|   the turns of the passes before the last | -0.73 | -1.14 | -1.39 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.91 | 3.21 | 3.12 | 3.31 |

verify_gpu per card, median ms: 11.36, 11.46, 11.18, 11.74, 9.23, 10.71, 9.54, 10.42, 7.81

**f0, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.02 | 107.94 | 106.22 | 119.28 |
| verify_span | 91.79 | 91.62 | 90.53 | 92.83 |
|   verify_gpu (the nine cards) | 88.28 | 88.13 | 87.09 | 89.29 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.45 | 3.42 | 3.82 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.33 | 3.31 | 3.25 | 3.38 |
| pass 0: turn | -1.67 | -2.19 | -2.26 | 0.29 |
| pass 1 (draft 1): lead | 2.29 | 2.69 | 0.51 | 2.76 |
| pass 1: gpu | 3.23 | 2.50 | 2.47 | 6.25 |
| pass 1: turn | 0.60 | 0.58 | 0.52 | 0.65 |
| pass 2 (draft 2): lead | 0.55 | 0.51 | 0.49 | 0.54 |
| pass 2: gpu | 3.25 | 2.50 | 2.48 | 6.24 |
| pass 2: turn | 0.57 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.16 | 2.50 | 2.47 | 6.26 |
| pass 3: turn (the server before the next verify) | 1.91 | 1.86 | 1.11 | 2.41 |
| the draft's span (all passes) | 18.24 | 16.02 | 15.06 | 27.21 |
|   the draft's GPU time on its last card | 12.97 | 10.81 | 10.72 | 22.13 |
|   the leads (crossings) | 3.85 | 4.21 | 2.05 | 4.29 |
|   the turns of the passes before the last | -0.50 | -1.06 | -1.15 | 1.52 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.17 | 3.11 | 3.27 |

verify_gpu per card, median ms: 10.94, 10.61, 10.75, 10.79, 8.80, 10.28, 8.68, 10.02, 7.21

**f0, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.71 | 116.67 | 115.46 | 128.17 |
| verify_span | 99.53 | 99.18 | 98.24 | 100.37 |
|   verify_gpu (the nine cards) | 95.77 | 95.46 | 94.50 | 96.47 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.76 | 3.71 | 3.68 | 3.77 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 0: gpu | 3.91 | 3.89 | 3.82 | 3.96 |
| pass 0: turn | -2.13 | -2.48 | -2.55 | 0.28 |
| pass 1 (draft 1): lead | 2.70 | 2.98 | 0.51 | 3.04 |
| pass 1: gpu | 3.30 | 2.82 | 2.75 | 6.47 |
| pass 1: turn | 0.60 | 0.60 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.32 | 2.82 | 2.75 | 6.47 |
| pass 2: turn | 0.58 | 0.57 | 0.53 | 0.62 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.26 | 2.82 | 2.77 | 6.46 |
| pass 3: turn (the server before the next verify) | 2.12 | 1.85 | 1.06 | 2.71 |
| the draft's span (all passes) | 19.18 | 17.35 | 16.49 | 28.68 |
|   the draft's GPU time on its last card | 13.79 | 12.30 | 12.21 | 23.38 |
|   the leads (crossings) | 4.23 | 4.49 | 2.05 | 4.57 |
|   the turns of the passes before the last | -0.96 | -1.32 | -1.41 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.07 | 3.22 | 3.19 | 3.37 |

verify_gpu per card, median ms: 11.58, 11.71, 11.38, 12.00, 9.42, 10.91, 9.82, 10.65, 7.95

