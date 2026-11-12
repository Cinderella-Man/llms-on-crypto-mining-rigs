e771: 3028 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1487 evaluations and 89 main passes, 1530 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 688, card 8: 688; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 4: 84 | 81 |
| long | 92 | 6 | 85 | 4: 84, 8: 1 | 82 |

**e771, all requests: 163 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.29 | 115.65 | 106.71 | 121.50 |
| verify_span | 95.48 | 97.11 | 90.70 | 99.65 |
|   verify_gpu (the nine cards) | 91.86 | 93.38 | 87.27 | 95.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.66 | 3.43 | 3.74 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.61 | 3.77 | 3.28 | 3.91 |
| pass 0: turn | -1.89 | -2.25 | -2.54 | 0.31 |
| pass 1 (draft 1): lead | 2.49 | 2.76 | 0.51 | 3.03 |
| pass 1: gpu | 3.23 | 2.79 | 2.50 | 6.28 |
| pass 1: turn | 0.61 | 0.59 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.24 | 2.78 | 2.49 | 6.27 |
| pass 2: turn | 0.57 | 0.56 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.75 | 2.49 | 6.24 |
| pass 3: turn (the server before the next verify) | 2.24 | 2.01 | 1.29 | 2.65 |
| the draft's span (all passes) | 18.81 | 17.21 | 15.63 | 27.98 |
|   the draft's GPU time on its last card | 13.26 | 12.16 | 10.78 | 22.16 |
|   the leads (crossings) | 4.01 | 4.29 | 2.05 | 4.55 |
|   the turns of the passes before the last | -0.71 | -1.14 | -1.39 | 1.55 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.19 | 3.12 | 3.31 |

verify_gpu per card, median ms: 11.35, 11.43, 11.12, 11.57, 9.20, 10.68, 9.54, 10.41, 7.78

**e771, greedy: 81 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.46 | 108.10 | 106.43 | 119.74 |
| verify_span | 91.77 | 91.64 | 90.50 | 92.75 |
|   verify_gpu (the nine cards) | 88.26 | 88.14 | 87.08 | 89.26 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.50 | 3.45 | 3.42 | 3.83 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.51 |
| pass 0: gpu | 3.34 | 3.32 | 3.26 | 3.38 |
| pass 0: turn | -1.64 | -2.20 | -2.25 | 0.36 |
| pass 1 (draft 1): lead | 2.27 | 2.69 | 0.50 | 2.76 |
| pass 1: gpu | 3.23 | 2.51 | 2.49 | 6.27 |
| pass 1: turn | 0.63 | 0.59 | 0.53 | 0.66 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.54 |
| pass 2: gpu | 3.25 | 2.51 | 2.47 | 6.27 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.60 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.16 | 2.51 | 2.48 | 6.24 |
| pass 3: turn (the server before the next verify) | 2.39 | 2.01 | 1.31 | 2.78 |
| the draft's span (all passes) | 18.69 | 16.14 | 15.34 | 27.73 |
|   the draft's GPU time on its last card | 12.97 | 10.84 | 10.76 | 22.15 |
|   the leads (crossings) | 3.79 | 4.21 | 2.04 | 4.29 |
|   the turns of the passes before the last | -0.46 | -1.03 | -1.14 | 1.78 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.17 | 3.09 | 3.30 |

verify_gpu per card, median ms: 10.93, 10.60, 10.77, 10.82, 8.79, 10.28, 8.66, 9.98, 7.24

**e771, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.06 | 116.52 | 114.79 | 127.47 |
| verify_span | 99.15 | 98.83 | 97.74 | 100.06 |
|   verify_gpu (the nine cards) | 95.42 | 95.11 | 94.09 | 96.37 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.73 | 3.69 | 3.66 | 3.74 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.89 | 3.87 | 3.79 | 3.94 |
| pass 0: turn | -2.14 | -2.48 | -2.57 | 0.31 |
| pass 1 (draft 1): lead | 2.71 | 2.97 | 0.54 | 3.05 |
| pass 1: gpu | 3.23 | 2.80 | 2.74 | 6.43 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.24 | 2.80 | 2.74 | 6.40 |
| pass 2: turn | 0.58 | 0.57 | 0.54 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.19 | 2.79 | 2.74 | 6.40 |
| pass 3: turn (the server before the next verify) | 2.10 | 2.01 | 1.27 | 2.31 |
| the draft's span (all passes) | 18.92 | 17.43 | 16.62 | 28.60 |
|   the draft's GPU time on its last card | 13.55 | 12.24 | 12.12 | 23.18 |
|   the leads (crossings) | 4.23 | 4.48 | 2.08 | 4.58 |
|   the turns of the passes before the last | -0.96 | -1.31 | -1.42 | 1.48 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.20 | 3.17 | 3.32 |

verify_gpu per card, median ms: 11.59, 11.75, 11.30, 11.91, 9.39, 10.88, 9.83, 10.62, 7.91

