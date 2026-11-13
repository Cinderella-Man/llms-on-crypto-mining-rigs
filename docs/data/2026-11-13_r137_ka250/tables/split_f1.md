f1: 2958 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1462 evaluations and 88 main passes
evaluations between main passes, by card: card 4: 671, card 8: 671; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 88 | 7 | 80 | 4: 80 | 76 |

**f1, all requests: 154 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.03 | 114.40 | 106.29 | 118.84 |
| verify_span | 94.79 | 94.62 | 90.49 | 99.08 |
|   verify_gpu (the nine cards) | 91.19 | 91.03 | 87.05 | 95.37 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.67 | 3.43 | 3.74 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.58 | 3.54 | 3.26 | 3.89 |
| pass 0: turn | -1.99 | -2.27 | -2.50 | 0.27 |
| pass 1 (draft 1): lead | 2.56 | 2.76 | 0.54 | 2.99 |
| pass 1: gpu | 3.25 | 2.73 | 2.48 | 6.22 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.11 | 2.74 | 2.47 | 6.19 |
| pass 2: turn | 0.57 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.04 | 2.74 | 2.47 | 6.19 |
| pass 3: turn (the server before the next verify) | 1.98 | 1.72 | 1.10 | 2.36 |
| the draft's span (all passes) | 18.24 | 16.84 | 15.34 | 27.06 |
|   the draft's GPU time on its last card | 12.98 | 12.11 | 10.73 | 21.93 |
|   the leads (crossings) | 4.10 | 4.31 | 2.10 | 4.54 |
|   the turns of the passes before the last | -0.82 | -1.14 | -1.35 | 1.38 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.62 | 3.24 | 3.16 | 3.35 |

verify_gpu per card, median ms: 11.11, 11.03, 10.98, 11.19, 9.10, 10.61, 9.14, 10.31, 7.45

**f1, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.15 | 107.66 | 105.90 | 119.01 |
| verify_span | 91.36 | 91.32 | 90.26 | 92.39 |
|   verify_gpu (the nine cards) | 87.87 | 87.82 | 86.82 | 88.92 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.49 | 3.45 | 3.42 | 3.50 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.25 | 3.35 |
| pass 0: turn | -1.63 | -2.19 | -2.24 | 0.28 |
| pass 1 (draft 1): lead | 2.25 | 2.69 | 0.51 | 2.74 |
| pass 1: gpu | 3.55 | 2.49 | 2.48 | 6.25 |
| pass 1: turn | 0.60 | 0.59 | 0.54 | 0.65 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 2: gpu | 3.29 | 2.49 | 2.47 | 6.23 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.17 | 2.49 | 2.45 | 6.21 |
| pass 3: turn (the server before the next verify) | 2.15 | 1.80 | 1.12 | 2.44 |
| the draft's span (all passes) | 18.79 | 15.83 | 15.13 | 27.31 |
|   the draft's GPU time on its last card | 13.32 | 10.78 | 10.70 | 22.07 |
|   the leads (crossings) | 3.79 | 4.22 | 2.05 | 4.29 |
|   the turns of the passes before the last | -0.47 | -1.02 | -1.12 | 1.44 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.61 | 3.20 | 3.14 | 3.32 |

verify_gpu per card, median ms: 10.90, 10.57, 10.70, 10.76, 8.80, 10.23, 8.67, 9.97, 7.19

**f1, long: 76 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 115.99 | 115.30 | 114.03 | 117.21 |
| verify_span | 98.32 | 98.31 | 97.19 | 99.31 |
|   verify_gpu (the nine cards) | 94.61 | 94.56 | 93.50 | 95.62 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.71 | 3.70 | 3.67 | 3.74 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.86 | 3.85 | 3.77 | 3.90 |
| pass 0: turn | -2.36 | -2.44 | -2.52 | -2.35 |
| pass 1 (draft 1): lead | 2.88 | 2.94 | 2.86 | 3.02 |
| pass 1: gpu | 2.94 | 2.77 | 2.72 | 2.81 |
| pass 1: turn | 0.60 | 0.60 | 0.55 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 2.93 | 2.77 | 2.73 | 2.81 |
| pass 2: turn | 0.58 | 0.58 | 0.54 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 2.91 | 2.77 | 2.73 | 2.80 |
| pass 3: turn (the server before the next verify) | 1.81 | 1.67 | 1.08 | 2.04 |
| the draft's span (all passes) | 17.68 | 17.01 | 16.40 | 18.38 |
|   the draft's GPU time on its last card | 12.63 | 12.15 | 12.07 | 12.25 |
|   the leads (crossings) | 4.42 | 4.48 | 4.39 | 4.56 |
|   the turns of the passes before the last | -1.19 | -1.28 | -1.38 | -1.16 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.63 | 3.27 | 3.23 | 3.39 |

verify_gpu per card, median ms: 11.48, 11.65, 11.31, 11.87, 9.36, 10.87, 9.79, 10.62, 7.74

