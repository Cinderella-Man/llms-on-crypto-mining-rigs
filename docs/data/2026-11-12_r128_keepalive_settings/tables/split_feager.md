feager: 3026 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1530 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 687, card 8: 687; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 79 |
| long | 92 | 6 | 85 | 4: 84, 8: 1 | 82 |

**feager, all requests: 161 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 161.01 | 122.59 | 115.16 | 227.37 |
| verify_span | 97.22 | 97.63 | 90.85 | 99.54 |
|   verify_gpu (the nine cards) | 91.85 | 93.75 | 87.40 | 95.79 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.37 | 3.69 | 3.45 | 3.84 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.64 | 3.75 | 3.29 | 3.92 |
| pass 0: turn | 19.62 | -2.18 | -2.49 | 111.20 |
| pass 1 (draft 1): lead | 1.90 | 2.67 | 0.52 | 2.99 |
| pass 1: gpu | 3.32 | 2.80 | 2.50 | 6.37 |
| pass 1: turn | 0.65 | 0.58 | 0.54 | 0.91 |
| pass 2 (draft 2): lead | 0.54 | 0.51 | 0.50 | 0.65 |
| pass 2: gpu | 3.37 | 2.79 | 2.49 | 6.29 |
| pass 2: turn | 0.62 | 0.57 | 0.53 | 0.88 |
| pass 3 (draft 3): lead | 0.53 | 0.51 | 0.50 | 0.63 |
| pass 3: gpu | 3.22 | 2.78 | 2.50 | 6.28 |
| pass 3: turn (the server before the next verify) | 25.87 | 2.23 | 1.12 | 106.13 |
| the draft's span (all passes) | 63.78 | 25.77 | 16.73 | 133.66 |
|   the draft's GPU time on its last card | 13.55 | 12.23 | 10.81 | 22.29 |
|   the leads (crossings) | 3.48 | 4.19 | 2.05 | 4.52 |
|   the turns of the passes before the last | 20.89 | -1.04 | -1.32 | 112.21 |
|   the draft's GPU time on the embedding card | 0.10 | 0.09 | 0.08 | 0.14 |
| the host's launch of the main pass (inside verify_span) | 5.26 | 3.34 | 3.25 | 6.12 |

verify_gpu per card, median ms: 11.33, 11.53, 11.12, 11.63, 9.21, 10.68, 9.59, 10.44, 7.73

**feager, greedy: 79 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 205.45 | 214.93 | 118.57 | 248.95 |
| verify_span | 95.44 | 91.76 | 90.62 | 93.12 |
|   verify_gpu (the nine cards) | 88.39 | 88.16 | 87.20 | 89.32 |
|   verify_rest (crossings, rows back, acceptance, server) | 7.05 | 3.47 | 3.43 | 3.87 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 0: gpu | 3.36 | 3.31 | 3.28 | 3.39 |
| pass 0: turn | 41.39 | -1.78 | -2.23 | 117.01 |
| pass 1 (draft 1): lead | 1.69 | 2.28 | 0.59 | 2.73 |
| pass 1: gpu | 3.33 | 2.53 | 2.49 | 6.29 |
| pass 1: turn | 0.71 | 0.57 | 0.54 | 0.98 |
| pass 2 (draft 2): lead | 0.57 | 0.52 | 0.50 | 0.66 |
| pass 2: gpu | 3.41 | 2.54 | 2.48 | 6.27 |
| pass 2: turn | 0.66 | 0.56 | 0.52 | 0.90 |
| pass 3 (draft 3): lead | 0.55 | 0.51 | 0.50 | 0.65 |
| pass 3: gpu | 3.19 | 2.53 | 2.49 | 6.27 |
| pass 3: turn (the server before the next verify) | 50.63 | 2.90 | 1.82 | 108.22 |
| the draft's span (all passes) | 110.01 | 121.01 | 21.05 | 143.08 |
|   the draft's GPU time on its last card | 13.30 | 10.89 | 10.78 | 22.24 |
|   the leads (crossings) | 3.32 | 3.79 | 2.10 | 4.25 |
|   the turns of the passes before the last | 42.76 | -0.69 | -1.13 | 118.90 |
|   the draft's GPU time on the embedding card | 0.11 | 0.10 | 0.08 | 0.16 |
| the host's launch of the main pass (inside verify_span) | 6.44 | 5.86 | 3.28 | 6.66 |

verify_gpu per card, median ms: 10.97, 10.61, 10.77, 10.82, 8.82, 10.27, 8.65, 10.00, 7.24

**feager, long: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.19 | 116.48 | 115.12 | 126.96 |
| verify_span | 98.94 | 98.82 | 97.78 | 99.69 |
|   verify_gpu (the nine cards) | 95.18 | 95.11 | 94.09 | 95.94 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.76 | 3.72 | 3.69 | 3.76 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.90 | 3.88 | 3.79 | 3.93 |
| pass 0: turn | -1.35 | -2.39 | -2.52 | 0.77 |
| pass 1 (draft 1): lead | 2.10 | 2.89 | 0.51 | 3.02 |
| pass 1: gpu | 3.30 | 2.81 | 2.75 | 6.45 |
| pass 1: turn | 0.59 | 0.59 | 0.54 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.34 | 2.80 | 2.75 | 6.44 |
| pass 2: turn | 0.58 | 0.57 | 0.54 | 0.61 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.50 | 0.52 |
| pass 3: gpu | 3.25 | 2.81 | 2.75 | 6.45 |
| pass 3: turn (the server before the next verify) | 2.01 | 1.83 | 1.02 | 2.56 |
| the draft's span (all passes) | 19.25 | 17.47 | 16.63 | 27.77 |
|   the draft's GPU time on its last card | 13.79 | 12.29 | 12.17 | 23.28 |
|   the leads (crossings) | 3.63 | 4.42 | 2.04 | 4.56 |
|   the turns of the passes before the last | -0.18 | -1.23 | -1.37 | 1.91 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 4.13 | 3.29 | 3.24 | 3.42 |

verify_gpu per card, median ms: 11.59, 11.71, 11.28, 11.91, 9.41, 10.85, 9.82, 10.62, 7.95

