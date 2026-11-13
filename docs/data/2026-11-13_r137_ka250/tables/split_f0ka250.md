f0ka250: 1870 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 374 evaluations and 24 main passes
evaluations between main passes, by card: card 4: 415, card 8: 415; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 24 | 6 | 17 | 4: 16, 8: 1 | 14 |

**f0ka250, all requests: 94 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.78 | 107.90 | 105.97 | 119.19 |
| verify_span | 92.43 | 91.53 | 90.32 | 97.41 |
|   verify_gpu (the nine cards) | 88.89 | 88.04 | 86.88 | 93.73 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.55 | 3.46 | 3.42 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 0: gpu | 3.39 | 3.31 | 3.26 | 3.79 |
| pass 0: turn | -1.70 | -2.19 | -2.29 | 0.28 |
| pass 1 (draft 1): lead | 2.33 | 2.69 | 0.51 | 2.78 |
| pass 1: gpu | 3.25 | 2.51 | 2.48 | 6.26 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.27 | 2.51 | 2.47 | 6.23 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.18 | 2.50 | 2.47 | 6.22 |
| pass 3: turn (the server before the next verify) | 1.94 | 1.75 | 1.11 | 2.17 |
| the draft's span (all passes) | 18.34 | 15.93 | 15.13 | 27.38 |
|   the draft's GPU time on its last card | 13.09 | 10.82 | 10.71 | 22.06 |
|   the leads (crossings) | 3.86 | 4.22 | 2.04 | 4.32 |
|   the turns of the passes before the last | -0.54 | -1.05 | -1.19 | 1.58 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.68 | 3.18 | 3.11 | 3.32 |

verify_gpu per card, median ms: 10.92, 10.61, 10.74, 10.78, 8.83, 10.28, 8.70, 9.98, 7.22

**f0ka250, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.65 | 107.48 | 105.73 | 118.84 |
| verify_span | 91.52 | 91.40 | 90.19 | 92.40 |
|   verify_gpu (the nine cards) | 88.01 | 87.77 | 86.78 | 88.94 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.45 | 3.42 | 3.84 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.25 | 3.36 |
| pass 0: turn | -1.67 | -2.19 | -2.26 | 0.28 |
| pass 1 (draft 1): lead | 2.30 | 2.68 | 0.51 | 2.75 |
| pass 1: gpu | 3.22 | 2.50 | 2.48 | 6.25 |
| pass 1: turn | 0.60 | 0.58 | 0.54 | 0.66 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.24 | 2.50 | 2.47 | 6.23 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.57 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.15 | 2.49 | 2.47 | 6.21 |
| pass 3: turn (the server before the next verify) | 1.87 | 1.76 | 1.11 | 2.17 |
| the draft's span (all passes) | 18.13 | 15.82 | 15.05 | 27.16 |
|   the draft's GPU time on its last card | 12.94 | 10.80 | 10.71 | 22.03 |
|   the leads (crossings) | 3.83 | 4.21 | 2.04 | 4.29 |
|   the turns of the passes before the last | -0.52 | -1.04 | -1.14 | 1.58 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.17 | 3.11 | 3.32 |

verify_gpu per card, median ms: 10.91, 10.58, 10.70, 10.75, 8.80, 10.23, 8.67, 9.98, 7.21

**f0ka250, long: 14 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.20 | 114.75 | 113.44 | 127.23 |
| verify_span | 97.64 | 97.56 | 96.02 | 99.10 |
|   verify_gpu (the nine cards) | 93.89 | 93.86 | 92.35 | 95.23 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.75 | 3.69 | 3.67 | 4.13 |
| pass 0 (catch-up): lead | 0.52 | 0.52 | 0.51 | 0.55 |
| pass 0: gpu | 3.77 | 3.79 | 3.68 | 3.84 |
| pass 0: turn | -1.89 | -2.32 | -2.43 | 0.28 |
| pass 1 (draft 1): lead | 2.48 | 2.81 | 0.51 | 2.91 |
| pass 1: gpu | 3.37 | 2.78 | 2.74 | 6.44 |
| pass 1: turn | 0.60 | 0.60 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.44 | 2.78 | 2.74 | 6.43 |
| pass 2: turn | 0.61 | 0.58 | 0.54 | 0.62 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.52 |
| pass 3: gpu | 3.35 | 2.79 | 2.77 | 6.45 |
| pass 3: turn (the server before the next verify) | 2.30 | 1.65 | 1.06 | 3.81 |
| the draft's span (all passes) | 19.56 | 17.17 | 16.63 | 27.88 |
|   the draft's GPU time on its last card | 13.93 | 12.15 | 12.00 | 23.13 |
|   the leads (crossings) | 4.01 | 4.35 | 2.05 | 4.45 |
|   the turns of the passes before the last | -0.68 | -1.17 | -1.26 | 1.53 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.09 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 3.25 | 3.24 | 3.18 | 3.35 |

verify_gpu per card, median ms: 11.33, 11.54, 11.27, 11.72, 9.25, 10.86, 9.67, 10.53, 7.78

