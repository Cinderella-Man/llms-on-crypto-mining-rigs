f1ka250: 2924 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1428 evaluations and 86 main passes
evaluations between main passes, by card: card 4: 663, card 8: 663; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 86 | 7 | 78 | 4: 78 | 73 |

**f1ka250, all requests: 151 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.99 | 115.13 | 106.21 | 123.01 |
| verify_span | 95.10 | 93.25 | 90.46 | 99.85 |
|   verify_gpu (the nine cards) | 91.49 | 89.59 | 87.02 | 96.10 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.61 | 3.67 | 3.43 | 3.80 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.57 | 3.41 | 3.27 | 3.87 |
| pass 0: turn | -1.82 | -2.23 | -2.47 | 0.28 |
| pass 1 (draft 1): lead | 2.45 | 2.74 | 0.51 | 2.98 |
| pass 1: gpu | 3.55 | 2.74 | 2.48 | 6.27 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.32 | 2.76 | 2.48 | 6.27 |
| pass 2: turn | 0.58 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.54 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.22 | 2.75 | 2.48 | 6.25 |
| pass 3: turn (the server before the next verify) | 1.85 | 1.69 | 1.07 | 2.27 |
| the draft's span (all passes) | 18.89 | 16.86 | 15.37 | 27.55 |
|   the draft's GPU time on its last card | 13.67 | 12.10 | 10.73 | 22.17 |
|   the leads (crossings) | 4.02 | 4.27 | 2.05 | 4.53 |
|   the turns of the passes before the last | -0.65 | -1.08 | -1.34 | 1.45 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.63 | 3.26 | 3.17 | 3.34 |

verify_gpu per card, median ms: 11.10, 10.83, 10.97, 10.98, 9.10, 10.59, 8.95, 10.34, 7.31

**f1ka250, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.97 | 107.61 | 105.81 | 118.80 |
| verify_span | 91.38 | 91.37 | 90.17 | 92.46 |
|   verify_gpu (the nine cards) | 87.89 | 87.86 | 86.75 | 88.98 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.49 | 3.45 | 3.42 | 3.49 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.31 | 3.25 | 3.35 |
| pass 0: turn | -1.64 | -2.16 | -2.25 | 0.28 |
| pass 1 (draft 1): lead | 2.28 | 2.66 | 0.51 | 2.74 |
| pass 1: gpu | 3.52 | 2.49 | 2.46 | 6.24 |
| pass 1: turn | 0.61 | 0.60 | 0.53 | 0.66 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.29 | 2.50 | 2.47 | 6.24 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.17 | 2.50 | 2.47 | 6.21 |
| pass 3: turn (the server before the next verify) | 1.94 | 1.73 | 1.11 | 2.50 |
| the draft's span (all passes) | 18.59 | 15.92 | 15.17 | 27.08 |
|   the draft's GPU time on its last card | 13.31 | 10.79 | 10.71 | 22.01 |
|   the leads (crossings) | 3.81 | 4.21 | 2.04 | 4.27 |
|   the turns of the passes before the last | -0.47 | -0.99 | -1.11 | 1.45 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.61 | 3.20 | 3.15 | 3.33 |

verify_gpu per card, median ms: 10.92, 10.57, 10.70, 10.74, 8.81, 10.22, 8.66, 9.98, 7.18

**f1ka250, long: 73 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.29 | 116.46 | 114.94 | 127.91 |
| verify_span | 99.08 | 98.97 | 97.93 | 100.27 |
|   verify_gpu (the nine cards) | 95.34 | 95.28 | 94.20 | 96.45 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.71 | 3.68 | 3.80 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.51 | 0.54 |
| pass 0: gpu | 3.85 | 3.83 | 3.76 | 3.92 |
| pass 0: turn | -2.02 | -2.42 | -2.51 | 0.27 |
| pass 1 (draft 1): lead | 2.63 | 2.92 | 0.52 | 3.01 |
| pass 1: gpu | 3.58 | 2.78 | 2.73 | 6.45 |
| pass 1: turn | 0.58 | 0.59 | 0.53 | 0.62 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.35 | 2.79 | 2.74 | 6.42 |
| pass 2: turn | 0.59 | 0.56 | 0.54 | 0.59 |
| pass 3 (draft 3): lead | 0.58 | 0.51 | 0.49 | 0.52 |
| pass 3: gpu | 3.27 | 2.78 | 2.74 | 6.42 |
| pass 3: turn (the server before the next verify) | 1.76 | 1.66 | 1.03 | 1.98 |
| the draft's span (all passes) | 19.21 | 17.04 | 16.37 | 28.09 |
|   the draft's GPU time on its last card | 14.05 | 12.15 | 12.06 | 23.20 |
|   the leads (crossings) | 4.24 | 4.44 | 2.08 | 4.57 |
|   the turns of the passes before the last | -0.84 | -1.28 | -1.37 | 1.44 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.66 | 3.27 | 3.24 | 3.37 |

verify_gpu per card, median ms: 11.47, 11.67, 11.36, 11.94, 9.44, 11.03, 9.83, 10.73, 7.87

