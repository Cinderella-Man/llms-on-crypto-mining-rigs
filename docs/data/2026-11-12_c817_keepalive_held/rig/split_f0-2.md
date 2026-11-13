f0-2: 2890 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1394 evaluations and 84 main passes
evaluations between main passes, by card: card 4: 655, card 8: 655; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 84 | 6 | 77 | 4: 76, 8: 1 | 74 |

**f0-2, all requests: 154 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.24 | 114.36 | 106.12 | 119.67 |
| verify_span | 94.88 | 93.59 | 90.44 | 99.05 |
|   verify_gpu (the nine cards) | 91.25 | 89.92 | 87.01 | 95.36 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.67 | 3.44 | 3.75 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 0: gpu | 3.58 | 3.42 | 3.26 | 3.91 |
| pass 0: turn | -1.86 | -2.24 | -2.50 | 0.28 |
| pass 1 (draft 1): lead | 2.46 | 2.73 | 0.51 | 2.99 |
| pass 1: gpu | 3.23 | 2.75 | 2.48 | 6.27 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.24 | 2.76 | 2.48 | 6.23 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.49 | 0.52 |
| pass 3: gpu | 3.17 | 2.77 | 2.46 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.87 | 1.69 | 1.05 | 2.31 |
| the draft's span (all passes) | 18.37 | 16.71 | 15.33 | 27.17 |
|   the draft's GPU time on its last card | 13.22 | 12.12 | 10.71 | 22.07 |
|   the leads (crossings) | 3.99 | 4.26 | 2.04 | 4.52 |
|   the turns of the passes before the last | -0.71 | -1.12 | -1.38 | 1.50 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.21 | 3.14 | 3.35 |

verify_gpu per card, median ms: 11.14, 10.78, 10.91, 11.00, 9.06, 10.64, 8.95, 10.27, 7.34

**f0-2, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 109.71 | 107.33 | 105.79 | 118.82 |
| verify_span | 91.50 | 91.39 | 90.20 | 92.53 |
|   verify_gpu (the nine cards) | 87.99 | 87.81 | 86.78 | 89.07 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.51 | 3.45 | 3.43 | 3.81 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.32 | 3.30 | 3.24 | 3.37 |
| pass 0: turn | -1.66 | -2.17 | -2.25 | 0.30 |
| pass 1 (draft 1): lead | 2.27 | 2.67 | 0.51 | 2.75 |
| pass 1: gpu | 3.22 | 2.50 | 2.47 | 6.25 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.24 | 2.49 | 2.46 | 6.23 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.50 | 0.52 |
| pass 3: gpu | 3.15 | 2.49 | 2.46 | 6.23 |
| pass 3: turn (the server before the next verify) | 1.97 | 1.72 | 1.11 | 2.37 |
| the draft's span (all passes) | 18.20 | 15.77 | 15.07 | 27.00 |
|   the draft's GPU time on its last card | 12.94 | 10.78 | 10.67 | 22.05 |
|   the leads (crossings) | 3.81 | 4.20 | 2.05 | 4.28 |
|   the turns of the passes before the last | -0.51 | -1.03 | -1.14 | 1.52 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.18 | 3.12 | 3.30 |

verify_gpu per card, median ms: 10.95, 10.55, 10.70, 10.75, 8.81, 10.27, 8.63, 9.96, 7.16

**f0-2, long: 74 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.07 | 115.41 | 113.88 | 126.08 |
| verify_span | 98.52 | 98.28 | 97.05 | 99.46 |
|   verify_gpu (the nine cards) | 94.78 | 94.51 | 93.27 | 95.74 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.75 | 3.70 | 3.68 | 3.75 |
| pass 0 (catch-up): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 0: gpu | 3.87 | 3.85 | 3.78 | 3.94 |
| pass 0: turn | -2.08 | -2.44 | -2.53 | 0.28 |
| pass 1 (draft 1): lead | 2.65 | 2.94 | 0.52 | 3.02 |
| pass 1: gpu | 3.23 | 2.79 | 2.74 | 6.46 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.24 | 2.79 | 2.74 | 6.41 |
| pass 2: turn | 0.57 | 0.57 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.19 | 2.79 | 2.75 | 6.42 |
| pass 3: turn (the server before the next verify) | 1.76 | 1.64 | 0.99 | 1.99 |
| the draft's span (all passes) | 18.55 | 17.07 | 16.25 | 28.26 |
|   the draft's GPU time on its last card | 13.53 | 12.21 | 12.08 | 23.21 |
|   the leads (crossings) | 4.18 | 4.47 | 2.04 | 4.56 |
|   the turns of the passes before the last | -0.93 | -1.29 | -1.41 | 1.47 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.86 | 3.22 | 3.19 | 3.41 |

verify_gpu per card, median ms: 11.48, 11.62, 11.20, 11.82, 9.34, 10.83, 9.74, 10.59, 7.93

