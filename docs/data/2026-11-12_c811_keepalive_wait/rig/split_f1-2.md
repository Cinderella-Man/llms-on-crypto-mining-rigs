f1-2: 3077 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1581 evaluations and 95 main passes
evaluations between main passes, by card: card 4: 699, card 8: 699; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 5 | 83 | 3: 1, 4: 82 | 78 |
| long | 95 | 7 | 87 | 4: 87 | 83 |

**f1-2, all requests: 161 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.98 | 115.56 | 106.58 | 119.88 |
| verify_span | 95.58 | 97.31 | 90.88 | 99.88 |
|   verify_gpu (the nine cards) | 91.98 | 93.60 | 87.42 | 96.17 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.66 | 3.43 | 3.72 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.63 | 3.79 | 3.28 | 3.94 |
| pass 0: turn | -2.01 | -2.32 | -2.56 | 0.31 |
| pass 1 (draft 1): lead | 2.58 | 2.82 | 0.52 | 3.06 |
| pass 1: gpu | 3.32 | 2.80 | 2.49 | 6.25 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.17 | 2.80 | 2.49 | 6.22 |
| pass 2: turn | 0.57 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.10 | 2.78 | 2.49 | 6.21 |
| pass 3: turn (the server before the next verify) | 1.90 | 1.80 | 1.10 | 2.55 |
| the draft's span (all passes) | 18.40 | 16.97 | 15.42 | 27.04 |
|   the draft's GPU time on its last card | 13.21 | 12.26 | 10.76 | 22.06 |
|   the leads (crossings) | 4.11 | 4.35 | 2.06 | 4.59 |
|   the turns of the passes before the last | -0.83 | -1.19 | -1.41 | 1.45 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.55 | 3.22 | 3.13 | 3.34 |

verify_gpu per card, median ms: 11.28, 11.38, 11.12, 11.56, 9.20, 10.70, 9.57, 10.40, 7.82

**f1-2, greedy: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.35 | 108.00 | 106.35 | 119.42 |
| verify_span | 91.74 | 91.72 | 90.58 | 92.77 |
|   verify_gpu (the nine cards) | 88.25 | 88.21 | 87.14 | 89.31 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.49 | 3.45 | 3.43 | 3.50 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.33 | 3.32 | 3.27 | 3.37 |
| pass 0: turn | -1.63 | -2.20 | -2.28 | 0.31 |
| pass 1 (draft 1): lead | 2.26 | 2.69 | 0.51 | 2.78 |
| pass 1: gpu | 3.56 | 2.51 | 2.47 | 6.27 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.30 | 2.51 | 2.48 | 6.25 |
| pass 2: turn | 0.57 | 0.55 | 0.51 | 0.58 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.18 | 2.51 | 2.47 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.91 | 1.80 | 1.11 | 2.55 |
| the draft's span (all passes) | 18.62 | 15.99 | 15.12 | 27.62 |
|   the draft's GPU time on its last card | 13.37 | 10.84 | 10.74 | 22.09 |
|   the leads (crossings) | 3.80 | 4.23 | 2.05 | 4.31 |
|   the turns of the passes before the last | -0.46 | -1.06 | -1.17 | 1.51 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.53 | 3.17 | 3.12 | 3.27 |

verify_gpu per card, median ms: 11.01, 10.61, 10.75, 10.81, 8.80, 10.26, 8.68, 10.02, 7.21

**f1-2, long: 83 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.39 | 116.47 | 114.80 | 121.24 |
| verify_span | 99.20 | 99.13 | 97.74 | 100.23 |
|   verify_gpu (the nine cards) | 95.50 | 95.45 | 94.10 | 96.54 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.70 | 3.70 | 3.66 | 3.72 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.90 | 3.90 | 3.82 | 3.95 |
| pass 0: turn | -2.35 | -2.51 | -2.59 | -2.43 |
| pass 1 (draft 1): lead | 2.88 | 3.01 | 2.92 | 3.08 |
| pass 1: gpu | 3.09 | 2.81 | 2.76 | 2.84 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 2: gpu | 3.05 | 2.81 | 2.77 | 2.85 |
| pass 2: turn | 0.57 | 0.58 | 0.52 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.01 | 2.80 | 2.76 | 2.84 |
| pass 3: turn (the server before the next verify) | 1.90 | 1.79 | 1.08 | 2.53 |
| the draft's span (all passes) | 18.19 | 17.29 | 16.55 | 19.43 |
|   the draft's GPU time on its last card | 13.06 | 12.30 | 12.20 | 12.45 |
|   the leads (crossings) | 4.41 | 4.54 | 4.45 | 4.62 |
|   the turns of the passes before the last | -1.18 | -1.35 | -1.48 | -1.23 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.57 | 3.23 | 3.19 | 3.36 |

verify_gpu per card, median ms: 11.53, 11.73, 11.39, 12.00, 9.41, 10.91, 9.83, 10.66, 7.95

