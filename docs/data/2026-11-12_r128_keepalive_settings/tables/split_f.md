f: 2958 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1462 evaluations and 88 main passes
evaluations between main passes, by card: card 4: 671, card 8: 671; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 88 | 6 | 81 | 4: 80, 8: 1 | 78 |

**f, all requests: 158 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.51 | 114.69 | 106.69 | 119.56 |
| verify_span | 95.52 | 96.12 | 90.98 | 99.82 |
|   verify_gpu (the nine cards) | 91.88 | 92.46 | 87.49 | 96.13 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.67 | 3.43 | 3.73 |
| pass 0 (catch-up): lead | 0.53 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.60 | 3.73 | 3.28 | 3.91 |
| pass 0: turn | -2.02 | -2.28 | -2.52 | 0.27 |
| pass 1 (draft 1): lead | 2.58 | 2.77 | 0.57 | 3.01 |
| pass 1: gpu | 3.06 | 2.73 | 2.49 | 3.54 |
| pass 1: turn | 0.60 | 0.59 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.08 | 2.75 | 2.49 | 3.47 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.03 | 2.74 | 2.49 | 3.38 |
| pass 3: turn (the server before the next verify) | 1.96 | 1.76 | 1.02 | 2.55 |
| the draft's span (all passes) | 17.99 | 16.57 | 15.38 | 26.95 |
|   the draft's GPU time on its last card | 12.77 | 12.10 | 10.78 | 14.50 |
|   the leads (crossings) | 4.12 | 4.31 | 2.20 | 4.52 |
|   the turns of the passes before the last | -0.86 | -1.16 | -1.39 | 1.39 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.92 | 3.20 | 3.12 | 3.33 |

verify_gpu per card, median ms: 11.26, 11.38, 10.96, 11.40, 9.18, 10.64, 9.35, 10.39, 7.70

**f, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.30 | 108.09 | 106.47 | 119.41 |
| verify_span | 92.04 | 91.87 | 90.74 | 93.10 |
|   verify_gpu (the nine cards) | 88.47 | 88.27 | 87.27 | 89.63 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.57 | 3.44 | 3.42 | 3.83 |
| pass 0 (catch-up): lead | 0.54 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.34 | 3.31 | 3.27 | 3.38 |
| pass 0: turn | -1.68 | -2.20 | -2.27 | 0.29 |
| pass 1 (draft 1): lead | 2.29 | 2.69 | 0.50 | 2.77 |
| pass 1: gpu | 3.23 | 2.51 | 2.48 | 6.27 |
| pass 1: turn | 0.61 | 0.60 | 0.54 | 0.66 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.26 | 2.51 | 2.49 | 6.25 |
| pass 2: turn | 0.55 | 0.55 | 0.52 | 0.56 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.17 | 2.50 | 2.47 | 6.24 |
| pass 3: turn (the server before the next verify) | 1.95 | 1.84 | 1.10 | 2.54 |
| the draft's span (all passes) | 18.27 | 16.00 | 15.06 | 27.17 |
|   the draft's GPU time on its last card | 12.99 | 10.83 | 10.76 | 22.14 |
|   the leads (crossings) | 3.84 | 4.21 | 2.03 | 4.30 |
|   the turns of the passes before the last | -0.52 | -1.05 | -1.14 | 1.57 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.77 | 3.16 | 3.11 | 3.31 |

verify_gpu per card, median ms: 11.03, 10.62, 10.75, 10.81, 8.86, 10.27, 8.68, 10.01, 7.23

**f, long: 78 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 116.81 | 115.68 | 113.98 | 119.79 |
| verify_span | 99.10 | 98.85 | 97.32 | 100.30 |
|   verify_gpu (the nine cards) | 95.38 | 95.15 | 93.63 | 96.59 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.72 | 3.69 | 3.67 | 3.73 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.88 | 3.85 | 3.78 | 3.92 |
| pass 0: turn | -2.38 | -2.46 | -2.54 | -2.38 |
| pass 1 (draft 1): lead | 2.89 | 2.95 | 2.87 | 3.02 |
| pass 1: gpu | 2.89 | 2.75 | 2.73 | 2.82 |
| pass 1: turn | 0.59 | 0.59 | 0.53 | 0.64 |
| pass 2 (draft 2): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 2: gpu | 2.90 | 2.78 | 2.73 | 2.81 |
| pass 2: turn | 0.58 | 0.57 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 2.88 | 2.78 | 2.73 | 2.82 |
| pass 3: turn (the server before the next verify) | 1.96 | 1.39 | 0.96 | 2.69 |
| the draft's span (all passes) | 17.70 | 16.86 | 16.19 | 19.08 |
|   the draft's GPU time on its last card | 12.55 | 12.18 | 12.07 | 12.29 |
|   the leads (crossings) | 4.40 | 4.46 | 4.38 | 4.54 |
|   the turns of the passes before the last | -1.21 | -1.32 | -1.41 | -1.20 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.08 | 3.21 | 3.17 | 3.34 |

verify_gpu per card, median ms: 11.54, 11.77, 11.31, 11.90, 9.45, 10.85, 9.81, 10.68, 7.84

