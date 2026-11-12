b-2: 3765 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1923 evaluations and 94 main passes
evaluations between main passes, by card: card 4: 696, card 8: 1392; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 94 | 6 | 87 | 8: 86, 16: 1 | 84 |

**b-2, all requests: 165 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 116.44 | 117.61 | 108.94 | 123.07 |
| verify_span | 96.61 | 98.43 | 92.05 | 100.61 |
|   verify_gpu (the nine cards) | 91.97 | 93.70 | 87.49 | 95.76 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.64 | 4.67 | 4.46 | 4.79 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.61 | 3.78 | 3.27 | 3.92 |
| pass 1: turn | 0.43 | 0.36 | 0.35 | 0.43 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.34 | 2.78 | 2.52 | 6.45 |
| pass 3: turn | 0.60 | 0.58 | 0.55 | 0.64 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.38 | 2.74 | 2.51 | 6.41 |
| pass 5: turn | 0.56 | 0.56 | 0.53 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.20 | 2.74 | 2.52 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.65 | 2.35 | 2.09 | 2.66 |
| the draft's span (all passes) | 19.84 | 18.17 | 16.66 | 29.26 |
|   the draft's GPU time on its last card | 13.67 | 12.31 | 10.97 | 23.33 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.84 |
|   the turns of the passes before the last | 1.70 | 1.62 | 1.55 | 1.78 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.73 | 3.19 | 3.11 | 3.28 |

verify_gpu per card, median ms: 11.28, 11.46, 11.07, 11.66, 9.22, 10.66, 9.59, 10.46, 7.77

**b-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 112.68 | 110.13 | 108.65 | 122.15 |
| verify_span | 93.01 | 92.79 | 91.72 | 94.20 |
|   verify_gpu (the nine cards) | 88.49 | 88.33 | 87.18 | 89.67 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.53 | 4.50 | 4.45 | 4.58 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.33 | 3.30 | 3.25 | 3.36 |
| pass 1: turn | 0.44 | 0.36 | 0.35 | 0.48 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.36 | 2.56 | 2.49 | 6.31 |
| pass 3: turn | 0.60 | 0.57 | 0.54 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.39 | 2.54 | 2.49 | 6.27 |
| pass 5: turn | 0.55 | 0.55 | 0.53 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.18 | 2.54 | 2.49 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.76 | 2.35 | 2.10 | 2.66 |
| the draft's span (all passes) | 19.67 | 16.93 | 16.53 | 29.26 |
|   the draft's GPU time on its last card | 13.39 | 11.09 | 10.94 | 22.36 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.84 |
|   the turns of the passes before the last | 1.70 | 1.62 | 1.54 | 2.09 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.72 | 3.15 | 3.10 | 3.27 |

verify_gpu per card, median ms: 11.02, 10.60, 10.75, 10.81, 8.86, 10.28, 8.70, 10.02, 7.21

**b-2, long: 84 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 120.07 | 118.22 | 116.96 | 129.42 |
| verify_span | 100.07 | 99.90 | 98.91 | 101.05 |
|   verify_gpu (the nine cards) | 95.32 | 95.18 | 94.20 | 96.14 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.75 | 4.72 | 4.68 | 4.81 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.89 | 3.87 | 3.81 | 3.95 |
| pass 1: turn | 0.42 | 0.36 | 0.36 | 0.38 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.31 | 2.83 | 2.77 | 6.45 |
| pass 3: turn | 0.60 | 0.59 | 0.55 | 0.63 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.37 | 2.79 | 2.73 | 6.41 |
| pass 5: turn | 0.58 | 0.56 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.22 | 2.79 | 2.74 | 6.41 |
| pass 7: turn (the server before the next verify) | 2.55 | 2.36 | 2.09 | 2.65 |
| the draft's span (all passes) | 20.00 | 18.25 | 17.95 | 29.24 |
|   the draft's GPU time on its last card | 13.93 | 12.41 | 12.25 | 23.43 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.84 |
|   the turns of the passes before the last | 1.71 | 1.62 | 1.56 | 1.75 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.20 | 3.17 | 3.28 |

verify_gpu per card, median ms: 11.56, 11.72, 11.29, 11.93, 9.43, 10.84, 9.83, 10.71, 7.88

