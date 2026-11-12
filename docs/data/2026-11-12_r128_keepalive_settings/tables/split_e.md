e: 3672 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1830 evaluations and 90 main passes
evaluations between main passes, by card: card 4: 680, card 8: 1360; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 90 | 6 | 83 | 8: 82, 16: 1 | 80 |

**e, all requests: 161 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.78 | 115.55 | 107.01 | 123.54 |
| verify_span | 95.47 | 97.03 | 90.87 | 99.60 |
|   verify_gpu (the nine cards) | 91.83 | 93.36 | 87.40 | 95.87 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.69 | 3.44 | 3.84 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.58 | 3.73 | 3.24 | 3.90 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.31 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.47 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.35 | 2.75 | 2.51 | 6.30 |
| pass 3: turn | 0.62 | 0.60 | 0.55 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.36 | 2.76 | 2.51 | 6.28 |
| pass 5: turn | 0.58 | 0.57 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.29 | 2.74 | 2.51 | 6.28 |
| pass 7: turn (the server before the next verify) | 2.25 | 1.96 | 1.30 | 2.62 |
| the draft's span (all passes) | 19.31 | 17.13 | 15.79 | 28.58 |
|   the draft's GPU time on its last card | 13.73 | 12.22 | 10.91 | 22.33 |
|   the leads (crossings) | 1.84 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.49 | 1.37 | 1.29 | 1.94 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.79 | 3.21 | 3.13 | 3.35 |

verify_gpu per card, median ms: 11.25, 11.40, 11.05, 11.47, 9.18, 10.68, 9.48, 10.39, 7.68

**e, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.73 | 108.32 | 106.56 | 119.66 |
| verify_span | 91.90 | 91.74 | 90.58 | 92.97 |
|   verify_gpu (the nine cards) | 88.39 | 88.20 | 87.14 | 89.48 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.52 | 3.46 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.30 | 3.27 | 3.22 | 3.37 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.30 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.48 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.54 | 2.50 | 6.26 |
| pass 3: turn | 0.61 | 0.59 | 0.55 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.25 | 2.53 | 2.51 | 6.24 |
| pass 5: turn | 0.56 | 0.56 | 0.52 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.17 | 2.53 | 2.49 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.43 | 1.99 | 1.31 | 2.71 |
| the draft's span (all passes) | 18.83 | 16.28 | 15.53 | 27.58 |
|   the draft's GPU time on its last card | 13.09 | 10.98 | 10.88 | 22.19 |
|   the leads (crossings) | 1.86 | 1.83 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.46 | 1.35 | 1.29 | 1.63 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.77 | 3.18 | 3.12 | 3.32 |

verify_gpu per card, median ms: 10.94, 10.64, 10.74, 10.82, 8.85, 10.28, 8.68, 10.02, 7.23

**e, long: 80 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.88 | 116.56 | 115.13 | 128.21 |
| verify_span | 99.08 | 98.90 | 97.92 | 99.75 |
|   verify_gpu (the nine cards) | 95.31 | 95.15 | 94.21 | 96.08 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.77 | 3.71 | 3.69 | 3.81 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.87 | 3.86 | 3.78 | 3.92 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.32 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.48 | 2.78 | 2.74 | 6.45 |
| pass 3: turn | 0.62 | 0.61 | 0.56 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.47 | 2.78 | 2.74 | 6.39 |
| pass 5: turn | 0.60 | 0.59 | 0.54 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.42 | 2.78 | 2.74 | 6.40 |
| pass 7: turn (the server before the next verify) | 2.08 | 1.91 | 1.26 | 2.41 |
| the draft's span (all passes) | 19.80 | 17.46 | 16.76 | 29.12 |
|   the draft's GPU time on its last card | 14.37 | 12.32 | 12.18 | 23.26 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.52 | 1.38 | 1.30 | 2.00 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.22 | 3.18 | 3.37 |

verify_gpu per card, median ms: 11.54, 11.72, 11.30, 11.92, 9.43, 10.86, 9.84, 10.70, 7.84

