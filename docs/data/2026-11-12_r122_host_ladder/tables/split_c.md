c: 3723 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1881 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 688, card 8: 1376; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 92 | 6 | 85 | 8: 84, 16: 1 | 82 |

**c, all requests: 163 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.78 | 116.25 | 107.58 | 121.31 |
| verify_span | 95.72 | 97.60 | 91.00 | 99.92 |
|   verify_gpu (the nine cards) | 91.94 | 93.78 | 87.44 | 96.06 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.77 | 3.77 | 3.52 | 3.96 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.59 | 3.75 | 3.23 | 3.92 |
| pass 1: turn | 0.18 | 0.08 | 0.07 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 3 (draft 3): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.15 | 2.75 | 2.51 | 6.25 |
| pass 3: turn | 0.62 | 0.60 | 0.55 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.45 | 0.45 | 0.47 |
| pass 5: gpu | 3.17 | 2.74 | 2.51 | 6.23 |
| pass 5: turn | 0.57 | 0.57 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.11 | 2.74 | 2.51 | 6.22 |
| pass 7: turn (the server before the next verify) | 2.58 | 2.34 | 2.06 | 2.65 |
| the draft's span (all passes) | 19.06 | 17.75 | 16.35 | 28.06 |
|   the draft's GPU time on its last card | 13.16 | 12.23 | 10.91 | 22.19 |
|   the leads (crossings) | 1.82 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.49 | 1.38 | 1.30 | 1.73 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.17 | 3.11 | 3.35 |

verify_gpu per card, median ms: 11.28, 11.48, 11.07, 11.55, 9.28, 10.69, 9.54, 10.40, 7.79

**c, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 111.19 | 108.72 | 107.30 | 120.32 |
| verify_span | 92.12 | 91.94 | 90.75 | 93.14 |
|   verify_gpu (the nine cards) | 88.42 | 88.26 | 87.20 | 89.55 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.70 | 3.56 | 3.52 | 3.98 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.30 | 3.28 | 3.21 | 3.37 |
| pass 1: turn | 0.19 | 0.08 | 0.07 | 0.35 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.53 | 2.50 | 6.25 |
| pass 3: turn | 0.61 | 0.59 | 0.55 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.25 | 2.53 | 2.50 | 6.25 |
| pass 5: turn | 0.56 | 0.56 | 0.53 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.50 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.66 | 2.33 | 2.09 | 2.64 |
| the draft's span (all passes) | 19.07 | 16.56 | 16.25 | 28.22 |
|   the draft's GPU time on its last card | 13.10 | 10.98 | 10.89 | 22.20 |
|   the leads (crossings) | 1.82 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.49 | 1.35 | 1.30 | 1.93 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.72 | 3.15 | 3.09 | 3.26 |

verify_gpu per card, median ms: 11.00, 10.65, 10.75, 10.82, 8.85, 10.28, 8.68, 10.01, 7.19

**c, long: 82 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.32 | 117.05 | 115.81 | 124.60 |
| verify_span | 99.27 | 99.00 | 98.07 | 100.06 |
|   verify_gpu (the nine cards) | 95.42 | 95.12 | 94.14 | 96.16 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.85 | 3.80 | 3.77 | 3.87 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.88 | 3.87 | 3.78 | 3.94 |
| pass 1: turn | 0.17 | 0.08 | 0.08 | 0.09 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 3 (draft 3): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.08 | 2.77 | 2.74 | 3.36 |
| pass 3: turn | 0.62 | 0.61 | 0.55 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.47 |
| pass 5: gpu | 3.09 | 2.76 | 2.74 | 3.44 |
| pass 5: turn | 0.59 | 0.58 | 0.55 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.04 | 2.75 | 2.73 | 2.81 |
| pass 7: turn (the server before the next verify) | 2.51 | 2.35 | 2.02 | 2.74 |
| the draft's span (all passes) | 19.05 | 17.83 | 17.53 | 21.66 |
|   the draft's GPU time on its last card | 13.23 | 12.28 | 12.21 | 13.58 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.50 | 1.39 | 1.31 | 1.68 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.78 | 3.19 | 3.16 | 3.36 |

verify_gpu per card, median ms: 11.53, 11.74, 11.30, 11.92, 9.43, 10.89, 9.81, 10.65, 7.92

