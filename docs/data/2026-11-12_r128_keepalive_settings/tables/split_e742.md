e742: 3681 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1839 evaluations and 90 main passes
evaluations between main passes, by card: card 4: 680, card 8: 1360; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 90 | 6 | 83 | 8: 82, 16: 1 | 80 |

**e742, all requests: 161 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.85 | 115.07 | 106.59 | 120.34 |
| verify_span | 95.37 | 96.85 | 90.72 | 99.75 |
|   verify_gpu (the nine cards) | 91.74 | 93.14 | 87.25 | 95.97 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.67 | 3.44 | 3.76 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.59 | 3.73 | 3.24 | 3.92 |
| pass 1: turn | 0.16 | 0.07 | 0.07 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.15 | 2.75 | 2.52 | 6.25 |
| pass 3: turn | 0.62 | 0.61 | 0.55 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.17 | 2.75 | 2.51 | 6.24 |
| pass 5: turn | 0.58 | 0.57 | 0.53 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.11 | 2.74 | 2.51 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.02 | 1.78 | 1.08 | 2.46 |
| the draft's span (all passes) | 18.48 | 17.00 | 15.49 | 27.42 |
|   the draft's GPU time on its last card | 13.17 | 12.22 | 10.93 | 22.19 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.39 | 1.30 | 1.66 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.89 | 3.20 | 3.12 | 3.33 |

verify_gpu per card, median ms: 11.18, 11.37, 11.08, 11.46, 9.16, 10.64, 9.36, 10.31, 7.77

**e742, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.18 | 107.89 | 106.32 | 119.36 |
| verify_span | 91.72 | 91.55 | 90.45 | 92.68 |
|   verify_gpu (the nine cards) | 88.21 | 88.05 | 87.03 | 89.20 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.52 | 3.46 | 3.43 | 3.84 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.30 | 3.28 | 3.22 | 3.37 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.53 | 2.51 | 6.28 |
| pass 3: turn | 0.62 | 0.61 | 0.54 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.26 | 2.54 | 2.51 | 6.25 |
| pass 5: turn | 0.56 | 0.56 | 0.53 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.50 | 6.26 |
| pass 7: turn (the server before the next verify) | 2.04 | 1.78 | 1.11 | 2.50 |
| the draft's span (all passes) | 18.46 | 16.12 | 15.37 | 27.47 |
|   the draft's GPU time on its last card | 13.11 | 11.02 | 10.90 | 22.20 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.48 | 1.38 | 1.29 | 1.69 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.15 | 3.10 | 3.29 |

verify_gpu per card, median ms: 10.93, 10.61, 10.76, 10.79, 8.79, 10.29, 8.66, 9.98, 7.20

**e742, long: 80 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.57 | 116.21 | 114.52 | 124.23 |
| verify_span | 99.07 | 98.78 | 97.46 | 100.09 |
|   verify_gpu (the nine cards) | 95.32 | 95.02 | 93.80 | 96.40 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.75 | 3.71 | 3.67 | 3.75 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.89 | 3.87 | 3.79 | 3.92 |
| pass 1: turn | 0.13 | 0.07 | 0.07 | 0.08 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 3: gpu | 3.07 | 2.76 | 2.74 | 3.32 |
| pass 3: turn | 0.62 | 0.61 | 0.56 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.08 | 2.76 | 2.74 | 3.42 |
| pass 5: turn | 0.59 | 0.59 | 0.55 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.47 |
| pass 7: gpu | 3.05 | 2.76 | 2.74 | 2.82 |
| pass 7: turn (the server before the next verify) | 1.99 | 1.77 | 1.04 | 2.19 |
| the draft's span (all passes) | 18.51 | 17.29 | 16.49 | 25.60 |
|   the draft's GPU time on its last card | 13.22 | 12.29 | 12.19 | 13.51 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.46 | 1.39 | 1.31 | 1.63 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 4.02 | 3.21 | 3.17 | 3.33 |

verify_gpu per card, median ms: 11.55, 11.62, 11.29, 11.91, 9.38, 10.87, 9.85, 10.63, 7.91

