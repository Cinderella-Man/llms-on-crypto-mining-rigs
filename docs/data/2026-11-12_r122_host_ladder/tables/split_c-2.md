c-2: 3660 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1818 evaluations and 89 main passes
evaluations between main passes, by card: card 4: 676, card 8: 1352; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 89 | 6 | 82 | 8: 81, 16: 1 | 79 |

**c-2, all requests: 160 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.57 | 115.92 | 107.69 | 120.36 |
| verify_span | 95.49 | 96.02 | 90.92 | 99.92 |
|   verify_gpu (the nine cards) | 91.75 | 92.24 | 87.32 | 96.09 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.77 | 3.53 | 3.93 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.59 | 3.71 | 3.24 | 3.91 |
| pass 1: turn | 0.16 | 0.08 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.16 | 2.75 | 2.52 | 6.26 |
| pass 3: turn | 0.63 | 0.61 | 0.55 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.17 | 2.74 | 2.52 | 6.24 |
| pass 5: turn | 0.58 | 0.57 | 0.53 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.11 | 2.74 | 2.52 | 6.23 |
| pass 7: turn (the server before the next verify) | 2.59 | 2.38 | 2.03 | 2.66 |
| the draft's span (all passes) | 19.08 | 17.75 | 16.49 | 28.02 |
|   the draft's GPU time on its last card | 13.17 | 12.23 | 10.93 | 22.17 |
|   the leads (crossings) | 1.84 | 1.83 | 1.82 | 1.86 |
|   the turns of the passes before the last | 1.47 | 1.40 | 1.30 | 1.65 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.78 | 3.21 | 3.13 | 3.34 |

verify_gpu per card, median ms: 11.19, 11.37, 10.99, 11.31, 9.15, 10.63, 9.35, 10.34, 7.73

**c-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 111.17 | 108.97 | 107.33 | 120.36 |
| verify_span | 91.98 | 91.83 | 90.66 | 93.06 |
|   verify_gpu (the nine cards) | 88.34 | 88.16 | 87.10 | 89.45 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.55 | 3.51 | 3.96 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.31 | 3.29 | 3.21 | 3.38 |
| pass 1: turn | 0.18 | 0.08 | 0.07 | 0.30 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 3: gpu | 3.23 | 2.53 | 2.50 | 6.26 |
| pass 3: turn | 0.63 | 0.60 | 0.54 | 0.67 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.48 |
| pass 5: gpu | 3.26 | 2.53 | 2.51 | 6.25 |
| pass 5: turn | 0.57 | 0.57 | 0.53 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.52 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.75 | 2.40 | 2.09 | 2.69 |
| the draft's span (all passes) | 19.20 | 16.70 | 16.27 | 28.12 |
|   the draft's GPU time on its last card | 13.12 | 11.01 | 10.89 | 22.21 |
|   the leads (crossings) | 1.84 | 1.83 | 1.82 | 1.88 |
|   the turns of the passes before the last | 1.49 | 1.39 | 1.31 | 1.76 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.07 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.18 | 3.12 | 3.35 |

verify_gpu per card, median ms: 10.96, 10.61, 10.74, 10.80, 8.86, 10.30, 8.68, 10.01, 7.18

**c-2, long: 79 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.05 | 117.18 | 115.33 | 120.07 |
| verify_span | 99.10 | 99.17 | 97.56 | 100.25 |
|   verify_gpu (the nine cards) | 95.24 | 95.32 | 93.78 | 96.43 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.86 | 3.80 | 3.77 | 3.88 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.88 | 3.86 | 3.78 | 3.93 |
| pass 1: turn | 0.13 | 0.08 | 0.08 | 0.09 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.09 | 2.78 | 2.74 | 3.31 |
| pass 3: turn | 0.63 | 0.62 | 0.56 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.09 | 2.77 | 2.74 | 3.38 |
| pass 5: turn | 0.58 | 0.59 | 0.53 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.05 | 2.76 | 2.74 | 2.81 |
| pass 7: turn (the server before the next verify) | 2.43 | 2.35 | 1.96 | 2.51 |
| the draft's span (all passes) | 18.95 | 17.89 | 17.32 | 19.33 |
|   the draft's GPU time on its last card | 13.23 | 12.32 | 12.16 | 13.47 |
|   the leads (crossings) | 1.83 | 1.83 | 1.82 | 1.85 |
|   the turns of the passes before the last | 1.46 | 1.40 | 1.30 | 1.48 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.22 | 3.18 | 3.34 |

verify_gpu per card, median ms: 11.58, 11.74, 11.32, 11.91, 9.46, 10.87, 9.79, 10.67, 7.91

