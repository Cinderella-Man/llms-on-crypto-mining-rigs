a-2: 3744 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1902 evaluations and 93 main passes
evaluations between main passes, by card: card 4: 692, card 8: 1384; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 93 | 6 | 86 | 8: 85, 16: 1 | 83 |

**a-2, all requests: 164 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 116.86 | 117.82 | 109.04 | 125.01 |
| verify_span | 96.50 | 97.83 | 91.88 | 100.61 |
|   verify_gpu (the nine cards) | 91.84 | 93.09 | 87.38 | 95.84 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.66 | 4.70 | 4.48 | 4.80 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.60 | 3.73 | 3.27 | 3.92 |
| pass 1: turn | 0.44 | 0.35 | 0.34 | 0.38 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.33 | 2.78 | 2.50 | 6.48 |
| pass 3: turn | 0.60 | 0.59 | 0.55 | 0.64 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.42 | 2.74 | 2.50 | 6.41 |
| pass 5: turn | 0.57 | 0.56 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.24 | 2.74 | 2.50 | 6.26 |
| pass 7: turn (the server before the next verify) | 3.10 | 2.83 | 2.56 | 3.13 |
| the draft's span (all passes) | 20.36 | 18.62 | 17.04 | 29.92 |
|   the draft's GPU time on its last card | 13.73 | 12.26 | 10.94 | 23.36 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.71 | 1.62 | 1.55 | 1.75 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.19 | 3.11 | 3.30 |

verify_gpu per card, median ms: 11.24, 11.46, 11.03, 11.49, 9.19, 10.67, 9.50, 10.41, 7.69

**a-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 112.79 | 110.43 | 108.74 | 122.67 |
| verify_span | 92.92 | 92.74 | 91.60 | 94.22 |
|   verify_gpu (the nine cards) | 88.36 | 88.17 | 87.15 | 89.45 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.55 | 4.51 | 4.46 | 4.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.32 | 3.31 | 3.26 | 3.36 |
| pass 1: turn | 0.45 | 0.35 | 0.34 | 0.53 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.27 | 2.56 | 2.49 | 6.30 |
| pass 3: turn | 0.60 | 0.58 | 0.54 | 0.64 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.38 | 2.53 | 2.48 | 6.28 |
| pass 5: turn | 0.55 | 0.55 | 0.53 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.17 | 2.53 | 2.46 | 6.24 |
| pass 7: turn (the server before the next verify) | 3.05 | 2.72 | 2.45 | 2.87 |
| the draft's span (all passes) | 19.87 | 17.27 | 16.90 | 30.47 |
|   the draft's GPU time on its last card | 13.28 | 11.07 | 10.86 | 22.35 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.72 | 1.62 | 1.55 | 2.41 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.16 | 3.10 | 3.29 |

verify_gpu per card, median ms: 10.94, 10.62, 10.75, 10.81, 8.86, 10.28, 8.70, 10.02, 7.20

**a-2, long: 83 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 120.84 | 118.89 | 117.32 | 129.90 |
| verify_span | 99.99 | 99.88 | 98.66 | 100.82 |
|   verify_gpu (the nine cards) | 95.24 | 95.10 | 93.93 | 96.09 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.75 | 4.74 | 4.71 | 4.82 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 1: gpu | 3.87 | 3.86 | 3.78 | 3.93 |
| pass 1: turn | 0.43 | 0.35 | 0.35 | 0.37 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.40 | 2.81 | 2.77 | 6.50 |
| pass 3: turn | 0.59 | 0.59 | 0.55 | 0.63 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.46 | 2.77 | 2.73 | 6.44 |
| pass 5: turn | 0.58 | 0.57 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.30 | 2.77 | 2.73 | 6.41 |
| pass 7: turn (the server before the next verify) | 3.15 | 2.91 | 2.62 | 3.21 |
| the draft's span (all passes) | 20.85 | 18.73 | 18.20 | 29.91 |
|   the draft's GPU time on its last card | 14.17 | 12.33 | 12.19 | 23.40 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.84 |
|   the turns of the passes before the last | 1.70 | 1.62 | 1.55 | 1.72 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.77 | 3.20 | 3.16 | 3.30 |

verify_gpu per card, median ms: 11.53, 11.76, 11.26, 11.91, 9.42, 10.84, 9.80, 10.70, 7.86

