e744: 3699 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1827 evaluations and 89 main passes, 1860 evaluations and 91 main passes
evaluations between main passes, by card: card 4: 683, card 8: 1366; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 6: 1, 8: 83 | 80 |
| long | 91 | 6 | 84 | 8: 83, 16: 1 | 81 |

**e744, all requests: 161 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.61 | 115.21 | 106.84 | 119.83 |
| verify_span | 95.35 | 97.02 | 90.72 | 99.36 |
|   verify_gpu (the nine cards) | 91.72 | 93.30 | 87.28 | 95.66 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.68 | 3.44 | 3.74 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.59 | 3.77 | 3.24 | 3.90 |
| pass 1: turn | 0.14 | 0.07 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 3: gpu | 3.06 | 2.75 | 2.51 | 3.33 |
| pass 3: turn | 0.61 | 0.60 | 0.55 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.08 | 2.75 | 2.51 | 3.47 |
| pass 5: turn | 0.60 | 0.57 | 0.54 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.03 | 2.75 | 2.51 | 3.34 |
| pass 7: turn (the server before the next verify) | 2.08 | 2.00 | 1.30 | 2.67 |
| the draft's span (all passes) | 18.27 | 17.10 | 15.77 | 27.11 |
|   the draft's GPU time on its last card | 12.89 | 12.26 | 10.91 | 14.16 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.38 | 1.30 | 1.64 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.80 | 3.21 | 3.13 | 3.35 |

verify_gpu per card, median ms: 11.21, 11.40, 11.11, 11.55, 9.17, 10.67, 9.51, 10.34, 7.69

**e744, greedy: 80 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.26 | 108.13 | 106.50 | 119.83 |
| verify_span | 91.75 | 91.66 | 90.57 | 92.65 |
|   verify_gpu (the nine cards) | 88.23 | 88.14 | 87.10 | 89.16 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.52 | 3.46 | 3.44 | 3.83 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 1: gpu | 3.30 | 3.28 | 3.23 | 3.36 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.31 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.24 | 2.53 | 2.49 | 6.25 |
| pass 3: turn | 0.61 | 0.59 | 0.54 | 0.67 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.26 | 2.54 | 2.50 | 6.24 |
| pass 5: turn | 0.57 | 0.57 | 0.54 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.49 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.07 | 2.02 | 1.31 | 2.60 |
| the draft's span (all passes) | 18.51 | 16.27 | 15.59 | 27.57 |
|   the draft's GPU time on its last card | 13.13 | 10.99 | 10.89 | 22.23 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.48 | 1.37 | 1.30 | 1.83 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.79 | 3.17 | 3.12 | 3.34 |

verify_gpu per card, median ms: 10.93, 10.60, 10.76, 10.81, 8.79, 10.28, 8.66, 9.98, 7.21

**e744, long: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 116.92 | 116.26 | 114.72 | 119.83 |
| verify_span | 98.89 | 98.77 | 97.72 | 99.71 |
|   verify_gpu (the nine cards) | 95.17 | 95.08 | 94.01 | 96.02 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.73 | 3.70 | 3.68 | 3.74 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.88 | 3.86 | 3.80 | 3.91 |
| pass 1: turn | 0.11 | 0.07 | 0.07 | 0.08 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 3: gpu | 2.88 | 2.78 | 2.74 | 2.81 |
| pass 3: turn | 0.60 | 0.61 | 0.56 | 0.64 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 2.90 | 2.78 | 2.74 | 2.81 |
| pass 5: turn | 0.63 | 0.59 | 0.55 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 2.87 | 2.77 | 2.74 | 2.81 |
| pass 7: turn (the server before the next verify) | 2.08 | 1.98 | 1.24 | 2.69 |
| the draft's span (all passes) | 18.03 | 17.49 | 16.76 | 19.33 |
|   the draft's GPU time on its last card | 12.66 | 12.32 | 12.24 | 12.41 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.46 | 1.38 | 1.31 | 1.45 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.22 | 3.19 | 3.35 |

verify_gpu per card, median ms: 11.54, 11.65, 11.29, 11.90, 9.36, 10.85, 9.85, 10.62, 7.85

