e-2: 3798 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1956 evaluations and 96 main passes
evaluations between main passes, by card: card 4: 704, card 8: 1408; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 96 | 6 | 89 | 8: 88, 16: 1 | 86 |

**e-2, all requests: 167 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.72 | 116.05 | 107.05 | 122.15 |
| verify_span | 95.69 | 97.41 | 90.84 | 99.80 |
|   verify_gpu (the nine cards) | 92.03 | 93.57 | 87.37 | 96.09 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.67 | 3.68 | 3.44 | 3.79 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.60 | 3.77 | 3.24 | 3.89 |
| pass 1: turn | 0.17 | 0.07 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.75 | 2.51 | 6.25 |
| pass 3: turn | 0.61 | 0.61 | 0.55 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.25 | 2.75 | 2.51 | 6.25 |
| pass 5: turn | 0.58 | 0.57 | 0.54 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.19 | 2.74 | 2.51 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.32 | 2.03 | 1.30 | 2.72 |
| the draft's span (all passes) | 19.03 | 17.44 | 15.76 | 27.88 |
|   the draft's GPU time on its last card | 13.40 | 12.23 | 10.91 | 22.21 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.38 | 1.29 | 1.69 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.21 | 3.13 | 3.35 |

verify_gpu per card, median ms: 11.27, 11.46, 11.11, 11.67, 9.29, 10.72, 9.48, 10.45, 7.82

**e-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.81 | 108.51 | 106.55 | 120.04 |
| verify_span | 91.94 | 91.67 | 90.58 | 93.04 |
|   verify_gpu (the nine cards) | 88.35 | 88.15 | 87.12 | 89.44 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.59 | 3.46 | 3.43 | 3.85 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.30 | 3.28 | 3.22 | 3.37 |
| pass 1: turn | 0.18 | 0.07 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.53 | 2.50 | 6.25 |
| pass 3: turn | 0.62 | 0.60 | 0.54 | 0.67 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.25 | 2.53 | 2.50 | 6.23 |
| pass 5: turn | 0.56 | 0.56 | 0.54 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 7: gpu | 3.17 | 2.53 | 2.50 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.48 | 2.03 | 1.30 | 3.04 |
| the draft's span (all passes) | 18.88 | 16.36 | 15.56 | 27.43 |
|   the draft's GPU time on its last card | 13.09 | 10.97 | 10.88 | 22.20 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.36 | 1.28 | 1.70 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.18 | 3.12 | 3.36 |

verify_gpu per card, median ms: 10.94, 10.62, 10.75, 10.81, 8.87, 10.30, 8.67, 10.03, 7.21

**e-2, long: 86 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.40 | 116.76 | 115.29 | 127.77 |
| verify_span | 99.23 | 99.02 | 97.89 | 100.18 |
|   verify_gpu (the nine cards) | 95.49 | 95.28 | 94.20 | 96.43 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.70 | 3.67 | 3.77 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 1: gpu | 3.87 | 3.85 | 3.79 | 3.91 |
| pass 1: turn | 0.16 | 0.07 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.24 | 2.78 | 2.74 | 6.45 |
| pass 3: turn | 0.61 | 0.61 | 0.56 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.26 | 2.77 | 2.74 | 6.42 |
| pass 5: turn | 0.59 | 0.59 | 0.55 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.20 | 2.75 | 2.73 | 6.42 |
| pass 7: turn (the server before the next verify) | 2.17 | 2.02 | 1.30 | 2.36 |
| the draft's span (all passes) | 19.17 | 17.56 | 16.74 | 28.47 |
|   the draft's GPU time on its last card | 13.70 | 12.29 | 12.17 | 23.36 |
|   the leads (crossings) | 1.83 | 1.83 | 1.82 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.39 | 1.30 | 1.65 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.21 | 3.18 | 3.35 |

verify_gpu per card, median ms: 11.53, 11.76, 11.29, 11.92, 9.44, 10.89, 9.82, 10.72, 7.93

