d: 3723 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1881 evaluations and 92 main passes
evaluations between main passes, by card: card 4: 688, card 8: 1376; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 92 | 6 | 85 | 8: 84, 16: 1 | 82 |

**d, all requests: 163 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.72 | 115.95 | 107.16 | 122.10 |
| verify_span | 95.75 | 97.40 | 91.07 | 99.91 |
|   verify_gpu (the nine cards) | 92.00 | 93.64 | 87.40 | 96.03 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.74 | 3.77 | 3.53 | 3.90 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.59 | 3.76 | 3.24 | 3.92 |
| pass 1: turn | 0.17 | 0.08 | 0.07 | 0.30 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.20 | 2.75 | 2.51 | 6.25 |
| pass 3: turn | 0.62 | 0.60 | 0.55 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.22 | 2.74 | 2.51 | 6.24 |
| pass 5: turn | 0.58 | 0.57 | 0.54 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.16 | 2.74 | 2.51 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.37 | 1.98 | 1.32 | 2.72 |
| the draft's span (all passes) | 18.97 | 17.23 | 15.77 | 27.86 |
|   the draft's GPU time on its last card | 13.30 | 12.24 | 10.92 | 22.23 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.48 | 1.38 | 1.30 | 1.64 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.78 | 3.21 | 3.12 | 3.39 |

verify_gpu per card, median ms: 11.27, 11.43, 11.08, 11.58, 9.23, 10.67, 9.62, 10.43, 7.70

**d, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.70 | 108.48 | 107.00 | 119.72 |
| verify_span | 92.04 | 91.91 | 90.80 | 93.11 |
|   verify_gpu (the nine cards) | 88.42 | 88.23 | 87.26 | 89.40 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.56 | 3.52 | 3.92 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.30 | 3.28 | 3.22 | 3.36 |
| pass 1: turn | 0.18 | 0.08 | 0.07 | 0.30 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.53 | 2.49 | 6.25 |
| pass 3: turn | 0.62 | 0.60 | 0.54 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.03 | 0.03 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.25 | 2.53 | 2.49 | 6.24 |
| pass 5: turn | 0.57 | 0.56 | 0.53 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.50 | 6.25 |
| pass 7: turn (the server before the next verify) | 2.26 | 1.98 | 1.40 | 2.57 |
| the draft's span (all passes) | 18.67 | 16.30 | 15.56 | 27.72 |
|   the draft's GPU time on its last card | 13.10 | 11.00 | 10.90 | 22.23 |
|   the leads (crossings) | 1.83 | 1.82 | 1.81 | 1.84 |
|   the turns of the passes before the last | 1.48 | 1.37 | 1.30 | 1.88 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.18 | 3.11 | 3.35 |

verify_gpu per card, median ms: 10.97, 10.65, 10.75, 10.81, 8.86, 10.29, 8.69, 9.99, 7.24

**d, long: 82 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.69 | 116.83 | 115.51 | 127.49 |
| verify_span | 99.41 | 99.22 | 98.14 | 100.26 |
|   verify_gpu (the nine cards) | 95.55 | 95.37 | 94.30 | 96.31 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.86 | 3.80 | 3.76 | 3.89 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.88 | 3.86 | 3.80 | 3.94 |
| pass 1: turn | 0.16 | 0.08 | 0.07 | 0.29 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.17 | 2.78 | 2.74 | 3.34 |
| pass 3: turn | 0.61 | 0.60 | 0.55 | 0.64 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.18 | 2.76 | 2.73 | 3.45 |
| pass 5: turn | 0.58 | 0.59 | 0.55 | 0.61 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 7: gpu | 3.14 | 2.77 | 2.74 | 3.49 |
| pass 7: turn (the server before the next verify) | 2.47 | 1.98 | 1.26 | 2.72 |
| the draft's span (all passes) | 19.28 | 17.56 | 16.66 | 28.63 |
|   the draft's GPU time on its last card | 13.51 | 12.30 | 12.20 | 14.11 |
|   the leads (crossings) | 1.83 | 1.83 | 1.81 | 1.85 |
|   the turns of the passes before the last | 1.47 | 1.39 | 1.29 | 1.61 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.22 | 3.19 | 3.39 |

verify_gpu per card, median ms: 11.56, 11.72, 11.29, 11.92, 9.44, 10.89, 9.85, 10.77, 7.93

