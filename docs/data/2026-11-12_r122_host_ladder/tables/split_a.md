a: 3849 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 2007 evaluations and 98 main passes
evaluations between main passes, by card: card 4: 712, card 8: 1424; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 98 | 6 | 91 | 8: 90, 16: 1 | 88 |

**a, all requests: 169 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.04 | 118.30 | 109.32 | 123.83 |
| verify_span | 96.74 | 98.66 | 92.03 | 100.72 |
|   verify_gpu (the nine cards) | 92.06 | 93.90 | 87.52 | 95.98 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.68 | 4.71 | 4.51 | 4.82 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.45 | 0.45 | 0.47 |
| pass 1: gpu | 3.63 | 3.80 | 3.27 | 3.94 |
| pass 1: turn | 0.42 | 0.32 | 0.31 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.28 | 2.78 | 2.50 | 6.31 |
| pass 3: turn | 0.60 | 0.59 | 0.55 | 0.63 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.36 | 2.74 | 2.50 | 6.32 |
| pass 5: turn | 0.57 | 0.56 | 0.53 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.18 | 2.74 | 2.50 | 6.25 |
| pass 7: turn (the server before the next verify) | 3.21 | 2.84 | 2.57 | 3.16 |
| the draft's span (all passes) | 20.30 | 18.69 | 16.97 | 29.82 |
|   the draft's GPU time on its last card | 13.58 | 12.33 | 10.95 | 22.32 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.70 | 1.58 | 1.50 | 1.96 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.20 | 3.13 | 3.33 |

verify_gpu per card, median ms: 11.35, 11.52, 11.11, 11.70, 9.27, 10.71, 9.57, 10.42, 7.80

**a, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.09 | 110.57 | 108.87 | 122.55 |
| verify_span | 93.05 | 92.87 | 91.83 | 94.21 |
|   verify_gpu (the nine cards) | 88.47 | 88.32 | 87.25 | 89.63 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.58 | 4.54 | 4.49 | 4.62 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.32 | 3.30 | 3.25 | 3.36 |
| pass 1: turn | 0.41 | 0.32 | 0.31 | 0.75 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.27 | 2.56 | 2.49 | 6.30 |
| pass 3: turn | 0.62 | 0.56 | 0.53 | 0.63 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.38 | 2.53 | 2.45 | 6.27 |
| pass 5: turn | 0.56 | 0.54 | 0.53 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.17 | 2.53 | 2.46 | 6.23 |
| pass 7: turn (the server before the next verify) | 3.26 | 2.72 | 2.44 | 3.06 |
| the draft's span (all passes) | 20.04 | 17.18 | 16.83 | 30.51 |
|   the draft's GPU time on its last card | 13.27 | 11.06 | 10.88 | 22.28 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.70 | 1.56 | 1.49 | 2.38 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.07 |
| the host's launch of the main pass (inside verify_span) | 3.76 | 3.20 | 3.11 | 3.34 |

verify_gpu per card, median ms: 11.04, 10.64, 10.74, 10.81, 8.86, 10.29, 8.69, 10.00, 7.22

**a, long: 88 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 120.68 | 118.95 | 117.42 | 129.63 |
| verify_span | 100.13 | 100.04 | 98.99 | 101.19 |
|   verify_gpu (the nine cards) | 95.36 | 95.29 | 94.25 | 96.39 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.77 | 4.76 | 4.72 | 4.83 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.45 | 0.45 | 0.47 |
| pass 1: gpu | 3.91 | 3.90 | 3.82 | 3.95 |
| pass 1: turn | 0.42 | 0.32 | 0.32 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.29 | 2.83 | 2.77 | 6.43 |
| pass 3: turn | 0.59 | 0.59 | 0.56 | 0.62 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.34 | 2.78 | 2.73 | 6.41 |
| pass 5: turn | 0.58 | 0.56 | 0.55 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 3.19 | 2.78 | 2.73 | 6.39 |
| pass 7: turn (the server before the next verify) | 3.16 | 2.96 | 2.68 | 3.16 |
| the draft's span (all passes) | 20.54 | 18.80 | 18.45 | 29.79 |
|   the draft's GPU time on its last card | 13.86 | 12.42 | 12.24 | 23.25 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.82 |
|   the turns of the passes before the last | 1.70 | 1.59 | 1.55 | 1.69 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.74 | 3.20 | 3.17 | 3.31 |

verify_gpu per card, median ms: 11.56, 11.76, 11.29, 11.95, 9.44, 10.86, 9.84, 10.66, 7.90

