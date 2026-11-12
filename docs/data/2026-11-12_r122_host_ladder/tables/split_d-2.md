d-2: 3693 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1851 evaluations and 91 main passes
evaluations between main passes, by card: card 4: 684, card 8: 1368; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 91 | 6 | 84 | 8: 83, 16: 1 | 81 |

**d-2, all requests: 162 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.58 | 115.92 | 107.24 | 120.86 |
| verify_span | 95.65 | 96.92 | 91.04 | 100.01 |
|   verify_gpu (the nine cards) | 91.89 | 92.74 | 87.38 | 96.19 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.76 | 3.76 | 3.53 | 3.92 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.60 | 3.72 | 3.24 | 3.93 |
| pass 1: turn | 0.17 | 0.08 | 0.07 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.13 | 2.75 | 2.51 | 6.25 |
| pass 3: turn | 0.63 | 0.60 | 0.56 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.04 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.15 | 2.74 | 2.51 | 6.22 |
| pass 5: turn | 0.59 | 0.57 | 0.54 | 0.62 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.08 | 2.74 | 2.51 | 6.23 |
| pass 7: turn (the server before the next verify) | 2.49 | 2.00 | 1.38 | 2.75 |
| the draft's span (all passes) | 18.94 | 17.22 | 15.81 | 27.76 |
|   the draft's GPU time on its last card | 13.10 | 12.24 | 10.93 | 22.15 |
|   the leads (crossings) | 1.83 | 1.83 | 1.82 | 1.85 |
|   the turns of the passes before the last | 1.51 | 1.39 | 1.31 | 1.69 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 4.03 | 3.21 | 3.12 | 3.37 |

verify_gpu per card, median ms: 11.25, 11.34, 11.02, 11.42, 9.18, 10.65, 9.46, 10.37, 7.74

**d-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.94 | 108.54 | 106.79 | 119.91 |
| verify_span | 92.00 | 91.84 | 90.67 | 93.02 |
|   verify_gpu (the nine cards) | 88.37 | 88.21 | 87.09 | 89.44 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.63 | 3.56 | 3.51 | 3.95 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.31 | 3.29 | 3.22 | 3.37 |
| pass 1: turn | 0.18 | 0.08 | 0.07 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 3: gpu | 3.23 | 2.53 | 2.50 | 6.27 |
| pass 3: turn | 0.66 | 0.60 | 0.54 | 0.66 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 4: turn | 0.05 | 0.02 | 0.02 | 0.04 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.26 | 2.53 | 2.51 | 6.25 |
| pass 5: turn | 0.59 | 0.57 | 0.53 | 0.60 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.46 |
| pass 7: gpu | 3.18 | 2.53 | 2.50 | 6.24 |
| pass 7: turn (the server before the next verify) | 2.43 | 2.04 | 1.41 | 2.68 |
| the draft's span (all passes) | 18.94 | 16.46 | 15.63 | 27.76 |
|   the draft's GPU time on its last card | 13.11 | 11.01 | 10.90 | 22.23 |
|   the leads (crossings) | 1.83 | 1.83 | 1.82 | 1.84 |
|   the turns of the passes before the last | 1.57 | 1.38 | 1.31 | 1.98 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.77 | 3.16 | 3.10 | 3.30 |

verify_gpu per card, median ms: 10.95, 10.62, 10.74, 10.81, 8.85, 10.29, 8.71, 10.02, 7.19

**d-2, long: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.23 | 116.75 | 115.49 | 124.36 |
| verify_span | 99.29 | 99.10 | 97.74 | 100.36 |
|   verify_gpu (the nine cards) | 95.40 | 95.24 | 93.95 | 96.50 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.89 | 3.80 | 3.76 | 3.89 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.03 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.46 | 0.47 |
| pass 1: gpu | 3.89 | 3.88 | 3.79 | 3.95 |
| pass 1: turn | 0.15 | 0.08 | 0.07 | 0.08 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 3 (draft 3): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 3: gpu | 3.03 | 2.77 | 2.74 | 2.83 |
| pass 3: turn | 0.61 | 0.61 | 0.56 | 0.65 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.03 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 5: gpu | 3.04 | 2.77 | 2.74 | 2.83 |
| pass 5: turn | 0.59 | 0.59 | 0.55 | 0.62 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 7: gpu | 2.99 | 2.76 | 2.73 | 2.81 |
| pass 7: turn (the server before the next verify) | 2.56 | 1.93 | 1.35 | 2.80 |
| the draft's span (all passes) | 18.94 | 17.56 | 16.81 | 19.54 |
|   the draft's GPU time on its last card | 13.09 | 12.30 | 12.20 | 13.48 |
|   the leads (crossings) | 1.83 | 1.83 | 1.82 | 1.85 |
|   the turns of the passes before the last | 1.46 | 1.40 | 1.31 | 1.48 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 4.29 | 3.22 | 3.18 | 3.40 |

verify_gpu per card, median ms: 11.53, 11.71, 11.31, 11.92, 9.45, 10.86, 9.83, 10.72, 7.92

