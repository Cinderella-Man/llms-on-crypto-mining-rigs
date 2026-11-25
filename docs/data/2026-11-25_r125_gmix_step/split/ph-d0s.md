ph: 1102 evaluations in 1 pieces (cut at gaps over 2000 ms): 1102 evaluations and 85 main passes
evaluations between main passes, by card: card 8: 336; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 85 | 0 | 84 | 4: 84 | 80 |

**ph, all requests: 80 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 99.17 | 97.75 | 95.62 | 107.03 |
| verify_span | 84.15 | 84.28 | 82.34 | 85.58 |
|   verify_gpu (the nine cards) | 80.55 | 80.66 | 78.77 | 81.97 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.59 | 3.56 | 3.64 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 1.00 | 0.99 | 0.93 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.13 | 2.60 | 2.57 | 5.64 |
| pass 1: turn | 0.95 | 0.94 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.97 | 2.55 | 2.51 | 3.18 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.97 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.96 | 2.55 | 2.52 | 5.38 |
| pass 3: turn (the server before the next verify) | 2.04 | 1.46 | 0.95 | 2.06 |
| the draft's span (all passes) | 15.02 | 13.37 | 12.46 | 23.89 |
|   the draft's GPU time on its last card | 10.11 | 8.75 | 8.67 | 12.40 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.74 | 2.99 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.46 | 3.13 | 3.08 | 3.27 |

verify_gpu per card, median ms: 10.52, 9.39, 9.32, 9.29, 7.94, 9.28, 7.66, 9.47, 7.71

**ph, d0s: 80 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 99.17 | 97.75 | 95.62 | 107.03 |
| verify_span | 84.15 | 84.28 | 82.34 | 85.58 |
|   verify_gpu (the nine cards) | 80.55 | 80.66 | 78.77 | 81.97 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.59 | 3.56 | 3.64 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 1.00 | 0.99 | 0.93 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.13 | 2.60 | 2.57 | 5.64 |
| pass 1: turn | 0.95 | 0.94 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.97 | 2.55 | 2.51 | 3.18 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.97 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.96 | 2.55 | 2.52 | 5.38 |
| pass 3: turn (the server before the next verify) | 2.04 | 1.46 | 0.95 | 2.06 |
| the draft's span (all passes) | 15.02 | 13.37 | 12.46 | 23.89 |
|   the draft's GPU time on its last card | 10.11 | 8.75 | 8.67 | 12.40 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.74 | 2.99 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.46 | 3.13 | 3.08 | 3.27 |

verify_gpu per card, median ms: 10.52, 9.39, 9.32, 9.29, 7.94, 9.28, 7.66, 9.47, 7.71

