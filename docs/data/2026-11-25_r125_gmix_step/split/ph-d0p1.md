ph: 1161 evaluations in 1 pieces (cut at gaps over 2000 ms): 1161 evaluations and 90 main passes
evaluations between main passes, by card: card 8: 350; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p1 | 90 | 1 | 88 | 2: 1, 3: 1, 4: 85, 5: 1 | 81 |

**ph, all requests: 81 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 100.20 | 97.89 | 95.85 | 108.49 |
| verify_span | 84.16 | 84.15 | 82.91 | 85.49 |
|   verify_gpu (the nine cards) | 80.58 | 80.55 | 79.33 | 81.92 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.54 | 3.66 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 1.00 | 0.98 | 0.91 | 1.06 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.53 | 2.59 | 2.56 | 6.42 |
| pass 1: turn | 0.96 | 0.95 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.27 | 2.55 | 2.50 | 6.38 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.19 | 2.54 | 2.49 | 6.35 |
| pass 3: turn (the server before the next verify) | 2.13 | 1.76 | 0.99 | 2.57 |
| the draft's span (all passes) | 16.04 | 13.51 | 12.46 | 24.76 |
|   the draft's GPU time on its last card | 11.05 | 8.72 | 8.61 | 20.25 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.73 | 2.97 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.39 | 3.07 | 3.01 | 3.20 |

verify_gpu per card, median ms: 10.47, 9.42, 9.34, 9.33, 7.98, 9.30, 7.63, 9.42, 7.68

**ph, d0p1: 81 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 100.20 | 97.89 | 95.85 | 108.49 |
| verify_span | 84.16 | 84.15 | 82.91 | 85.49 |
|   verify_gpu (the nine cards) | 80.58 | 80.55 | 79.33 | 81.92 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.54 | 3.66 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 1.00 | 0.98 | 0.91 | 1.06 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.53 | 2.59 | 2.56 | 6.42 |
| pass 1: turn | 0.96 | 0.95 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.27 | 2.55 | 2.50 | 6.38 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.19 | 2.54 | 2.49 | 6.35 |
| pass 3: turn (the server before the next verify) | 2.13 | 1.76 | 0.99 | 2.57 |
| the draft's span (all passes) | 16.04 | 13.51 | 12.46 | 24.76 |
|   the draft's GPU time on its last card | 11.05 | 8.72 | 8.61 | 20.25 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.73 | 2.97 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.39 | 3.07 | 3.01 | 3.20 |

verify_gpu per card, median ms: 10.47, 9.42, 9.34, 9.33, 7.98, 9.30, 7.63, 9.42, 7.68

