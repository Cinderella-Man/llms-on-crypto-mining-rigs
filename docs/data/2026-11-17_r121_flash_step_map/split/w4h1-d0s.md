w4h1: 1323 evaluations in 1 pieces (cut at gaps over 2000 ms): 1323 evaluations and 102 main passes
evaluations between main passes, by card: card 8: 404; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 102 | 0 | 101 | 4: 101 | 96 |

**w4h1, all requests: 96 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.08 | 100.35 | 98.95 | 111.47 |
| verify_span | 87.78 | 87.82 | 86.57 | 88.89 |
|   verify_gpu (the nine cards) | 84.16 | 84.17 | 82.97 | 85.25 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.61 | 3.59 | 3.64 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.99 | 0.99 | 0.98 | 1.01 |
| pass 0: turn | 1.06 | 1.05 | 0.96 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.14 | 2.50 | 2.47 | 6.18 |
| pass 1: turn | 0.91 | 0.91 | 0.88 | 0.95 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.95 | 2.42 | 2.39 | 6.11 |
| pass 2: turn | 0.87 | 0.86 | 0.84 | 0.91 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.90 | 2.44 | 2.39 | 6.12 |
| pass 3: turn (the server before the next verify) | 1.49 | 1.14 | 1.04 | 1.18 |
| the draft's span (all passes) | 14.30 | 12.30 | 12.17 | 23.43 |
|   the draft's GPU time on its last card | 9.97 | 8.35 | 8.27 | 19.43 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.84 | 2.83 | 2.74 | 2.95 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.49 | 3.18 | 3.08 | 3.36 |

verify_gpu per card, median ms: 10.59, 9.97, 10.03, 10.24, 8.31, 9.79, 8.29, 9.60, 7.38

**w4h1, d0s: 96 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.08 | 100.35 | 98.95 | 111.47 |
| verify_span | 87.78 | 87.82 | 86.57 | 88.89 |
|   verify_gpu (the nine cards) | 84.16 | 84.17 | 82.97 | 85.25 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.61 | 3.59 | 3.64 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.99 | 0.99 | 0.98 | 1.01 |
| pass 0: turn | 1.06 | 1.05 | 0.96 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.14 | 2.50 | 2.47 | 6.18 |
| pass 1: turn | 0.91 | 0.91 | 0.88 | 0.95 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.95 | 2.42 | 2.39 | 6.11 |
| pass 2: turn | 0.87 | 0.86 | 0.84 | 0.91 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.90 | 2.44 | 2.39 | 6.12 |
| pass 3: turn (the server before the next verify) | 1.49 | 1.14 | 1.04 | 1.18 |
| the draft's span (all passes) | 14.30 | 12.30 | 12.17 | 23.43 |
|   the draft's GPU time on its last card | 9.97 | 8.35 | 8.27 | 19.43 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.84 | 2.83 | 2.74 | 2.95 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.49 | 3.18 | 3.08 | 3.36 |

verify_gpu per card, median ms: 10.59, 9.97, 10.03, 10.24, 8.31, 9.79, 8.29, 9.60, 7.38

