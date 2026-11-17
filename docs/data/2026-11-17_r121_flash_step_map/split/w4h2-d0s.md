w4h2: 1323 evaluations in 1 pieces (cut at gaps over 2000 ms): 1323 evaluations and 102 main passes
evaluations between main passes, by card: card 8: 404; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 102 | 0 | 101 | 4: 101 | 96 |

**w4h2, all requests: 96 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 103.27 | 101.22 | 99.44 | 111.98 |
| verify_span | 87.77 | 87.83 | 86.56 | 88.94 |
|   verify_gpu (the nine cards) | 84.13 | 84.13 | 82.95 | 85.28 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.62 | 3.60 | 3.66 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.00 | 1.00 | 0.99 | 1.01 |
| pass 0: turn | 1.06 | 1.05 | 0.97 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.51 | 2.48 | 6.18 |
| pass 1: turn | 0.97 | 0.96 | 0.92 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.96 | 2.43 | 2.40 | 6.17 |
| pass 2: turn | 0.91 | 0.90 | 0.88 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.43 | 2.39 | 6.15 |
| pass 3: turn (the server before the next verify) | 2.56 | 1.81 | 1.04 | 3.67 |
| the draft's span (all passes) | 15.50 | 13.23 | 12.40 | 24.32 |
|   the draft's GPU time on its last card | 10.00 | 8.37 | 8.30 | 19.53 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.93 | 2.93 | 2.82 | 3.05 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.45 | 3.14 | 3.10 | 3.27 |

verify_gpu per card, median ms: 10.63, 9.96, 10.03, 10.23, 8.31, 9.79, 8.29, 9.58, 7.37

**w4h2, d0s: 96 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 103.27 | 101.22 | 99.44 | 111.98 |
| verify_span | 87.77 | 87.83 | 86.56 | 88.94 |
|   verify_gpu (the nine cards) | 84.13 | 84.13 | 82.95 | 85.28 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.64 | 3.62 | 3.60 | 3.66 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.00 | 1.00 | 0.99 | 1.01 |
| pass 0: turn | 1.06 | 1.05 | 0.97 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.51 | 2.48 | 6.18 |
| pass 1: turn | 0.97 | 0.96 | 0.92 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.96 | 2.43 | 2.40 | 6.17 |
| pass 2: turn | 0.91 | 0.90 | 0.88 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.43 | 2.39 | 6.15 |
| pass 3: turn (the server before the next verify) | 2.56 | 1.81 | 1.04 | 3.67 |
| the draft's span (all passes) | 15.50 | 13.23 | 12.40 | 24.32 |
|   the draft's GPU time on its last card | 10.00 | 8.37 | 8.30 | 19.53 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.93 | 2.93 | 2.82 | 3.05 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.45 | 3.14 | 3.10 | 3.27 |

verify_gpu per card, median ms: 10.63, 9.96, 10.03, 10.23, 8.31, 9.79, 8.29, 9.58, 7.37

