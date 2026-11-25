hn1: 1421 evaluations in 1 pieces (cut at gaps over 2000 ms): 1421 evaluations and 110 main passes
evaluations between main passes, by card: card 8: 430; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p1 | 110 | 1 | 108 | 1: 1, 4: 106, 5: 1 | 103 |

**hn1, all requests: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.53 | 104.18 | 101.86 | 114.83 |
| verify_span | 90.30 | 90.39 | 88.97 | 91.66 |
|   verify_gpu (the nine cards) | 86.71 | 86.83 | 85.42 | 88.06 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.59 | 3.57 | 3.54 | 3.62 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.06 |
| pass 0: turn | 1.00 | 1.00 | 0.90 | 1.11 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.59 | 2.56 | 5.77 |
| pass 1: turn | 0.98 | 0.97 | 0.91 | 1.07 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.96 | 2.53 | 2.49 | 5.53 |
| pass 2: turn | 0.91 | 0.91 | 0.86 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.51 | 2.48 | 3.24 |
| pass 3: turn (the server before the next verify) | 2.31 | 2.01 | 0.98 | 3.09 |
| the draft's span (all passes) | 15.23 | 13.69 | 12.41 | 24.56 |
|   the draft's GPU time on its last card | 10.03 | 8.67 | 8.58 | 15.59 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.89 | 2.87 | 2.74 | 3.03 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.38 | 3.08 | 3.04 | 3.26 |

verify_gpu per card, median ms: 10.90, 10.20, 10.40, 10.49, 8.62, 10.06, 8.58, 9.88, 7.67

**hn1, d0p1: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.53 | 104.18 | 101.86 | 114.83 |
| verify_span | 90.30 | 90.39 | 88.97 | 91.66 |
|   verify_gpu (the nine cards) | 86.71 | 86.83 | 85.42 | 88.06 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.59 | 3.57 | 3.54 | 3.62 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.06 |
| pass 0: turn | 1.00 | 1.00 | 0.90 | 1.11 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.59 | 2.56 | 5.77 |
| pass 1: turn | 0.98 | 0.97 | 0.91 | 1.07 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.96 | 2.53 | 2.49 | 5.53 |
| pass 2: turn | 0.91 | 0.91 | 0.86 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.51 | 2.48 | 3.24 |
| pass 3: turn (the server before the next verify) | 2.31 | 2.01 | 0.98 | 3.09 |
| the draft's span (all passes) | 15.23 | 13.69 | 12.41 | 24.56 |
|   the draft's GPU time on its last card | 10.03 | 8.67 | 8.58 | 15.59 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.89 | 2.87 | 2.74 | 3.03 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.38 | 3.08 | 3.04 | 3.26 |

verify_gpu per card, median ms: 10.90, 10.20, 10.40, 10.49, 8.62, 10.06, 8.58, 9.88, 7.67

