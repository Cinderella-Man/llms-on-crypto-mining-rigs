hn1: 1398 evaluations in 1 pieces (cut at gaps over 2000 ms): 1398 evaluations and 108 main passes
evaluations between main passes, by card: card 8: 425; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p2 | 108 | 0 | 107 | 1: 1, 4: 106 | 103 |

**hn1, all requests: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.97 | 102.72 | 101.41 | 104.22 |
| verify_span | 90.21 | 90.18 | 88.86 | 91.58 |
|   verify_gpu (the nine cards) | 86.64 | 86.63 | 85.32 | 88.02 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.57 | 3.57 | 3.54 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.06 |
| pass 0: turn | 1.05 | 0.99 | 0.90 | 1.11 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.61 | 2.59 | 2.55 | 2.61 |
| pass 1: turn | 0.92 | 0.91 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.51 | 2.51 | 2.48 | 2.54 |
| pass 2: turn | 0.88 | 0.87 | 0.84 | 0.92 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.53 | 2.52 | 2.48 | 2.57 |
| pass 3: turn (the server before the next verify) | 1.22 | 0.97 | 0.95 | 1.55 |
| the draft's span (all passes) | 12.76 | 12.42 | 12.29 | 13.04 |
|   the draft's GPU time on its last card | 8.69 | 8.65 | 8.57 | 8.75 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.84 | 2.77 | 2.67 | 2.93 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.17 | 3.06 | 3.31 |

verify_gpu per card, median ms: 10.85, 10.16, 10.39, 10.50, 8.59, 10.03, 8.53, 9.89, 7.71

**hn1, d0p2: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.97 | 102.72 | 101.41 | 104.22 |
| verify_span | 90.21 | 90.18 | 88.86 | 91.58 |
|   verify_gpu (the nine cards) | 86.64 | 86.63 | 85.32 | 88.02 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.57 | 3.57 | 3.54 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.06 |
| pass 0: turn | 1.05 | 0.99 | 0.90 | 1.11 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.61 | 2.59 | 2.55 | 2.61 |
| pass 1: turn | 0.92 | 0.91 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.51 | 2.51 | 2.48 | 2.54 |
| pass 2: turn | 0.88 | 0.87 | 0.84 | 0.92 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.53 | 2.52 | 2.48 | 2.57 |
| pass 3: turn (the server before the next verify) | 1.22 | 0.97 | 0.95 | 1.55 |
| the draft's span (all passes) | 12.76 | 12.42 | 12.29 | 13.04 |
|   the draft's GPU time on its last card | 8.69 | 8.65 | 8.57 | 8.75 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.84 | 2.77 | 2.67 | 2.93 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.17 | 3.06 | 3.31 |

verify_gpu per card, median ms: 10.85, 10.16, 10.39, 10.50, 8.59, 10.03, 8.53, 9.89, 7.71

