w5h2: 1130 evaluations in 1 pieces (cut at gaps over 2000 ms): 1130 evaluations and 81 main passes
evaluations between main passes, by card: card 8: 400; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 81 | 0 | 80 | 5: 80 | 76 |

**w5h2, all requests: 76 typical steps, 5 draft passes a step, 5 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 144.45 | 139.90 | 137.68 | 154.16 |
| verify_span | 122.47 | 122.46 | 120.99 | 124.13 |
|   verify_gpu (the nine cards) | 118.13 | 118.14 | 116.65 | 119.83 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.34 | 4.32 | 4.30 | 4.47 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.19 | 1.19 | 1.17 | 1.21 |
| pass 0: turn | 1.01 | 1.01 | 0.84 | 1.14 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.53 | 2.51 | 2.48 | 6.21 |
| pass 1: turn | 0.96 | 0.95 | 0.91 | 1.03 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.26 | 2.45 | 2.41 | 6.16 |
| pass 2: turn | 0.91 | 0.90 | 0.87 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.21 | 2.44 | 2.40 | 6.16 |
| pass 3: turn | 0.90 | 0.90 | 0.86 | 0.93 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 3.22 | 2.45 | 2.41 | 6.17 |
| pass 4: turn (the server before the next verify) | 3.78 | 1.75 | 1.18 | 2.52 |
| the draft's span (all passes) | 21.97 | 16.80 | 15.89 | 31.56 |
|   the draft's GPU time on its last card | 14.41 | 11.03 | 10.93 | 25.88 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 3.79 | 3.77 | 3.62 | 3.95 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.70 | 3.25 | 3.21 | 3.47 |

verify_gpu per card, median ms: 14.81, 12.88, 13.28, 13.38, 10.99, 13.40, 11.11, 13.38, 14.83

**w5h2, d0s: 76 typical steps, 5 draft passes a step, 5 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 144.45 | 139.90 | 137.68 | 154.16 |
| verify_span | 122.47 | 122.46 | 120.99 | 124.13 |
|   verify_gpu (the nine cards) | 118.13 | 118.14 | 116.65 | 119.83 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.34 | 4.32 | 4.30 | 4.47 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.19 | 1.19 | 1.17 | 1.21 |
| pass 0: turn | 1.01 | 1.01 | 0.84 | 1.14 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.53 | 2.51 | 2.48 | 6.21 |
| pass 1: turn | 0.96 | 0.95 | 0.91 | 1.03 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.26 | 2.45 | 2.41 | 6.16 |
| pass 2: turn | 0.91 | 0.90 | 0.87 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.21 | 2.44 | 2.40 | 6.16 |
| pass 3: turn | 0.90 | 0.90 | 0.86 | 0.93 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 3.22 | 2.45 | 2.41 | 6.17 |
| pass 4: turn (the server before the next verify) | 3.78 | 1.75 | 1.18 | 2.52 |
| the draft's span (all passes) | 21.97 | 16.80 | 15.89 | 31.56 |
|   the draft's GPU time on its last card | 14.41 | 11.03 | 10.93 | 25.88 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 3.79 | 3.77 | 3.62 | 3.95 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.70 | 3.25 | 3.21 | 3.47 |

verify_gpu per card, median ms: 14.81, 12.88, 13.28, 13.38, 10.99, 13.40, 11.11, 13.38, 14.83

