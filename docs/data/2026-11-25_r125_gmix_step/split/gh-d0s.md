gh: 1544 evaluations in 1 pieces (cut at gaps over 2000 ms): 1544 evaluations and 119 main passes
evaluations between main passes, by card: card 8: 472; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 119 | 0 | 118 | 4: 118 | 113 |

**gh, all requests: 113 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.94 | 116.18 | 113.70 | 125.90 |
| verify_span | 102.07 | 102.01 | 100.28 | 103.87 |
|   verify_gpu (the nine cards) | 98.46 | 98.41 | 96.55 | 100.25 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.61 | 3.58 | 3.70 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.06 | 1.06 | 1.04 | 1.08 |
| pass 0: turn | 0.98 | 0.98 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.48 | 2.60 | 2.57 | 6.41 |
| pass 1: turn | 0.97 | 0.96 | 0.92 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.25 | 2.56 | 2.51 | 6.34 |
| pass 2: turn | 0.92 | 0.90 | 0.89 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.18 | 2.54 | 2.51 | 6.33 |
| pass 3: turn (the server before the next verify) | 2.03 | 1.80 | 1.15 | 2.22 |
| the draft's span (all passes) | 15.86 | 13.46 | 12.98 | 24.96 |
|   the draft's GPU time on its last card | 10.97 | 8.76 | 8.65 | 20.16 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.75 | 3.02 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.43 | 3.15 | 3.11 | 3.29 |

verify_gpu per card, median ms: 15.04, 14.12, 11.94, 10.13, 11.44, 11.09, 7.62, 9.35, 7.75

**gh, d0s: 113 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.94 | 116.18 | 113.70 | 125.90 |
| verify_span | 102.07 | 102.01 | 100.28 | 103.87 |
|   verify_gpu (the nine cards) | 98.46 | 98.41 | 96.55 | 100.25 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.61 | 3.58 | 3.70 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.06 | 1.06 | 1.04 | 1.08 |
| pass 0: turn | 0.98 | 0.98 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.48 | 2.60 | 2.57 | 6.41 |
| pass 1: turn | 0.97 | 0.96 | 0.92 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.25 | 2.56 | 2.51 | 6.34 |
| pass 2: turn | 0.92 | 0.90 | 0.89 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.18 | 2.54 | 2.51 | 6.33 |
| pass 3: turn (the server before the next verify) | 2.03 | 1.80 | 1.15 | 2.22 |
| the draft's span (all passes) | 15.86 | 13.46 | 12.98 | 24.96 |
|   the draft's GPU time on its last card | 10.97 | 8.76 | 8.65 | 20.16 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.87 | 2.86 | 2.75 | 3.02 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.43 | 3.15 | 3.11 | 3.29 |

verify_gpu per card, median ms: 15.04, 14.12, 11.94, 10.13, 11.44, 11.09, 7.62, 9.35, 7.75

