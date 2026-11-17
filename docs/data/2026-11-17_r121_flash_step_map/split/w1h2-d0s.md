w1h2: 2304 evaluations in 1 pieces (cut at gaps over 2000 ms): 2304 evaluations and 256 main passes
evaluations between main passes, by card: ; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 256 | 1 | 254 | 0: 254 | 249 |

**w1h2, all requests: 249 typical steps, 0 draft passes a step, 0 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 52.30 | 52.14 | 51.17 | 54.06 |
| verify_span | 52.30 | 52.14 | 51.17 | 54.06 |
|   verify_gpu (the nine cards) | 49.54 | 49.45 | 49.34 | 49.79 |
|   verify_rest (crossings, rows back, acceptance, server) | 2.76 | 2.61 | 1.72 | 4.44 |
| the draft's span (all passes) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the draft's GPU time on its last card | 0.00 | 0.00 | 0.00 | 0.00 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 0.00 | 0.00 | 0.00 | 0.00 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.18 | 3.10 | 3.01 | 3.20 |

verify_gpu per card, median ms: 6.00, 5.69, 5.79, 5.87, 4.81, 5.68, 4.87, 5.59, 5.17

**w1h2, d0s: 249 typical steps, 0 draft passes a step, 0 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 52.30 | 52.14 | 51.17 | 54.06 |
| verify_span | 52.30 | 52.14 | 51.17 | 54.06 |
|   verify_gpu (the nine cards) | 49.54 | 49.45 | 49.34 | 49.79 |
|   verify_rest (crossings, rows back, acceptance, server) | 2.76 | 2.61 | 1.72 | 4.44 |
| the draft's span (all passes) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the draft's GPU time on its last card | 0.00 | 0.00 | 0.00 | 0.00 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 0.00 | 0.00 | 0.00 | 0.00 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.18 | 3.10 | 3.01 | 3.20 |

verify_gpu per card, median ms: 6.00, 5.69, 5.79, 5.87, 4.81, 5.68, 4.87, 5.59, 5.17

