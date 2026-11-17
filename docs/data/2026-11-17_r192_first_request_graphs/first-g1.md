trace 175 lines in 4 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 4

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 2827.74 | 0.10 | 2827.714 | 45 | 14 | 3/0 | 20/42 | 2/0 | 2007.61 | 180.39 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 77 | 229 | 178 | 8947.80 | 0.13 | 8947.483 | 229 | 178 | 1/0 | 80/226 | 2/0 | 7486.58 | 752.38 |
| decode2@4096 | 1 | 77 | 229 | 178 | 8656.82 | 0.12 | 8656.51 | 229 | 178 | 1/0 | 78/228 | 1/0 | 7215.10 | 717.54 |

### Depth 4096: pass 1 against pass 2

- generation lines: 77 and 77; drafted and accepted per line equal in 77 of 77 paired lines
- the span, pass 1 − pass 2, ms: all 77 paired steps 290.97; steps 0-2 104.46 (109.72, -0.59, -4.67); steps 3 on 186.52 (median 0.24 a step); the server's gen_ms 8947.483 − 8656.51 = 290.97
- the span, median a step: pass 1 112.61, pass 2 112.07 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.91 | 2.04 | -0.13 | -0.00 | -0.001 |
| draft | 953.73 | 912.62 | 41.10 | 6.11 | 0.050 |
| checkpoints | 0.36 | 0.30 | 0.06 | 0.01 | 0.000 |
| batch | 0.60 | 0.56 | 0.04 | 0.01 | 0.000 |
| verify | 7826.10 | 7575.44 | 250.66 | 98.33 | 0.165 |
| catch-up | 107.00 | 107.30 | -0.31 | -0.05 | -0.007 |
| acceptance | 38.87 | 38.77 | 0.10 | 0.04 | 0.001 |
| between | 19.22 | 19.79 | -0.56 | 0.02 | 0.000 |
| verify: waits for the cards | 7378.64 | 7186.42 | 192.22 | 83.40 | -0.049 |
| main passes' GPU time (all cards) | 7486.58 | 7215.10 | 271.48 | 94.98 | -0.049 |
| the draft's GPU time | 752.38 | 717.54 | 34.83 | 5.43 | -0.015 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 7378.64 | 7186.42 | 192.22 |
| `v.ub.in#0 -> v.ub.comp` | 415.40 | 357.02 | 58.38 |
| `d.sync#1 -> d.sync.end` | 10.90 | 1.99 | 8.91 |
| `mtp.dp.ret#0 -> d.sync` | 11.66 | 4.60 | 7.06 |
| `d.sync#8 -> d.sync.end` | 8.09 | 1.49 | 6.61 |
| `s.vp.end#0 -> d.sync` | 10.81 | 4.35 | 6.46 |
| `mtp.dp.ret#1 -> d.sync` | 10.72 | 4.31 | 6.40 |
| `d.sync#7 -> d.sync.end` | 206.78 | 213.01 | -6.23 |
| `mtp.dp.row#2 -> mtp.drf.end` | 9.20 | 3.67 | 5.53 |
| `d.sync#15 -> d.sync.end` | 6.89 | 1.38 | 5.51 |

Step by step, pass 1 − pass 2, ms: 109.7, -0.6, -4.7, -2.0, 0.1, 0.2, 1.0, 1.1, 16.2, 11.9, 4.3, 0.9, -0.2, 1.3, 0.3, 0.7, 1.2, 0.3, 0.9, -0.2, -0.0, 1.3, 0.8, 0.7, 0.7, 1.3, 0.4, 0.1, 1.6, 1.6, 7.9, 1.1, -0.2, 0.9, 0.0, 1.0, 0.9, 0.1, 0.9, 1.0, 0.7, 0.0, 0.2, -0.2, -0.2, -0.0, -0.6, -0.9, -0.2, 0.2, -0.0, 4.2, 0.3, 0.4, -0.3, 2.9, 0.3, -0.1, -0.1, -0.2, -0.2, 0.1, 0.0, -0.1, 2.4, 0.1, -0.4, -0.0, -0.1, -0.1, 0.8, -0.2, -0.3, -0.1, 0.4, -0.3, 118.5

