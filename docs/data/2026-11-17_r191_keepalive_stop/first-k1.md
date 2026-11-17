trace 175 lines in 4 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 4

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 4485.58 | 0.09 | 4485.546 | 45 | 14 | 3/0 | 20/42 | 2/0 | 3701.11 | 172.08 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 77 | 229 | 178 | 9686.08 | 0.09 | 9685.701 | 229 | 178 | 1/0 | 80/226 | 2/0 | 7237.43 | 1744.54 |
| decode2@4096 | 1 | 77 | 229 | 178 | 8632.66 | 0.13 | 8632.355 | 229 | 178 | 1/0 | 78/228 | 1/0 | 7172.31 | 718.88 |

### Depth 4096: pass 1 against pass 2

- generation lines: 77 and 77; drafted and accepted per line equal in 77 of 77 paired lines
- the span, pass 1 − pass 2, ms: all 77 paired steps 1053.42; steps 0-2 1007.94 (1001.76, 5.79, 0.39); steps 3 on 45.48 (median 0.07 a step); the server's gen_ms 9685.701 − 8632.355 = 1053.35
- the span, median a step: pass 1 112.24, pass 2 112.21 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.98 | 2.10 | -0.12 | -0.01 | -0.002 |
| draft | 1949.67 | 915.43 | 1034.24 | 1003.82 | 0.030 |
| checkpoints | 0.35 | 0.32 | 0.03 | 0.01 | 0.000 |
| batch | 0.64 | 0.60 | 0.04 | 0.00 | 0.000 |
| verify | 7568.41 | 7546.47 | 21.94 | 4.02 | 0.079 |
| catch-up | 106.81 | 109.07 | -2.26 | 0.02 | -0.016 |
| acceptance | 38.55 | 39.45 | -0.90 | 0.04 | -0.016 |
| between | 19.66 | 19.21 | 0.45 | 0.03 | 0.000 |
| verify: waits for the cards | 7138.39 | 7149.68 | -11.28 | -6.54 | 0.001 |
| main passes' GPU time (all cards) | 7237.43 | 7172.31 | 65.12 | 1.70 | -0.003 |
| the draft's GPU time | 1744.54 | 718.88 | 1025.66 | 1002.68 | -0.010 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `d.sync#0 -> d.sync.end` | 1210.99 | 213.88 | 997.11 |
| `v.ub.in#0 -> v.ub.comp` | 397.45 | 365.18 | 32.26 |
| `v.sync#0 -> v.sync.end` | 7138.39 | 7149.68 | -11.28 |
| `d.sync#8 -> d.sync.end` | 8.15 | 1.66 | 6.50 |
| `d.sync#1 -> d.sync.end` | 8.33 | 2.23 | 6.10 |
| `d.sync#15 -> d.sync.end` | 6.95 | 1.45 | 5.50 |
| `mtp.dp.ret#1 -> d.sync` | 10.03 | 5.69 | 4.34 |
| `mtp.dp.ret#0 -> d.sync` | 10.46 | 6.48 | 3.98 |
| `mtp.dp.row#2 -> mtp.drf.end` | 8.64 | 5.47 | 3.16 |
| `s.vp.end#0 -> d.sync` | 9.59 | 6.53 | 3.06 |

Step by step, pass 1 − pass 2, ms: 1001.8, 5.8, 0.4, 0.0, 1.1, 1.2, 1.0, 1.4, 12.5, 11.8, -0.8, 0.8, -0.1, 1.6, -0.6, 0.3, 2.6, -0.2, -1.8, -0.6, -1.1, -1.7, 0.5, 0.6, 1.5, 0.9, 0.3, 0.1, 0.9, 1.6, 0.9, 0.7, -0.1, 0.6, -1.6, 1.4, 1.6, -0.2, -1.7, -1.3, 0.8, -1.1, 0.1, -3.4, -0.2, -0.0, -0.1, -0.1, -0.1, -0.3, -0.2, 3.8, 0.2, 0.2, 0.1, 2.7, -0.0, 0.0, -0.3, -0.1, 0.1, -0.0, -0.1, 5.4, 2.5, 0.7, -0.1, -0.1, 0.1, 0.2, 0.5, -0.2, -0.2, -0.2, -0.2, -0.0, 0.8

