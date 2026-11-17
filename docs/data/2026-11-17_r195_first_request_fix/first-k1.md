trace 457 lines in 10 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 10

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 2878.13 | 0.11 | 2878.135 | 45 | 14 | 3/0 | 20/42 | 2/0 | 2012.57 | 181.15 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 77 | 229 | 178 | 8868.41 | 0.09 | 8868.062 | 229 | 178 | 1/0 | 80/226 | 2/0 | 7390.10 | 762.31 |
| decode2@4096 | 1 | 77 | 229 | 178 | 8639.63 | 1.66 | 8640.834 | 229 | 178 | 1/0 | 78/228 | 1/0 | 7250.56 | 727.00 |
| prime@6144 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 13/0 | 0.00 | 0.00 |
| decode1@6144 | 1 | 69 | 204 | 186 | 8354.76 | 0.14 | 8354.336 | 204 | 186 | 1/0 | 71/202 | 2/0 | 6987.36 | 684.56 |
| decode2@6144 | 1 | 68 | 201 | 187 | 8053.73 | 0.09 | 8053.278 | 201 | 187 | 1/0 | 69/200 | 1/0 | 6760.02 | 651.43 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 69 | 207 | 186 | 8264.02 | 0.10 | 8263.418 | 207 | 186 | 0/0 | 72/204 | 2/0 | 6879.00 | 697.54 |
| decode2@8192 | 1 | 68 | 204 | 187 | 7957.98 | 0.14 | 7956.58 | 204 | 187 | 0/0 | 69/203 | 1/0 | 6561.36 | 657.47 |

### Depth 4096: pass 1 against pass 2

- generation lines: 77 and 77; drafted and accepted per line equal in 77 of 77 paired lines
- the span, pass 1 − pass 2, ms: all 77 paired steps 228.78; steps 0-2 29.56 (20.69, 7.52, 1.35); steps 3 on 199.22 (median 0.39 a step); the server's gen_ms 8868.062 − 8640.834 = 227.23
- the span, median a step: pass 1 112.88, pass 2 112.26 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.91 | 1.95 | -0.04 | -0.01 | -0.001 |
| draft | 965.86 | 921.57 | 44.29 | 7.03 | 0.078 |
| checkpoints | 0.36 | 0.31 | 0.05 | 0.00 | 0.000 |
| batch | 0.60 | 0.54 | 0.06 | 0.01 | 0.000 |
| verify | 7733.00 | 7547.07 | 185.92 | 22.63 | 0.316 |
| catch-up | 108.13 | 109.18 | -1.04 | -0.12 | -0.020 |
| acceptance | 39.01 | 38.84 | 0.17 | -0.00 | 0.008 |
| between | 19.54 | 20.16 | -0.62 | 0.02 | -0.001 |
| verify: waits for the cards | 7287.41 | 7158.71 | 128.70 | 13.35 | 0.045 |
| main passes' GPU time (all cards) | 7390.10 | 7250.56 | 139.54 | 18.69 | 0.035 |
| the draft's GPU time | 762.31 | 727.00 | 35.31 | 5.90 | -0.004 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 7287.41 | 7158.71 | 128.70 |
| `v.ub.in#0 -> v.ub.comp` | 413.52 | 356.40 | 57.13 |
| `d.sync#1 -> d.sync.end` | 10.93 | 1.96 | 8.98 |
| `mtp.dp.ret#0 -> d.sync` | 12.02 | 4.62 | 7.41 |
| `s.vp.end#0 -> d.sync` | 11.19 | 4.44 | 6.75 |
| `d.sync#8 -> d.sync.end` | 8.14 | 1.52 | 6.63 |
| `mtp.dp.ret#1 -> d.sync` | 10.99 | 4.44 | 6.55 |
| `mtp.dp.row#2 -> mtp.drf.end` | 9.72 | 3.80 | 5.93 |
| `d.sync#15 -> d.sync.end` | 6.96 | 1.40 | 5.56 |
| `d.sync#7 -> d.sync.end` | 210.26 | 215.64 | -5.37 |

Step by step, pass 1 − pass 2, ms: 20.7, 7.5, 1.3, 1.2, 0.6, 0.8, 1.1, 1.6, 16.3, 12.4, 5.1, 0.8, 0.3, 1.7, 1.0, 0.6, 0.7, 0.4, 0.2, 0.3, 0.1, 1.3, 0.9, 0.4, 2.8, 7.5, 0.6, 0.3, 0.7, 0.9, 1.0, 1.1, -0.2, 1.0, 0.1, 1.7, 0.8, -0.0, 0.6, 1.4, 1.2, 0.3, 0.1, -0.0, -1.4, -0.3, -0.3, 0.1, 0.4, 0.3, 0.1, 3.5, 0.4, 0.3, 0.1, 3.5, -0.0, 1.5, 0.2, 0.0, 0.1, 0.0, -0.3, 0.1, 2.3, -0.1, 0.2, 0.0, 0.0, -0.0, 0.5, 0.1, 0.7, -0.0, -0.2, 0.0, 117.6

### Depth 6144: pass 1 against pass 2

- generation lines: 69 and 68; drafted and accepted per line equal in 55 of 68 paired lines
- the span, pass 1 − pass 2, ms: all 68 paired steps 96.44; steps 0-2 33.79 (9.26, 11.99, 12.54); steps 3 on 62.65 (median 0.11 a step); the server's gen_ms 8354.336 − 8053.278 = 301.06
- the span, median a step: pass 1 118.54, pass 2 118.35 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.00 | 2.11 | -0.11 | 0.00 | -0.001 |
| draft | 883.43 | 841.37 | 42.06 | 13.70 | 0.026 |
| checkpoints | 0.31 | 0.29 | 0.02 | -0.00 | 0.000 |
| batch | 0.53 | 0.49 | 0.04 | 0.00 | 0.000 |
| verify | 7098.36 | 7043.18 | 55.17 | 19.87 | 0.101 |
| catch-up | 112.96 | 113.69 | -0.73 | 0.05 | 0.006 |
| acceptance | 33.66 | 33.59 | 0.06 | 0.17 | -0.001 |
| between | 18.93 | 19.01 | -0.07 | -0.00 | 0.000 |
| verify: waits for the cards | 6722.73 | 6674.51 | 48.22 | 2.07 | -0.098 |
| main passes' GPU time (all cards) | 6799.58 | 6760.02 | 39.55 | 11.57 | -0.112 |
| the draft's GPU time | 682.97 | 651.43 | 31.54 | 11.76 | -0.040 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6722.73 | 6674.51 | 48.22 |
| `v.ub.in#0 -> v.ub.comp` | 356.67 | 338.38 | 18.29 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.10 | -11.10 |
| `d.sync#8 -> d.sync.end` | 8.05 | 1.40 | 6.65 |
| `d.sync#1 -> d.sync.end` | 8.08 | 1.78 | 6.30 |
| `s.vp.end#0 -> d.sync` | 10.24 | 4.74 | 5.49 |
| `mtp.dp.ret#0 -> d.sync` | 10.03 | 4.77 | 5.25 |
| `mtp.dp.ret#1 -> d.sync` | 10.26 | 5.09 | 5.17 |
| `d.sync#15 -> d.sync.end` | 6.29 | 1.22 | 5.08 |
| `d.sync#14 -> d.sync.end` | 99.72 | 103.63 | -3.91 |

Step by step, pass 1 − pass 2, ms: 9.3, 12.0, 12.5, 10.1, 4.8, 1.6, -0.6, -2.2, -0.4, -0.9, 0.7, -0.0, 0.2, -1.6, 2.4, 4.1, -0.7, -0.1, -2.9, -1.3, 4.3, -0.1, 1.3, 0.1, -2.0, 2.5, -0.1, 0.6, 2.2, 1.7, -1.9, 1.1, -0.2, 0.2, -0.3, -0.3, -0.4, 0.6, -1.7, -0.3, 0.0, 1.3, -1.1, 1.0, -1.6, 0.2, 0.8, -8.9, -0.5, 0.5, 0.2, 1.8, -2.2, -0.1, -1.1, 0.9, 3.9, 8.1, 0.1, 0.6, 1.0, -2.8, -0.4, 1.1, -1.7, 1.5, 0.4, 39.0

### Depth 8192: pass 1 against pass 2

- generation lines: 69 and 68; drafted and accepted per line equal in 62 of 68 paired lines
- the span, pass 1 − pass 2, ms: all 68 paired steps 188.42; steps 0-2 135.73 (127.46, 7.52, 0.75); steps 3 on 52.69 (median 0.47 a step); the server's gen_ms 8263.418 − 7956.58 = 306.84
- the span, median a step: pass 1 117.13, pass 2 116.86 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.39 | 2.40 | -0.01 | 0.01 | 0.000 |
| draft | 902.58 | 866.51 | 36.07 | 4.68 | 0.061 |
| checkpoints | 0.36 | 0.31 | 0.06 | 0.01 | 0.000 |
| batch | 0.63 | 0.50 | 0.13 | 0.01 | 0.000 |
| verify | 7062.98 | 6909.12 | 153.86 | 131.08 | 0.379 |
| catch-up | 125.96 | 125.01 | 0.95 | -0.06 | 0.005 |
| acceptance | 32.82 | 34.53 | -1.70 | 0.01 | -0.004 |
| between | 18.66 | 19.60 | -0.93 | -0.01 | 0.002 |
| verify: waits for the cards | 6703.29 | 6589.93 | 113.36 | 117.31 | -0.006 |
| main passes' GPU time (all cards) | 6781.06 | 6561.36 | 219.70 | 127.90 | 0.124 |
| the draft's GPU time | 687.87 | 657.47 | 30.40 | 3.83 | -0.027 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6703.29 | 6589.93 | 113.36 |
| `v.ub.in#0 -> v.ub.comp` | 340.71 | 300.90 | 39.81 |
| `mtp.dp.ret#0 -> d.sync` | 10.88 | 4.13 | 6.74 |
| `d.sync#8 -> d.sync.end` | 7.99 | 1.38 | 6.61 |
| `d.sync#1 -> d.sync.end` | 8.10 | 1.81 | 6.30 |
| `d.sync#7 -> d.sync.end` | 191.30 | 197.43 | -6.14 |
| `mtp.dp.ret#1 -> d.sync` | 9.72 | 4.02 | 5.71 |
| `s.vp.end#0 -> d.sync` | 9.71 | 4.06 | 5.65 |
| `d.sync#15 -> d.sync.end` | 5.98 | 1.24 | 4.74 |
| `mtp.dp.row#2 -> mtp.drf.end` | 7.89 | 3.41 | 4.48 |

Step by step, pass 1 − pass 2, ms: 127.5, 7.5, 0.7, 1.0, 1.3, 11.1, 15.5, 2.5, -0.3, 3.6, -0.1, -0.7, 1.2, -0.9, 2.5, 1.5, -0.1, 1.7, -0.7, 0.6, 1.9, 1.2, -0.2, 0.8, -1.0, 0.1, 1.2, 0.1, 0.8, 0.1, 0.5, 2.5, -4.6, -0.1, 0.7, 0.8, -1.2, 1.8, 1.9, 2.8, -0.1, 0.4, 0.7, 1.0, 0.2, 0.3, -0.5, 1.7, 2.2, -0.8, 1.0, -1.5, 0.6, 0.3, 0.4, -0.6, 0.6, -1.1, -0.4, 0.1, 0.6, -1.3, 1.5, -1.1, -0.1, 0.9, -0.9, -1.3

