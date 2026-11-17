trace 792 lines in 12 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 12

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 2811.44 | 0.10 | 2811.424 | 45 | 14 | 3/0 | 20/42 | 2/0 | 2005.10 | 178.27 |
| decode1@0 | 1 | 108 | 321 | 147 | 11396.62 | 0.12 | 11396.586 | 321 | 147 | 1/0 | 116/313 | 2/0 | 9308.26 | 1036.81 |
| decode2@0 | 1 | 107 | 318 | 148 | 10720.71 | 0.09 | 10720.639 | 318 | 148 | 1/0 | 109/316 | 1/0 | 8988.03 | 899.72 |
| samples@0 | 1 | 101 | 302 | 154 | 10735.46 | 0.13 | 10735.405 | 302 | 154 | 1/0 | 110/294 | 1/0 | 8783.62 | 1006.75 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 78 | 231 | 177 | 9042.17 | 2.59 | 9044.29 | 231 | 177 | 1/0 | 80/229 | 2/0 | 7567.37 | 759.28 |
| decode2@4096 | 1 | 78 | 231 | 177 | 8717.42 | 0.13 | 8717.096 | 231 | 177 | 1/0 | 79/230 | 1/0 | 7341.91 | 724.33 |
| samples@4096 | 1 | 76 | 228 | 179 | 8660.35 | 0.14 | 8660.047 | 228 | 179 | 0/0 | 79/225 | 1/0 | 7208.47 | 767.29 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 71 | 212 | 184 | 8596.65 | 0.15 | 8596.101 | 212 | 184 | 1/0 | 75/208 | 2/0 | 7177.83 | 707.25 |
| decode2@8192 | 1 | 70 | 209 | 185 | 8158.99 | 0.09 | 8158.372 | 209 | 185 | 1/0 | 71/208 | 1/0 | 6834.04 | 671.03 |
| samples@8192 | 1 | 73 | 216 | 182 | 8727.46 | 0.09 | 8726.86 | 216 | 182 | 1/0 | 75/214 | 1/0 | 7089.45 | 713.12 |

### Depth 0: pass 1 against pass 2

- generation lines: 108 and 107; drafted and accepted per line equal in 92 of 107 paired lines
- the span, pass 1 − pass 2, ms: all 107 paired steps 479.11; steps 0-2 124.91 (128.61, -3.81, 0.12); steps 3 on 354.19 (median 0.82 a step); the server's gen_ms 11396.586 − 10720.639 = 675.95
- the span, median a step: pass 1 101.01, pass 2 99.73 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.29 | 1.28 | 0.01 | -0.00 | 0.000 |
| draft | 1292.73 | 1128.40 | 164.33 | -1.73 | 0.225 |
| checkpoints | 0.25 | 0.24 | 0.00 | 0.00 | 0.000 |
| batch | 0.92 | 0.71 | 0.20 | 0.01 | 0.002 |
| verify | 9724.45 | 9410.78 | 313.67 | 126.70 | 0.486 |
| catch-up | 105.68 | 105.54 | 0.14 | -0.09 | 0.000 |
| acceptance | 47.71 | 47.52 | 0.19 | 0.01 | -0.001 |
| between | 26.80 | 26.24 | 0.56 | 0.02 | 0.001 |
| verify: waits for the cards | 9005.01 | 8847.24 | 157.77 | -12.03 | 0.110 |
| main passes' GPU time (all cards) | 9129.09 | 8988.03 | 141.06 | -5.54 | 0.005 |
| the draft's GPU time | 1035.91 | 899.72 | 136.20 | -3.42 | -0.020 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.ub.in#0 -> v.ub.comp` | 692.27 | 526.70 | 165.57 |
| `v.sync#0 -> v.sync.end` | 9005.01 | 8847.24 | 157.77 |
| `d.sync#1 -> d.sync.end` | 35.59 | 2.69 | 32.90 |
| `d.sync#8 -> d.sync.end` | 32.53 | 1.96 | 30.57 |
| `d.sync#15 -> d.sync.end` | 29.99 | 1.85 | 28.14 |
| `mtp.dp.ret#0 -> d.sync` | 21.87 | 6.27 | 15.60 |
| `s.vp.end#0 -> d.sync` | 20.60 | 6.17 | 14.44 |
| `mtp.dp.ret#1 -> d.sync` | 20.07 | 5.82 | 14.25 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.05 | -11.05 |
| `mtp.dp.row#2 -> mtp.drf.end` | 16.07 | 5.14 | 10.93 |

Step by step, pass 1 − pass 2, ms: 128.6, -3.8, 0.1, 0.7, 1.7, 1.6, -3.7, 0.7, 2.2, 1.1, 0.8, 1.9, 1.3, 1.0, 1.2, 1.2, 1.1, 0.9, 1.5, -0.2, 0.7, 1.2, 12.2, 12.9, -1.2, 12.0, 11.8, 4.7, -0.3, -0.2, 1.2, 3.1, 0.3, 0.3, 0.1, 1.5, 2.4, -0.2, 0.1, -0.1, 0.9, 0.3, -0.2, 1.8, 0.3, 0.6, 0.9, 1.1, 1.3, 0.7, 0.5, -0.1, 0.1, 0.0, 2.3, 0.8, 0.0, 0.2, 1.7, 0.8, -3.3, 0.5, 0.5, 0.0, -0.0, 0.9, -0.2, -0.1, 0.1, 1.2, 0.0, 0.9, 0.5, 0.4, 0.6, 12.7, 14.5, 4.2, 1.4, -1.4, 10.4, 14.5, 3.4, -0.9, 3.2, 0.4, -22.1, 141.9, 5.1, 0.8, 0.4, 1.8, 1.9, -0.4, 2.0, 0.2, -0.4, 16.1, 13.3, 6.3, 0.3, -0.8, 0.9, 0.4, -0.2, 0.9, 29.8

### Depth 4096: pass 1 against pass 2

- generation lines: 78 and 78; drafted and accepted per line equal in 78 of 78 paired lines
- the span, pass 1 − pass 2, ms: all 78 paired steps 324.75; steps 0-2 124.81 (123.78, 2.18, -1.15); steps 3 on 199.94 (median 0.30 a step); the server's gen_ms 9044.29 − 8717.096 = 327.19
- the span, median a step: pass 1 112.54, pass 2 111.99 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.78 | 1.85 | -0.06 | 0.02 | -0.001 |
| draft | 960.67 | 911.59 | 49.08 | 7.13 | 0.122 |
| checkpoints | 0.19 | 0.16 | 0.03 | 0.02 | 0.000 |
| batch | 0.70 | 0.59 | 0.11 | 0.00 | 0.001 |
| verify | 7916.23 | 7639.28 | 276.95 | 117.88 | 0.087 |
| catch-up | 106.11 | 107.03 | -0.93 | -0.13 | -0.007 |
| acceptance | 36.14 | 36.43 | -0.29 | -0.07 | -0.002 |
| between | 20.35 | 20.48 | -0.14 | -0.03 | -0.000 |
| verify: waits for the cards | 7478.62 | 7261.63 | 216.99 | 105.98 | -0.163 |
| main passes' GPU time (all cards) | 7567.37 | 7341.91 | 225.46 | 114.69 | -0.151 |
| the draft's GPU time | 759.28 | 724.33 | 34.95 | 5.82 | -0.007 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 7478.62 | 7261.63 | 216.99 |
| `v.ub.in#0 -> v.ub.comp` | 401.96 | 346.99 | 54.97 |
| `d.sync#1 -> d.sync.end` | 10.95 | 2.03 | 8.92 |
| `mtp.dp.ret#0 -> d.sync` | 12.39 | 4.50 | 7.89 |
| `mtp.dp.ret#1 -> d.sync` | 11.61 | 4.31 | 7.29 |
| `d.sync#7 -> d.sync.end` | 212.77 | 219.85 | -7.08 |
| `s.vp.end#0 -> d.sync` | 11.27 | 4.38 | 6.89 |
| `d.sync#8 -> d.sync.end` | 8.11 | 1.42 | 6.69 |
| `mtp.dp.row#2 -> mtp.drf.end` | 9.81 | 3.69 | 6.12 |
| `d.sync#15 -> d.sync.end` | 6.87 | 1.36 | 5.51 |

Step by step, pass 1 − pass 2, ms: 123.8, 2.2, -1.1, 1.5, 0.8, 0.8, 1.3, 16.0, 11.5, 4.6, 0.4, 0.1, 1.2, 1.4, 4.5, 0.9, 1.1, -0.2, 1.4, -0.1, -0.1, 2.6, 0.8, 0.2, 0.9, 0.8, 0.4, -0.2, 1.3, 1.4, 0.6, 1.7, 0.2, 0.5, 0.9, 1.2, -0.2, 0.5, 0.1, 0.7, 0.6, -0.0, 0.4, 0.2, -0.3, -0.2, 0.0, 0.1, -0.3, -0.3, -0.3, 3.9, 0.8, -0.2, 2.6, 0.3, 0.4, 0.0, -0.3, -0.1, -0.3, -0.1, 0.1, -0.4, -0.6, 7.1, -0.2, -0.2, -0.2, -0.4, -0.4, 0.0, -0.2, -0.3, 0.5, 0.0, -0.4, 127.4

### Depth 8192: pass 1 against pass 2

- generation lines: 71 and 70; drafted and accepted per line equal in 53 of 70 paired lines
- the span, pass 1 − pass 2, ms: all 70 paired steps 193.01; steps 0-2 127.18 (125.20, 2.50, -0.52); steps 3 on 65.83 (median 0.46 a step); the server's gen_ms 8596.101 − 8158.372 = 437.73
- the span, median a step: pass 1 116.57, pass 2 116.19 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.36 | 2.42 | -0.07 | 0.01 | -0.001 |
| draft | 909.41 | 873.00 | 36.41 | 0.48 | 0.046 |
| checkpoints | 0.15 | 0.16 | -0.01 | 0.00 | 0.000 |
| batch | 0.64 | 0.58 | 0.06 | 0.03 | 0.000 |
| verify | 7261.08 | 7104.48 | 156.61 | 126.78 | 0.339 |
| catch-up | 127.56 | 127.91 | -0.35 | 0.04 | -0.005 |
| acceptance | 31.10 | 31.79 | -0.69 | -0.14 | -0.001 |
| between | 19.70 | 18.65 | 1.05 | -0.02 | 0.012 |
| verify: waits for the cards | 6901.28 | 6762.10 | 139.18 | 114.66 | -0.080 |
| main passes' GPU time (all cards) | 6963.14 | 6834.04 | 129.11 | 122.97 | -0.094 |
| the draft's GPU time | 699.83 | 671.03 | 28.80 | -2.03 | 0.011 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6901.28 | 6762.10 | 139.18 |
| `v.ub.in#0 -> v.ub.comp` | 341.90 | 312.75 | 29.15 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 10.90 | -10.90 |
| `d.sync#1 -> d.sync.end` | 10.96 | 1.77 | 9.19 |
| `d.sync#8 -> d.sync.end` | 7.63 | 1.43 | 6.20 |
| `d.sync#7 -> d.sync.end` | 199.34 | 204.77 | -5.43 |
| `d.sync#15 -> d.sync.end` | 5.99 | 1.19 | 4.80 |
| `s.vp.end#0 -> d.sync` | 7.89 | 4.02 | 3.87 |
| `mtp.dp.ret#0 -> d.sync` | 8.00 | 4.19 | 3.82 |
| `mtp.dp.ret#1 -> d.sync` | 7.49 | 3.90 | 3.59 |

Step by step, pass 1 − pass 2, ms: 125.2, 2.5, -0.5, 12.6, 15.8, 4.7, 0.2, 3.9, -1.7, -2.0, 1.9, -0.8, 1.0, -0.0, 3.3, 0.1, -0.4, 1.9, 1.6, -1.3, 2.1, -1.1, 0.8, -0.9, 0.4, 1.0, -0.7, 0.7, -0.4, 0.3, 0.4, 10.9, -0.6, 0.5, 0.1, 2.8, 1.5, 1.0, 2.5, 0.6, 2.2, -0.2, 0.7, -0.0, 1.3, -0.4, -2.0, 1.9, -0.0, 1.5, -1.0, 0.8, -1.9, 0.6, -0.1, 0.5, -0.8, 0.5, -1.6, 0.9, 0.1, 0.5, -1.2, 1.1, -0.9, 0.0, -0.2, 1.1, -1.3, 0.9

