## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12811.9 | 148.98 | 19.90 | 8516.7 | 84 |
| e | long | 2760 | 256 | 249 | 171 | 0.687 | 85 | 3.012 | 12043.5 | 141.69 | 21.17 | 26477.0 | 83 |
| e-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12830.8 | 149.20 | 19.87 | 8518.5 | 84 |
| e-2 | long | 2760 | 256 | 267 | 165 | 0.618 | 91 | 2.813 | 12717.2 | 139.75 | 20.05 | 26499.0 | 89 |
| f | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10856.6 | 126.24 | 23.49 | 7522.2 | 84 |
| f | long | 2760 | 256 | 246 | 173 | 0.703 | 83 | 3.084 | 10624.5 | 128.01 | 24.00 | 26436.8 | 81 |
| f-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10824.7 | 125.87 | 23.56 | 7545.3 | 84 |
| f-2 | long | 2760 | 256 | 273 | 163 | 0.597 | 93 | 2.753 | 12863.7 | 138.32 | 19.82 | 26304.4 | 91 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 108.32 (110.73; 81) | 116.56 (118.88; 80) | 112.44 |
| e-2 | 108.51 (110.81; 81) | 116.76 (118.40; 86) | 112.64 |
| f | 108.09 (110.30; 80) | 115.68 (116.81; 78) | 111.89 |
| f-2 | 107.92 (110.20; 80) | 116.06 (117.43; 88) | 111.99 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 91.74 (91.90; 81) | 98.90 (99.08; 80) | 95.32 |
| e-2 | 91.67 (91.94; 81) | 99.02 (99.23; 86) | 95.34 |
| f | 91.87 (92.04; 80) | 98.85 (99.10; 78) | 95.36 |
| f-2 | 91.74 (91.94; 80) | 98.82 (99.05; 88) | 95.28 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 88.20 (88.38; 81) | 95.15 (95.31; 80) | 91.67 |
| e-2 | 88.15 (88.35; 81) | 95.28 (95.49; 86) | 91.71 |
| f | 88.27 (88.47; 80) | 95.15 (95.38; 78) | 91.71 |
| f-2 | 88.22 (88.42; 80) | 95.05 (95.31; 88) | 91.64 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 3.46 (3.52; 81) | 3.71 (3.77; 80) | 3.59 |
| e-2 | 3.46 (3.59; 81) | 3.70 (3.74; 86) | 3.58 |
| f | 3.44 (3.57; 80) | 3.70 (3.72; 78) | 3.57 |
| f-2 | 3.45 (3.52; 80) | 3.70 (3.74; 88) | 3.57 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 16.28 (18.83; 81) | 17.46 (19.80; 80) | 16.87 |
| e-2 | 16.36 (18.88; 81) | 17.56 (19.17; 86) | 16.96 |
| f | 16.00 (18.27; 80) | 16.86 (17.70; 78) | 16.43 |
| f-2 | 16.00 (18.27; 80) | 17.24 (18.38; 88) | 16.62 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 10.98 (13.09; 81) | 12.32 (14.37; 80) | 11.65 |
| e-2 | 10.97 (13.09; 81) | 12.29 (13.70; 86) | 11.63 |
| f | 10.83 (12.99; 80) | 12.18 (12.55; 78) | 11.50 |
| f-2 | 10.83 (12.99; 80) | 12.22 (13.31; 88) | 11.53 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.83 (1.86; 81) | 1.82 (1.83; 80) | 1.82 |
| e-2 | 1.83 (1.83; 81) | 1.83 (1.83; 86) | 1.83 |
| f | 4.21 (3.84; 80) | 4.46 (4.40; 78) | 4.33 |
| f-2 | 4.21 (3.82; 80) | 4.48 (4.24; 88) | 4.35 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.35 (1.46; 81) | 1.38 (1.52; 80) | 1.37 |
| e-2 | 1.36 (1.47; 81) | 1.39 (1.47; 86) | 1.38 |
| f | -1.05 (-0.52; 80) | -1.32 (-1.21; 78) | -1.19 |
| f-2 | -1.07 (-0.53; 80) | -1.31 (-1.01; 88) | -1.19 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.99 (2.43; 81) | 1.91 (2.08; 80) | 1.95 |
| e-2 | 2.03 (2.48; 81) | 2.02 (2.17; 86) | 2.03 |
| f | 1.84 (1.95; 80) | 1.39 (1.96; 78) | 1.61 |
| f-2 | 1.83 (1.98; 80) | 1.80 (1.84; 88) | 1.81 |

## Each arm against the reference loads

Reference: e and e-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| f | mean of the medians | 111.89 | -0.65 | 0.20 (112.44, 112.64) | gain |
| f | greedy, median | 108.09 | -0.32 | 0.19 (108.32, 108.51) | gain |
| f | long, median | 115.68 | -0.98 | 0.20 (116.56, 116.76) | gain |
| f-2 | mean of the medians | 111.99 | -0.55 | 0.20 (112.44, 112.64) | gain |
| f-2 | greedy, median | 107.92 | -0.49 | 0.19 (108.32, 108.51) | gain |
| f-2 | long, median | 116.06 | -0.60 | 0.20 (116.56, 116.76) | gain |

## The greedy request, step by step against e

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| e-2 | yes | 254 | 170 | 81 | +0.08 | 0.22 | +0.00 | +0.04 | +0.04 | -0.03 | +0.01 | +0.06 |
| f | yes | 254 | 170 | not paired (84 and 84 steps) |  |  |  |  |  |  |  |  |
| f-2 | yes | 254 | 170 | not paired (84 and 84 steps) |  |  |  |  |  |  |  |  |

