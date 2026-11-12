## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12866.9 | 149.62 | 19.82 | 8504.7 | 84 |
| d | long | 2760 | 256 | 255 | 169 | 0.663 | 87 | 2.943 | 11237.1 | 129.16 | 22.69 | 26445.5 | 85 |
| d-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12869.6 | 149.65 | 19.81 | 8499.6 | 84 |
| d-2 | long | 2760 | 256 | 249 | 170 | 0.683 | 86 | 2.977 | 12072.5 | 140.38 | 21.12 | 26484.9 | 84 |
| e | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12811.9 | 148.98 | 19.90 | 8516.7 | 84 |
| e | long | 2760 | 256 | 249 | 171 | 0.687 | 85 | 3.012 | 12043.5 | 141.69 | 21.17 | 26477.0 | 83 |
| e-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12830.8 | 149.20 | 19.87 | 8518.5 | 84 |
| e-2 | long | 2760 | 256 | 267 | 165 | 0.618 | 91 | 2.813 | 12717.2 | 139.75 | 20.05 | 26499.0 | 89 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 108.48 (110.70; 81) | 116.83 (118.69; 82) | 112.65 |
| d-2 | 108.54 (110.94; 81) | 116.75 (118.23; 81) | 112.64 |
| e | 108.32 (110.73; 81) | 116.56 (118.88; 80) | 112.44 |
| e-2 | 108.51 (110.81; 81) | 116.76 (118.40; 86) | 112.64 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 91.91 (92.04; 81) | 99.22 (99.41; 82) | 95.56 |
| d-2 | 91.84 (92.00; 81) | 99.10 (99.29; 81) | 95.47 |
| e | 91.74 (91.90; 81) | 98.90 (99.08; 80) | 95.32 |
| e-2 | 91.67 (91.94; 81) | 99.02 (99.23; 86) | 95.34 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 88.23 (88.42; 81) | 95.37 (95.55; 82) | 91.80 |
| d-2 | 88.21 (88.37; 81) | 95.23 (95.40; 81) | 91.72 |
| e | 88.20 (88.38; 81) | 95.15 (95.31; 80) | 91.67 |
| e-2 | 88.15 (88.35; 81) | 95.28 (95.49; 86) | 91.71 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 3.56 (3.62; 81) | 3.80 (3.86; 82) | 3.68 |
| d-2 | 3.56 (3.63; 81) | 3.80 (3.89; 81) | 3.68 |
| e | 3.46 (3.52; 81) | 3.71 (3.77; 80) | 3.59 |
| e-2 | 3.46 (3.59; 81) | 3.70 (3.74; 86) | 3.58 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 16.30 (18.67; 81) | 17.56 (19.28; 82) | 16.93 |
| d-2 | 16.46 (18.94; 81) | 17.56 (18.94; 81) | 17.01 |
| e | 16.28 (18.83; 81) | 17.46 (19.80; 80) | 16.87 |
| e-2 | 16.36 (18.88; 81) | 17.56 (19.17; 86) | 16.96 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 11.00 (13.10; 81) | 12.30 (13.51; 82) | 11.65 |
| d-2 | 11.01 (13.11; 81) | 12.30 (13.09; 81) | 11.66 |
| e | 10.98 (13.09; 81) | 12.32 (14.37; 80) | 11.65 |
| e-2 | 10.97 (13.09; 81) | 12.29 (13.70; 86) | 11.63 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 1.82 (1.83; 81) | 1.83 (1.83; 82) | 1.83 |
| d-2 | 1.83 (1.83; 81) | 1.83 (1.83; 81) | 1.83 |
| e | 1.83 (1.86; 81) | 1.82 (1.83; 80) | 1.82 |
| e-2 | 1.83 (1.83; 81) | 1.83 (1.83; 86) | 1.83 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 1.37 (1.48; 81) | 1.39 (1.47; 82) | 1.38 |
| d-2 | 1.38 (1.57; 81) | 1.40 (1.46; 81) | 1.39 |
| e | 1.35 (1.46; 81) | 1.38 (1.52; 80) | 1.37 |
| e-2 | 1.36 (1.47; 81) | 1.39 (1.47; 86) | 1.38 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| d | 1.98 (2.26; 81) | 1.98 (2.47; 82) | 1.98 |
| d-2 | 2.04 (2.43; 81) | 1.93 (2.56; 81) | 1.98 |
| e | 1.99 (2.43; 81) | 1.91 (2.08; 80) | 1.95 |
| e-2 | 2.03 (2.48; 81) | 2.02 (2.17; 86) | 2.03 |

## Each arm against the reference loads

Reference: d and d-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| e | mean of the medians | 112.44 | -0.21 | 0.01 (112.65, 112.64) | gain |
| e | greedy, median | 108.32 | -0.19 | 0.06 (108.48, 108.54) | gain |
| e | long, median | 116.56 | -0.23 | 0.08 (116.83, 116.75) | gain |
| e-2 | mean of the medians | 112.64 | -0.01 | 0.01 (112.65, 112.64) | gain |
| e-2 | greedy, median | 108.51 | -0.00 | 0.06 (108.48, 108.54) | negligible |
| e-2 | long, median | 116.76 | -0.03 | 0.08 (116.83, 116.75) | negligible |

## The greedy request, step by step against d

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| d-2 | yes | 254 | 170 | 81 | +0.23 | 0.15 | +0.00 | +0.27 | -0.04 | +0.01 | +0.08 | +0.17 |
| e | yes | 254 | 170 | 81 | +0.03 | 0.15 | -0.20 | +0.17 | -0.14 | +0.03 | -0.02 | +0.17 |
| e-2 | yes | 254 | 170 | 81 | +0.11 | 0.16 | -0.19 | +0.21 | -0.10 | +0.00 | -0.01 | +0.22 |

