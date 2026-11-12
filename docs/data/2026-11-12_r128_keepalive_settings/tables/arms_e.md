## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12811.9 | 148.98 | 19.90 | 8516.7 | 84 |
| e | long | 2760 | 256 | 249 | 171 | 0.687 | 85 | 3.012 | 12043.5 | 141.69 | 21.17 | 26477.0 | 83 |
| e-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12830.8 | 149.20 | 19.87 | 8518.5 | 84 |
| e-2 | long | 2760 | 256 | 267 | 165 | 0.618 | 91 | 2.813 | 12717.2 | 139.75 | 20.05 | 26499.0 | 89 |
| e771 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10869.2 | 126.39 | 23.46 | 7549.3 | 84 |
| e771 | long | 2760 | 256 | 256 | 169 | 0.660 | 87 | 2.943 | 11217.2 | 128.93 | 22.73 | 26306.0 | 85 |
| e744 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12835.2 | 149.25 | 19.87 | 8494.2 | 84 |
| e744 | long | 2760 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 10985.0 | 127.73 | 23.21 | 26388.2 | 84 |
| e742 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12801.8 | 148.86 | 19.92 | 8537.6 | 84 |
| e742 | long | 2760 | 256 | 251 | 171 | 0.681 | 85 | 3.012 | 10946.2 | 128.78 | 23.30 | 26302.9 | 83 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 108.32 (110.73; 81) | 116.56 (118.88; 80) | 112.44 |
| e-2 | 108.51 (110.81; 81) | 116.76 (118.40; 86) | 112.64 |
| e771 | 108.10 (110.46; 81) | 116.52 (118.06; 82) | 112.31 |
| e744 | 108.13 (110.26; 80) | 116.26 (116.92; 81) | 112.20 |
| e742 | 107.89 (110.18; 81) | 116.21 (117.57; 80) | 112.05 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 91.74 (91.90; 81) | 98.90 (99.08; 80) | 95.32 |
| e-2 | 91.67 (91.94; 81) | 99.02 (99.23; 86) | 95.34 |
| e771 | 91.64 (91.77; 81) | 98.83 (99.15; 82) | 95.24 |
| e744 | 91.66 (91.75; 80) | 98.77 (98.89; 81) | 95.21 |
| e742 | 91.55 (91.72; 81) | 98.78 (99.07; 80) | 95.16 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 88.20 (88.38; 81) | 95.15 (95.31; 80) | 91.67 |
| e-2 | 88.15 (88.35; 81) | 95.28 (95.49; 86) | 91.71 |
| e771 | 88.14 (88.26; 81) | 95.11 (95.42; 82) | 91.62 |
| e744 | 88.14 (88.23; 80) | 95.08 (95.17; 81) | 91.61 |
| e742 | 88.05 (88.21; 81) | 95.02 (95.32; 80) | 91.53 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 3.46 (3.52; 81) | 3.71 (3.77; 80) | 3.59 |
| e-2 | 3.46 (3.59; 81) | 3.70 (3.74; 86) | 3.58 |
| e771 | 3.45 (3.50; 81) | 3.69 (3.73; 82) | 3.57 |
| e744 | 3.46 (3.52; 80) | 3.71 (3.73; 81) | 3.58 |
| e742 | 3.46 (3.52; 81) | 3.71 (3.75; 80) | 3.59 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 16.28 (18.83; 81) | 17.46 (19.80; 80) | 16.87 |
| e-2 | 16.36 (18.88; 81) | 17.56 (19.17; 86) | 16.96 |
| e771 | 16.14 (18.69; 81) | 17.43 (18.92; 82) | 16.79 |
| e744 | 16.27 (18.51; 80) | 17.49 (18.03; 81) | 16.88 |
| e742 | 16.12 (18.46; 81) | 17.29 (18.51; 80) | 16.71 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 10.98 (13.09; 81) | 12.32 (14.37; 80) | 11.65 |
| e-2 | 10.97 (13.09; 81) | 12.29 (13.70; 86) | 11.63 |
| e771 | 10.84 (12.97; 81) | 12.24 (13.55; 82) | 11.54 |
| e744 | 10.99 (13.13; 80) | 12.32 (12.66; 81) | 11.66 |
| e742 | 11.02 (13.11; 81) | 12.29 (13.22; 80) | 11.66 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.83 (1.86; 81) | 1.82 (1.83; 80) | 1.82 |
| e-2 | 1.83 (1.83; 81) | 1.83 (1.83; 86) | 1.83 |
| e771 | 4.21 (3.79; 81) | 4.48 (4.23; 82) | 4.34 |
| e744 | 1.82 (1.83; 80) | 1.83 (1.83; 81) | 1.83 |
| e742 | 1.83 (1.83; 81) | 1.83 (1.83; 80) | 1.83 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.35 (1.46; 81) | 1.38 (1.52; 80) | 1.37 |
| e-2 | 1.36 (1.47; 81) | 1.39 (1.47; 86) | 1.38 |
| e771 | -1.03 (-0.46; 81) | -1.31 (-0.96; 82) | -1.17 |
| e744 | 1.37 (1.48; 80) | 1.38 (1.45; 81) | 1.38 |
| e742 | 1.38 (1.48; 81) | 1.39 (1.46; 80) | 1.38 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| e | 1.99 (2.43; 81) | 1.91 (2.08; 80) | 1.95 |
| e-2 | 2.03 (2.48; 81) | 2.02 (2.17; 86) | 2.03 |
| e771 | 2.00 (2.39; 81) | 2.01 (2.10; 82) | 2.01 |
| e744 | 2.02 (2.07; 80) | 1.98 (2.08; 81) | 2.00 |
| e742 | 1.78 (2.04; 81) | 1.77 (1.99; 80) | 1.77 |

## Each arm against the reference loads

Reference: e and e-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| e771 | mean of the medians | 112.31 | -0.23 | 0.20 (112.44, 112.64) | gain |
| e771 | greedy, median | 108.10 | -0.32 | 0.19 (108.32, 108.51) | gain |
| e771 | long, median | 116.52 | -0.14 | 0.20 (116.56, 116.76) | negligible |
| e744 | mean of the medians | 112.20 | -0.34 | 0.20 (112.44, 112.64) | gain |
| e744 | greedy, median | 108.13 | -0.28 | 0.19 (108.32, 108.51) | gain |
| e744 | long, median | 116.26 | -0.40 | 0.20 (116.56, 116.76) | gain |
| e742 | mean of the medians | 112.05 | -0.48 | 0.20 (112.44, 112.64) | gain |
| e742 | greedy, median | 107.89 | -0.52 | 0.19 (108.32, 108.51) | gain |
| e742 | long, median | 116.21 | -0.45 | 0.20 (116.56, 116.76) | gain |

## The greedy request, step by step against e

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| e-2 | yes | 254 | 170 | 81 | +0.08 | 0.22 | +0.00 | +0.04 | +0.04 | -0.03 | +0.01 | +0.06 |
| e771 | yes | 254 | 170 | not paired (84 and 84 steps) |  |  |  |  |  |  |  |  |
| e744 | yes | 254 | 170 | 80 | -0.31 | 0.15 | -0.13 | -0.17 | -0.14 | -0.03 | +0.02 | -0.17 |
| e742 | yes | 254 | 170 | 81 | -0.55 | 0.15 | -0.33 | -0.37 | -0.18 | -0.03 | +0.02 | -0.38 |

