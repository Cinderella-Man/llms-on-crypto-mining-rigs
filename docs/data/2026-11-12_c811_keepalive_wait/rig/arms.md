## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f0 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10829.0 | 125.92 | 23.55 | 7673.0 | 84 |
| f0 | long | 2760 | 256 | 255 | 169 | 0.663 | 87 | 2.943 | 12277.9 | 141.13 | 20.77 | 26460.0 | 85 |
| f1 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10091.0 | 117.34 | 25.27 | 4726.8 | 83 |
| f1 | long | 2760 | 256 | 244 | 172 | 0.705 | 84 | 3.048 | 10249.1 | 122.01 | 24.88 | 28311.1 | 81 |
| f0-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10835.6 | 126.00 | 23.53 | 7505.4 | 84 |
| f0-2 | long | 2760 | 256 | 249 | 171 | 0.687 | 85 | 3.012 | 11890.7 | 139.89 | 21.45 | 26447.5 | 83 |
| f1-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10108.4 | 117.54 | 25.23 | 4771.3 | 83 |
| f1-2 | long | 2760 | 256 | 265 | 166 | 0.626 | 90 | 2.844 | 10799.4 | 119.99 | 23.61 | 28267.0 | 87 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 107.94 (110.02; 80) | 116.67 (118.71; 82) | 112.31 |
| f1 | 107.90 (110.27; 78) | 116.36 (117.79; 77) | 112.13 |
| f0-2 | 107.96 (110.07; 80) | 115.86 (116.98; 80) | 111.91 |
| f1-2 | 108.00 (110.35; 78) | 116.47 (117.39; 83) | 112.23 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 91.62 (91.79; 80) | 99.18 (99.53; 82) | 95.40 |
| f1 | 91.72 (91.72; 78) | 99.17 (99.16; 77) | 95.45 |
| f0-2 | 91.67 (91.85; 80) | 98.83 (99.28; 80) | 95.25 |
| f1-2 | 91.72 (91.74; 78) | 99.13 (99.20; 83) | 95.43 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 88.13 (88.28; 80) | 95.46 (95.77; 82) | 91.80 |
| f1 | 88.24 (88.23; 78) | 95.45 (95.45; 77) | 91.84 |
| f0-2 | 88.16 (88.35; 80) | 95.13 (95.57; 80) | 91.65 |
| f1-2 | 88.21 (88.25; 78) | 95.45 (95.50; 83) | 91.83 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 3.44 (3.51; 80) | 3.71 (3.76; 82) | 3.58 |
| f1 | 3.45 (3.49; 78) | 3.69 (3.71; 77) | 3.57 |
| f0-2 | 3.45 (3.50; 80) | 3.68 (3.72; 80) | 3.57 |
| f1-2 | 3.45 (3.49; 78) | 3.70 (3.70; 83) | 3.57 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 16.02 (18.24; 80) | 17.35 (19.18; 82) | 16.69 |
| f1 | 15.99 (18.55; 78) | 17.25 (18.64; 77) | 16.62 |
| f0-2 | 15.94 (18.22; 80) | 17.12 (17.70; 80) | 16.53 |
| f1-2 | 15.99 (18.62; 78) | 17.29 (18.19; 83) | 16.64 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 10.81 (12.97; 80) | 12.30 (13.79; 82) | 11.55 |
| f1 | 10.84 (13.37; 78) | 12.29 (13.35; 77) | 11.56 |
| f0-2 | 10.82 (12.98; 80) | 12.23 (12.58; 80) | 11.52 |
| f1-2 | 10.84 (13.37; 78) | 12.30 (13.06; 83) | 11.57 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 4.21 (3.85; 80) | 4.49 (4.23; 82) | 4.35 |
| f1 | 4.23 (3.83; 78) | 4.51 (4.36; 77) | 4.37 |
| f0-2 | 4.22 (3.82; 80) | 4.51 (4.43; 80) | 4.37 |
| f1-2 | 4.23 (3.80; 78) | 4.54 (4.41; 83) | 4.39 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | -1.06 (-0.50; 80) | -1.32 (-0.96; 82) | -1.19 |
| f1 | -1.06 (-0.52; 78) | -1.34 (-1.11; 77) | -1.20 |
| f0-2 | -1.08 (-0.53; 80) | -1.35 (-1.24; 80) | -1.21 |
| f1-2 | -1.06 (-0.46; 78) | -1.35 (-1.18; 83) | -1.20 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 1.86 (1.91; 80) | 1.85 (2.12; 82) | 1.86 |
| f1 | 1.80 (1.87; 78) | 1.74 (2.03; 77) | 1.77 |
| f0-2 | 1.86 (1.95; 80) | 1.51 (1.93; 80) | 1.69 |
| f1-2 | 1.80 (1.91; 78) | 1.79 (1.90; 83) | 1.79 |

## Each arm against the reference loads

Reference: f0 and f0-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| f1 | mean of the medians | 112.13 | +0.02 | 0.40 (112.31, 111.91) | negligible |
| f1 | greedy, median | 107.90 | -0.04 | 0.02 (107.94, 107.96) | gain |
| f1 | long, median | 116.36 | +0.09 | 0.81 (116.67, 115.86) | negligible |
| f1-2 | mean of the medians | 112.23 | +0.12 | 0.40 (112.31, 111.91) | negligible |
| f1-2 | greedy, median | 108.00 | +0.05 | 0.02 (107.94, 107.96) | loss |
| f1-2 | long, median | 116.47 | +0.20 | 0.81 (116.67, 115.86) | negligible |

## The greedy request, step by step against f0

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f0 | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| f1 | yes | 254 | 170 | not paired (84 and 83 steps) |  |  |  |  |  |  |  |  |
| f0-2 | yes | 254 | 170 | 80 | +0.05 | 0.09 | +0.04 | -0.01 | +0.06 | -0.03 | -0.03 | +0.04 |
| f1-2 | yes | 254 | 170 | not paired (84 and 83 steps) |  |  |  |  |  |  |  |  |

