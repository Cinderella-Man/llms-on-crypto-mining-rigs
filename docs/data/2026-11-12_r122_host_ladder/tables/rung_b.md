## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13054.3 | 151.79 | 19.53 | 8530.0 | 84 |
| a | long | 2760 | 256 | 276 | 163 | 0.591 | 93 | 2.753 | 12133.3 | 130.47 | 21.02 | 26478.0 | 91 |
| a-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12993.0 | 151.08 | 19.63 | 8556.1 | 84 |
| a-2 | long | 2760 | 256 | 260 | 168 | 0.646 | 88 | 2.909 | 11566.3 | 131.44 | 22.05 | 26494.5 | 86 |
| b | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13018.5 | 151.38 | 19.59 | 8522.4 | 84 |
| b | long | 2760 | 256 | 231 | 178 | 0.771 | 78 | 3.282 | 10193.7 | 130.69 | 25.02 | 26482.8 | 76 |
| b-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12996.1 | 151.12 | 19.62 | 8548.4 | 84 |
| b-2 | long | 2760 | 256 | 263 | 167 | 0.635 | 89 | 2.876 | 11616.6 | 130.52 | 21.95 | 26467.4 | 87 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 110.57 (113.09; 81) | 118.95 (120.68; 88) | 114.76 |
| a-2 | 110.43 (112.79; 81) | 118.89 (120.84; 83) | 114.66 |
| b | 109.94 (112.51; 81) | 117.75 (118.93; 73) | 113.85 |
| b-2 | 110.13 (112.68; 81) | 118.22 (120.07; 84) | 114.18 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 92.87 (93.05; 81) | 100.04 (100.13; 88) | 96.46 |
| a-2 | 92.74 (92.92; 81) | 99.88 (99.99; 83) | 96.31 |
| b | 92.76 (92.96; 81) | 99.56 (99.83; 73) | 96.16 |
| b-2 | 92.79 (93.01; 81) | 99.90 (100.07; 84) | 96.34 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 88.32 (88.47; 81) | 95.29 (95.36; 88) | 91.80 |
| a-2 | 88.17 (88.36; 81) | 95.10 (95.24; 83) | 91.63 |
| b | 88.20 (88.41; 81) | 94.86 (95.08; 73) | 91.53 |
| b-2 | 88.33 (88.49; 81) | 95.18 (95.32; 84) | 91.76 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 4.54 (4.58; 81) | 4.76 (4.77; 88) | 4.65 |
| a-2 | 4.51 (4.55; 81) | 4.74 (4.75; 83) | 4.62 |
| b | 4.52 (4.54; 81) | 4.75 (4.75; 73) | 4.63 |
| b-2 | 4.50 (4.53; 81) | 4.72 (4.75; 84) | 4.61 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 17.18 (20.04; 81) | 18.80 (20.54; 88) | 17.99 |
| a-2 | 17.27 (19.87; 81) | 18.73 (20.85; 83) | 18.00 |
| b | 16.73 (19.56; 81) | 18.01 (19.10; 73) | 17.37 |
| b-2 | 16.93 (19.67; 81) | 18.25 (20.00; 84) | 17.59 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 11.06 (13.27; 81) | 12.42 (13.86; 88) | 11.74 |
| a-2 | 11.07 (13.28; 81) | 12.33 (14.17; 83) | 11.70 |
| b | 11.04 (13.27; 81) | 12.28 (13.02; 73) | 11.66 |
| b-2 | 11.09 (13.39; 81) | 12.41 (13.93; 84) | 11.75 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 1.81 (1.81; 81) | 1.81 (1.81; 88) | 1.81 |
| a-2 | 1.81 (1.82; 81) | 1.81 (1.82; 83) | 1.81 |
| b | 1.81 (1.81; 81) | 1.81 (1.81; 73) | 1.81 |
| b-2 | 1.81 (1.82; 81) | 1.81 (1.82; 84) | 1.81 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 1.55 (1.70; 81) | 1.59 (1.70; 88) | 1.57 |
| a-2 | 1.61 (1.72; 81) | 1.62 (1.70; 83) | 1.62 |
| b | 1.53 (1.67; 81) | 1.58 (1.60; 73) | 1.56 |
| b-2 | 1.62 (1.70; 81) | 1.62 (1.71; 84) | 1.62 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| a | 2.71 (3.26; 81) | 2.96 (3.16; 88) | 2.84 |
| a-2 | 2.72 (3.05; 81) | 2.91 (3.15; 83) | 2.82 |
| b | 2.32 (2.81; 81) | 2.34 (2.66; 73) | 2.33 |
| b-2 | 2.35 (2.76; 81) | 2.36 (2.55; 84) | 2.35 |

## Each arm against the reference loads

Reference: a and a-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| b | mean of the medians | 113.85 | -0.86 | 0.10 (114.76, 114.66) | gain |
| b | greedy, median | 109.94 | -0.55 | 0.14 (110.57, 110.43) | gain |
| b | long, median | 117.75 | -1.17 | 0.06 (118.95, 118.89) | gain |
| b-2 | mean of the medians | 114.18 | -0.53 | 0.10 (114.76, 114.66) | gain |
| b-2 | greedy, median | 110.13 | -0.36 | 0.14 (110.57, 110.43) | gain |
| b-2 | long, median | 118.22 | -0.70 | 0.06 (118.95, 118.89) | gain |

## The greedy request, step by step against a

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| a-2 | yes | 254 | 170 | 81 | -0.30 | 0.15 | -0.07 | -0.17 | -0.13 | +0.01 | +0.02 | -0.21 |
| b | yes | 254 | 170 | 81 | -0.58 | 0.22 | -0.51 | -0.49 | -0.09 | +0.00 | -0.03 | -0.45 |
| b-2 | yes | 254 | 170 | 81 | -0.41 | 0.15 | -0.32 | -0.37 | -0.04 | +0.00 | +0.00 | -0.50 |

