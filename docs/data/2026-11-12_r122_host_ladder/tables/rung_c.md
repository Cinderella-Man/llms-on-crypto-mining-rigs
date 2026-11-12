## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13018.5 | 151.38 | 19.59 | 8522.4 | 84 |
| b | long | 2760 | 256 | 231 | 178 | 0.771 | 78 | 3.282 | 10193.7 | 130.69 | 25.02 | 26482.8 | 76 |
| b-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12996.1 | 151.12 | 19.62 | 8548.4 | 84 |
| b-2 | long | 2760 | 256 | 263 | 167 | 0.635 | 89 | 2.876 | 11616.6 | 130.52 | 21.95 | 26467.4 | 87 |
| c | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12908.7 | 150.10 | 19.75 | 8522.4 | 84 |
| c | long | 2760 | 256 | 258 | 169 | 0.655 | 87 | 2.943 | 11223.6 | 129.01 | 22.72 | 26437.9 | 85 |
| c-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12891.2 | 149.90 | 19.78 | 8524.2 | 84 |
| c-2 | long | 2760 | 256 | 249 | 172 | 0.691 | 84 | 3.048 | 10834.7 | 128.98 | 23.54 | 26386.4 | 82 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 109.94 (112.51; 81) | 117.75 (118.93; 73) | 113.85 |
| b-2 | 110.13 (112.68; 81) | 118.22 (120.07; 84) | 114.18 |
| c | 108.72 (111.19; 81) | 117.05 (118.32; 82) | 112.89 |
| c-2 | 108.97 (111.17; 81) | 117.19 (118.05; 79) | 113.08 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 92.76 (92.96; 81) | 99.56 (99.83; 73) | 96.16 |
| b-2 | 92.79 (93.01; 81) | 99.90 (100.07; 84) | 96.34 |
| c | 91.94 (92.12; 81) | 99.00 (99.27; 82) | 95.47 |
| c-2 | 91.83 (91.98; 81) | 99.17 (99.10; 79) | 95.50 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 88.20 (88.41; 81) | 94.86 (95.08; 73) | 91.53 |
| b-2 | 88.33 (88.49; 81) | 95.18 (95.32; 84) | 91.76 |
| c | 88.26 (88.42; 81) | 95.12 (95.42; 82) | 91.69 |
| c-2 | 88.16 (88.34; 81) | 95.32 (95.24; 79) | 91.74 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 4.52 (4.54; 81) | 4.75 (4.75; 73) | 4.63 |
| b-2 | 4.50 (4.53; 81) | 4.72 (4.75; 84) | 4.61 |
| c | 3.56 (3.70; 81) | 3.80 (3.85; 82) | 3.68 |
| c-2 | 3.55 (3.63; 81) | 3.80 (3.86; 79) | 3.67 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 16.73 (19.56; 81) | 18.01 (19.10; 73) | 17.37 |
| b-2 | 16.93 (19.67; 81) | 18.25 (20.00; 84) | 17.59 |
| c | 16.56 (19.07; 81) | 17.82 (19.05; 82) | 17.19 |
| c-2 | 16.70 (19.20; 81) | 17.89 (18.95; 79) | 17.30 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 11.04 (13.27; 81) | 12.28 (13.02; 73) | 11.66 |
| b-2 | 11.09 (13.39; 81) | 12.41 (13.93; 84) | 11.75 |
| c | 10.98 (13.10; 81) | 12.28 (13.23; 82) | 11.63 |
| c-2 | 11.01 (13.12; 81) | 12.32 (13.23; 79) | 11.66 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 1.81 (1.81; 81) | 1.81 (1.81; 73) | 1.81 |
| b-2 | 1.81 (1.82; 81) | 1.81 (1.82; 84) | 1.81 |
| c | 1.82 (1.82; 81) | 1.82 (1.83; 82) | 1.82 |
| c-2 | 1.83 (1.84; 81) | 1.83 (1.83; 79) | 1.83 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 1.53 (1.67; 81) | 1.58 (1.60; 73) | 1.56 |
| b-2 | 1.62 (1.70; 81) | 1.62 (1.71; 84) | 1.62 |
| c | 1.34 (1.49; 81) | 1.39 (1.50; 82) | 1.37 |
| c-2 | 1.39 (1.49; 81) | 1.40 (1.46; 79) | 1.40 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| b | 2.32 (2.81; 81) | 2.34 (2.66; 73) | 2.33 |
| b-2 | 2.35 (2.76; 81) | 2.36 (2.55; 84) | 2.35 |
| c | 2.33 (2.66; 81) | 2.35 (2.51; 82) | 2.34 |
| c-2 | 2.40 (2.75; 81) | 2.35 (2.43; 79) | 2.37 |

## Each arm against the reference loads

Reference: b and b-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| c | mean of the medians | 112.89 | -1.13 | 0.33 (113.85, 114.18) | gain |
| c | greedy, median | 108.72 | -1.32 | 0.19 (109.94, 110.13) | gain |
| c | long, median | 117.05 | -0.93 | 0.47 (117.75, 118.22) | gain |
| c-2 | mean of the medians | 113.08 | -0.94 | 0.33 (113.85, 114.18) | gain |
| c-2 | greedy, median | 108.97 | -1.07 | 0.19 (109.94, 110.13) | gain |
| c-2 | long, median | 117.19 | -0.80 | 0.47 (117.75, 118.22) | gain |

## The greedy request, step by step against b

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| b-2 | yes | 254 | 170 | 81 | +0.17 | 0.22 | +0.24 | +0.11 | +0.06 | +0.00 | +0.03 | -0.05 |
| c | yes | 254 | 170 | 81 | -1.32 | 0.19 | -1.18 | -0.49 | -0.83 | +0.01 | -0.18 | -0.14 |
| c-2 | yes | 254 | 170 | 81 | -1.34 | 0.20 | -1.08 | -0.36 | -0.98 | +0.03 | -0.18 | -0.06 |

