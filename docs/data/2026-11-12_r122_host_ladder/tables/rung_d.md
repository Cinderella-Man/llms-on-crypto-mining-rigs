## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12908.7 | 150.10 | 19.75 | 8522.4 | 84 |
| c | long | 2760 | 256 | 258 | 169 | 0.655 | 87 | 2.943 | 11223.6 | 129.01 | 22.72 | 26437.9 | 85 |
| c-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12891.2 | 149.90 | 19.78 | 8524.2 | 84 |
| c-2 | long | 2760 | 256 | 249 | 172 | 0.691 | 84 | 3.048 | 10834.7 | 128.98 | 23.54 | 26386.4 | 82 |
| d | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12866.9 | 149.62 | 19.82 | 8504.7 | 84 |
| d | long | 2760 | 256 | 255 | 169 | 0.663 | 87 | 2.943 | 11237.1 | 129.16 | 22.69 | 26445.5 | 85 |
| d-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12869.6 | 149.65 | 19.81 | 8499.6 | 84 |
| d-2 | long | 2760 | 256 | 249 | 170 | 0.683 | 86 | 2.977 | 12072.5 | 140.38 | 21.12 | 26484.9 | 84 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 108.72 (111.19; 81) | 117.05 (118.32; 82) | 112.89 |
| c-2 | 108.97 (111.17; 81) | 117.19 (118.05; 79) | 113.08 |
| d | 108.48 (110.70; 81) | 116.83 (118.69; 82) | 112.65 |
| d-2 | 108.54 (110.94; 81) | 116.75 (118.23; 81) | 112.64 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 91.94 (92.12; 81) | 99.00 (99.27; 82) | 95.47 |
| c-2 | 91.83 (91.98; 81) | 99.17 (99.10; 79) | 95.50 |
| d | 91.91 (92.04; 81) | 99.22 (99.41; 82) | 95.56 |
| d-2 | 91.84 (92.00; 81) | 99.10 (99.29; 81) | 95.47 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 88.26 (88.42; 81) | 95.12 (95.42; 82) | 91.69 |
| c-2 | 88.16 (88.34; 81) | 95.32 (95.24; 79) | 91.74 |
| d | 88.23 (88.42; 81) | 95.37 (95.55; 82) | 91.80 |
| d-2 | 88.21 (88.37; 81) | 95.23 (95.40; 81) | 91.72 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 3.56 (3.70; 81) | 3.80 (3.85; 82) | 3.68 |
| c-2 | 3.55 (3.63; 81) | 3.80 (3.86; 79) | 3.67 |
| d | 3.56 (3.62; 81) | 3.80 (3.86; 82) | 3.68 |
| d-2 | 3.56 (3.63; 81) | 3.80 (3.89; 81) | 3.68 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 16.56 (19.07; 81) | 17.82 (19.05; 82) | 17.19 |
| c-2 | 16.70 (19.20; 81) | 17.89 (18.95; 79) | 17.30 |
| d | 16.30 (18.67; 81) | 17.56 (19.28; 82) | 16.93 |
| d-2 | 16.46 (18.94; 81) | 17.56 (18.94; 81) | 17.01 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 10.98 (13.10; 81) | 12.28 (13.23; 82) | 11.63 |
| c-2 | 11.01 (13.12; 81) | 12.32 (13.23; 79) | 11.66 |
| d | 11.00 (13.10; 81) | 12.30 (13.51; 82) | 11.65 |
| d-2 | 11.01 (13.11; 81) | 12.30 (13.09; 81) | 11.66 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 1.82 (1.82; 81) | 1.82 (1.83; 82) | 1.82 |
| c-2 | 1.83 (1.84; 81) | 1.83 (1.83; 79) | 1.83 |
| d | 1.82 (1.83; 81) | 1.83 (1.83; 82) | 1.83 |
| d-2 | 1.83 (1.83; 81) | 1.83 (1.83; 81) | 1.83 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 1.34 (1.49; 81) | 1.39 (1.50; 82) | 1.37 |
| c-2 | 1.39 (1.49; 81) | 1.40 (1.46; 79) | 1.40 |
| d | 1.37 (1.48; 81) | 1.39 (1.47; 82) | 1.38 |
| d-2 | 1.38 (1.57; 81) | 1.40 (1.46; 81) | 1.39 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| c | 2.33 (2.66; 81) | 2.35 (2.51; 82) | 2.34 |
| c-2 | 2.40 (2.75; 81) | 2.35 (2.43; 79) | 2.37 |
| d | 1.98 (2.26; 81) | 1.98 (2.47; 82) | 1.98 |
| d-2 | 2.04 (2.43; 81) | 1.93 (2.56; 81) | 1.98 |

## Each arm against the reference loads

Reference: c and c-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| d | mean of the medians | 112.65 | -0.33 | 0.19 (112.89, 113.08) | gain |
| d | greedy, median | 108.48 | -0.36 | 0.25 (108.72, 108.97) | gain |
| d | long, median | 116.83 | -0.29 | 0.13 (117.05, 117.19) | gain |
| d-2 | mean of the medians | 112.64 | -0.34 | 0.19 (112.89, 113.08) | gain |
| d-2 | greedy, median | 108.54 | -0.30 | 0.25 (108.72, 108.97) | gain |
| d-2 | long, median | 116.75 | -0.37 | 0.13 (117.05, 117.19) | gain |

## The greedy request, step by step against c

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| c-2 | yes | 254 | 170 | 81 | -0.01 | 0.10 | +0.01 | +0.13 | -0.14 | +0.02 | +0.00 | +0.08 |
| d | yes | 254 | 170 | 81 | -0.49 | 0.08 | -0.49 | -0.40 | -0.08 | +0.01 | -0.00 | -0.40 |
| d-2 | yes | 254 | 170 | 81 | -0.25 | 0.15 | -0.45 | -0.13 | -0.12 | +0.01 | +0.08 | -0.24 |

