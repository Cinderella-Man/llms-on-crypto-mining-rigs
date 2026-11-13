## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f0 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10811.5 | 125.71 | 23.59 | 7742.1 | 84 |
| f0 | long | 2760 | 75 | 66 | 54 | 0.818 | 21 | 3.571 | 3812.1 | 181.53 | 19.41 | 26637.9 | 21 |
| f1 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10019.3 | 116.50 | 25.45 | 4054.6 | 83 |
| f1 | long | 2760 | 256 | 240 | 174 | 0.725 | 82 | 3.122 | 9813.1 | 119.67 | 25.99 | 24712.6 | 79 |
| f0-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10739.6 | 124.88 | 23.74 | 7467.7 | 84 |
| f0-2 | long | 2760 | 256 | 232 | 177 | 0.763 | 79 | 3.241 | 10177.7 | 128.83 | 25.05 | 26065.2 | 77 |
| f1-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10096.0 | 117.40 | 25.26 | 4009.3 | 83 |
| f1-2 | long | 2760 | 256 | 250 | 171 | 0.684 | 85 | 3.012 | 10392.2 | 122.26 | 24.54 | 24744.1 | 82 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 107.72 (109.86; 80) | 116.18 (119.37; 18) | 111.95 |
| f1 | 107.47 (109.81; 78) | 115.80 (117.15; 75) | 111.63 |
| f0-2 | 107.33 (109.71; 80) | 115.41 (117.07; 74) | 111.37 |
| f1-2 | 107.34 (109.82; 78) | 116.16 (118.38; 77) | 111.75 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 91.34 (91.51; 80) | 98.47 (99.64; 18) | 94.90 |
| f1 | 91.41 (91.35; 78) | 98.67 (98.62; 75) | 95.04 |
| f0-2 | 91.39 (91.50; 80) | 98.28 (98.52; 74) | 94.84 |
| f1-2 | 91.29 (91.32; 78) | 98.53 (98.33; 77) | 94.91 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 87.78 (87.99; 80) | 94.75 (95.84; 18) | 91.26 |
| f1 | 87.85 (87.84; 78) | 94.94 (94.89; 75) | 91.40 |
| f0-2 | 87.81 (87.99; 80) | 94.51 (94.78; 74) | 91.16 |
| f1-2 | 87.72 (87.80; 78) | 94.73 (94.58; 77) | 91.23 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 3.45 (3.51; 80) | 3.70 (3.80; 18) | 3.58 |
| f1 | 3.46 (3.51; 78) | 3.71 (3.73; 75) | 3.58 |
| f0-2 | 3.45 (3.51; 80) | 3.70 (3.75; 74) | 3.58 |
| f1-2 | 3.45 (3.52; 78) | 3.71 (3.75; 77) | 3.58 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 16.03 (18.36; 80) | 17.15 (19.73; 18) | 16.59 |
| f1 | 15.84 (18.46; 78) | 17.21 (18.53; 75) | 16.52 |
| f0-2 | 15.77 (18.20; 80) | 17.07 (18.55; 74) | 16.42 |
| f1-2 | 15.85 (18.50; 78) | 17.15 (20.05; 77) | 16.50 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 10.78 (12.94; 80) | 12.20 (13.76; 18) | 11.49 |
| f1 | 10.80 (13.33; 78) | 12.24 (13.43; 75) | 11.52 |
| f0-2 | 10.78 (12.94; 80) | 12.21 (13.53; 74) | 11.49 |
| f1-2 | 10.79 (13.33; 78) | 12.16 (14.77; 77) | 11.47 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 4.22 (3.83; 80) | 4.29 (4.10; 18) | 4.26 |
| f1 | 4.22 (3.82; 78) | 4.50 (4.31; 75) | 4.36 |
| f0-2 | 4.20 (3.81; 80) | 4.47 (4.18; 74) | 4.33 |
| f1-2 | 4.21 (3.81; 78) | 4.45 (4.01; 77) | 4.33 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | -1.09 (-0.54; 80) | -1.12 (-0.75; 18) | -1.10 |
| f1 | -1.06 (-0.52; 78) | -1.30 (-1.05; 75) | -1.18 |
| f0-2 | -1.03 (-0.51; 80) | -1.29 (-0.93; 74) | -1.16 |
| f1-2 | -1.02 (-0.48; 78) | -1.26 (-0.68; 77) | -1.14 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f0 | 1.89 (2.13; 80) | 1.83 (2.63; 18) | 1.86 |
| f1 | 1.73 (1.83; 78) | 1.73 (1.85; 75) | 1.73 |
| f0-2 | 1.72 (1.97; 80) | 1.64 (1.76; 74) | 1.68 |
| f1-2 | 1.75 (1.83; 78) | 1.65 (1.95; 77) | 1.70 |

## Each arm against the reference loads

Reference: f0 and f0-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| f1 | mean of the medians | 111.63 | -0.02 | 0.58 (111.95, 111.37) | negligible |
| f1 | greedy, median | 107.47 | -0.06 | 0.39 (107.72, 107.33) | negligible |
| f1 | long, median | 115.80 | +0.01 | 0.77 (116.18, 115.41) | negligible |
| f1-2 | mean of the medians | 111.75 | +0.09 | 0.58 (111.95, 111.37) | negligible |
| f1-2 | greedy, median | 107.34 | -0.19 | 0.39 (107.72, 107.33) | negligible |
| f1-2 | long, median | 116.16 | +0.37 | 0.77 (116.18, 115.41) | negligible |

## The greedy request, step by step against f0

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f0 | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| f1 | yes | 254 | 170 | not paired (84 and 83 steps) |  |  |  |  |  |  |  |  |
| f0-2 | yes | 254 | 170 | 80 | -0.15 | 0.14 | -0.18 | -0.15 | -0.00 | -0.02 | +0.02 | -0.15 |
| f1-2 | yes | 254 | 170 | not paired (84 and 83 steps) |  |  |  |  |  |  |  |  |

