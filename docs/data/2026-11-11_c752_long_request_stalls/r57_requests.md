# Arms v4night, v3rel: generation time per request against one fit of the arms

Fit on 108 of 124 requests with 10 steps or more: generation ms = -54.2 + steps x (128.90 + 3.418 x depth / 1000); residual median 0.3 ms, scaled MAD 134.6 ms, outlier cut 404.2 ms.

| arm | tasks | requests | steps | median ms a step | requests above the cut | their excess, s | excess over all requests, s | excess a task, s | as 117 ms waits | swap-out, MiB | swap-in in generation, MiB | iowait in generation, % (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4night | 12 | 65 | 9943 | 144.9 | 5 | 5.33 | 4.12 | 0.34 | 35 | 864 | 24.1 | 0.93 |
| v3rel | 12 | 59 | 4944 | 145.7 | 10 | 9.63 | 11.12 | 0.93 | 95 | 985 | 22.6 | 1.15 |

## The requests above the cut

| arm | task | turn | depth at start | steps | ms a step | excess, ms | as 117 ms waits | checkpoints saved, ms | MemAvailable start, min | swap at start, at the end | swap-out in the prompt, MiB | swap-in in generation, MiB | iowait, % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| v4night | bugfix-javascript-space-age | 1 | 2352 | 19 | 172.8 | 734 | 6.3 | 3, 1929 | 254, 305 | 43, 106 | 54.8 | 0.00 | 2.7 |
| v4night | bugfix-javascript-space-age | 3 | 4122 | 54 | 155.2 | 698 | 6.0 | 2, 1280 | 333, 324 | 121, 128 | 4.2 | 0.02 | 4.0 |
| v4night | bugfix-python-variable-length-quantity | 1 | 2364 | 22 | 153.6 | 416 | 3.6 | 3, 1928 | 516, 524 | 324, 340 | 7.1 | 0.37 | 6.0 |
| v4night | bugfix-javascript-phone-number | 4 | 4894 | 976 | 152.8 | 2385 | 20.4 | 2, 1314 | 666, 637 | 433, 540 | 4.8 | 1.07 | 1.5 |
| v4night | bugfix-python-scale-generator | 2 | 4283 | 491 | 148.1 | 1100 | 9.4 | 2, 1304 | 942, 941 | 770, 843 | 11.6 | 4.64 | 1.8 |
| v3rel | bugfix-python-affine-cipher | 1 | 2780 | 19 | 206.9 | 1353 | 11.6 | 3, 1936 | 819, 173 | 10, 165 | 117.3 | 0.81 | 2.5 |
| v3rel | bugfix-python-affine-cipher | 2 | 4033 | 644 | 148.3 | 1544 | 13.2 | 2, 1285 | 192, 205 | 165, 350 | 113.3 | 2.05 | 1.4 |
| v3rel | bugfix-javascript-space-age | 1 | 2352 | 15 | 166.3 | 494 | 4.2 | 3, 1933 | 375, 350 | 411, 447 | 85.2 | 0.46 | 5.0 |
| v3rel | bugfix-python-phone-number | 1 | 2679 | 27 | 168.7 | 877 | 7.5 | 3, 1934 | 447, 484 | 516, 550 | 24.9 | 0.46 | 4.5 |
| v3rel | bugfix-python-phone-number | 5 | 6005 | 160 | 153.5 | 583 | 5.0 | 2, 1283 | 644, 556 | 612, 612 | 0.0 | 0.89 | 1.0 |
| v3rel | bugfix-python-variable-length-quantity | 5 | 6450 | 95 | 163.1 | 1159 | 9.9 | 2, 1291 | 722, 682 | 758, 806 | 0.0 | 0.18 | 4.0 |
| v3rel | bugfix-python-list-ops | 2 | 3569 | 433 | 146.9 | 1612 | 13.8 | 2, 1299 | 850, 765 | 819, 933 | 40.4 | 1.88 | 1.8 |
| v3rel | bugfix-python-scale-generator | 2 | 4280 | 384 | 147.4 | 753 | 6.4 | 2, 1286 | 930, 882 | 1004, 1004 | 0.0 | 0.80 | 1.0 |
| v3rel | bugfix-javascript-complex-numbers | 4 | 4424 | 108 | 147.9 | 429 | 3.7 | 2, 1308 | 916, 903 | 1004, 1004 | 0.0 | 0.21 | 1.0 |
| v3rel | bugfix-python-pig-latin | 3 | 3460 | 409 | 144.6 | 824 | 7.0 | 2, 1285 | 898, 898 | 1004, 1004 | 0.0 | 0.68 | 1.0 |

## Each arm's first request (the agent's first turn, the first request after the load)

| arm | task | prompt | steps | ms a step | excess, ms | checkpoints, save ms | MemAvailable start, min | swap start, end | swap-in in generation, MiB | iowait, % |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| v4night | bugfix-python-affine-cipher | 2780 | 19 | 152.2 | 314 | 3, 1895 | 919, 229 | 0, 1 | 0.00 | 3.0 |
| v3rel | bugfix-python-affine-cipher | 2780 | 19 | 206.9 | 1353 | 3, 1936 | 819, 173 | 10, 165 | 0.81 | 2.5 |

## Excess against swap traffic

Requests with 10 steps or more, split by the swap-in during their generation.

| arm | swap-in in generation | requests | median excess, ms | mean excess, ms | above the cut |
|---|---|---:|---:|---:|---:|
| v4night | none (< 0.1 MiB) | 27 | -9 | 42 | 2 |
| v4night | 0.1 to 1 MiB | 32 | -61 | -13 | 1 |
| v4night | 1 MiB or more | 6 | 71 | 568 | 2 |
| v3rel | none (< 0.1 MiB) | 16 | 43 | 59 | 0 |
| v3rel | 0.1 to 1 MiB | 38 | 43 | 188 | 8 |
| v3rel | 1 MiB or more | 5 | 0 | 606 | 2 |

## Excess against swap-out in the prompt

Requests with 10 steps or more, split by the swap-out (vmstat so) between the request's start and its first token: R86's waits followed a swap-out at the end of the prompt.

| arm | swap-out in the prompt | requests | median excess, ms | their excess, s | excess a task, s | above the cut |
|---|---|---:|---:|---:|---:|---:|
| v4night | under 20 MiB | 57 | -32 | 3.64 | 0.30 | 4 |
| v4night | 20 MiB or more | 8 | -24 | 0.48 | 0.04 | 1 |
| v3rel | under 20 MiB | 49 | 39 | 5.43 | 0.45 | 5 |
| v3rel | 20 MiB or more | 10 | 252 | 5.70 | 0.47 | 5 |
