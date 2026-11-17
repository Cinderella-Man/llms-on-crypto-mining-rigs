## What a pass of k tokens costs (the speculative arms, step by step, host profile)

Per load, depth, request and width k (the step's tokens: 1 + the drafted tokens, every one accepted): the steps, the step's time, the nine cards' GPU time and the rest (step less GPU), medians over the steps. A step of width 1 is a one-token pass with no draft found and no pipelining (a slot that can speculate never pipelines).

| Load | Depth | Request | k | Steps | Step ms | GPU ms | Rest ms | Step ms a token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s8 | 0 | 1 | 1 | 18 | 28.22 | 25.79 | 2.25 | 28.22 |
| s8 | 0 | 1 | 5 | 1 | 192.83 | 177.05 | 15.78 | 38.57 |
| s8 | 0 | 1 | 8 | 29 | 280.99 | 273.54 | 7.39 | 35.12 |
| s8 | 0 | 2 | 1 | 18 | 27.29 | 25.79 | 1.50 | 27.29 |
| s8 | 0 | 2 | 5 | 1 | 195.57 | 182.31 | 13.26 | 39.11 |
| s8 | 0 | 2 | 8 | 29 | 281.04 | 273.59 | 7.38 | 35.13 |
| s8 | 8192 | 1 | 8 | 26 | 286.38 | 276.55 | 8.31 | 35.80 |
| s8 | 8192 | 1 | 6 | 7 | 223.25 | 216.79 | 6.46 | 37.21 |
| s8 | 8192 | 1 | 5 | 1 | 213.59 | 201.90 | 11.69 | 42.72 |
| s8 | 8192 | 2 | 8 | 26 | 289.15 | 280.68 | 8.30 | 36.14 |
| s8 | 8192 | 2 | 6 | 7 | 222.38 | 215.91 | 6.47 | 37.06 |
| s4 | 0 | 1 | 1 | 36 | 29.82 | 25.84 | 4.00 | 29.82 |
| s4 | 0 | 1 | 4 | 23 | 146.17 | 140.76 | 6.58 | 36.54 |
| s4 | 0 | 1 | 2 | 63 | 74.02 | 71.64 | 2.39 | 37.01 |
| s4 | 0 | 2 | 1 | 36 | 27.42 | 25.90 | 1.52 | 27.42 |
| s4 | 0 | 2 | 4 | 23 | 144.83 | 140.83 | 4.06 | 36.21 |
| s4 | 0 | 2 | 2 | 63 | 74.05 | 71.60 | 2.39 | 37.03 |
| s4 | 8192 | 1 | 4 | 63 | 149.24 | 144.70 | 4.58 | 37.31 |
| s4 | 8192 | 1 | 3 | 1 | 125.23 | 115.61 | 9.62 | 41.74 |
| s4 | 8192 | 2 | 4 | 63 | 149.29 | 144.75 | 4.57 | 37.32 |
| s2 | 0 | 1 | 1 | 45 | 30.92 | 26.56 | 4.72 | 30.92 |
| s2 | 0 | 1 | 2 | 105 | 75.33 | 72.92 | 2.41 | 37.66 |
| s2 | 0 | 2 | 1 | 45 | 28.12 | 26.61 | 1.55 | 28.12 |
| s2 | 0 | 2 | 2 | 105 | 75.28 | 72.91 | 2.40 | 37.64 |
| s2 | 8192 | 1 | 2 | 123 | 77.45 | 74.72 | 2.72 | 38.73 |
| s2 | 8192 | 1 | 1 | 8 | 40.35 | 31.52 | 8.83 | 40.35 |
| s2 | 8192 | 2 | 2 | 123 | 77.46 | 74.74 | 2.73 | 38.73 |
| s2 | 8192 | 2 | 1 | 8 | 40.36 | 31.52 | 8.83 | 40.36 |

### Each card's GPU ms a pass by k (the repeated request; the load with the most steps of that k)

| Card | 0, k=1 (s2) | 0, k=2 (s2) | 0, k=4 (s4) | 0, k=5 (s8) | 0, k=8 (s8) | 8192, k=1 (s2) | 8192, k=2 (s2) | 8192, k=4 (s4) | 8192, k=6 (s8) | 8192, k=8 (s8) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.32 | 8.34 | 15.60 | 20.88 | 31.78 | 4.38 | 9.14 | 15.44 | 25.55 | 33.93 |
| 1 | 3.02 | 8.78 | 17.08 | 23.61 | 35.25 | 3.29 | 8.27 | 15.94 | 23.36 | 30.28 |
| 2 | 2.53 | 7.32 | 14.11 | 18.86 | 28.15 | 3.10 | 7.77 | 14.84 | 21.63 | 28.07 |
| 3 | 2.57 | 7.43 | 14.15 | 18.72 | 26.98 | 3.18 | 7.50 | 15.48 | 18.52 | 24.51 |
| 4 | 2.85 | 8.18 | 15.56 | 20.87 | 30.33 | 3.39 | 8.61 | 13.83 | 25.13 | 32.51 |
| 5 | 2.52 | 7.33 | 14.34 | 20.07 | 28.78 | 2.86 | 7.10 | 15.10 | 21.67 | 28.44 |
| 6 | 2.39 | 6.39 | 12.97 | 17.95 | 25.25 | 2.98 | 6.44 | 14.43 | 22.46 | 27.01 |
| 7 | 2.52 | 6.90 | 14.43 | 16.82 | 25.39 | 3.24 | 7.86 | 17.06 | 23.82 | 30.87 |
| 8 | 4.47 | 12.02 | 22.74 | 24.53 | 41.09 | 5.07 | 12.07 | 22.69 | 32.62 | 44.13 |

## Notes

- s8: 6 step pieces, 7 profile pieces, 6 requests
- s8 depth 8192 pass 2: 35 steps, 34 evaluations a card (the file's last evaluation is never written): the last step left out
- s4: 6 step pieces, 7 profile pieces, 6 requests
- s4 depth 8192 pass 2: 65 steps, 64 evaluations a card (the file's last evaluation is never written): the last step left out
- s2: 6 step pieces, 7 profile pieces, 6 requests
- s2 depth 8192 pass 2: 133 steps, 132 evaluations a card (the file's last evaluation is never written): the last step left out
