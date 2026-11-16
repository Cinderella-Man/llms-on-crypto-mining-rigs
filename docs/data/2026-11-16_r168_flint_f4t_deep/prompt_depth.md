### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v3` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v3 | 86.4 (11.57) | 83.8 (11.94) | 2.7 |  |  |
|  |  | v4 | 91.1 (10.98) | 88.3 (11.32) | 2.8 | +5.4 % | gain |
| 4,096 | 4,032 on 0 | v3 | 479.7 (2.08) | 479.9 (2.08) | 0.2 |  |  |
|  |  | v4 | 549.3 (1.82) | 566.8 (1.76) | 17.5 | +16.3 % | gain |
| 8,192 | 4,100 on 4,028 | v3 | 450.0 (2.22) | 432.1 (2.31) | 17.8 |  |  |
|  |  | v4 | 524.7 (1.91) | 540.5 (1.85) | 15.8 | +20.8 % | gain |
| 16,384 | 12,292 on 4,028 | v3 | 477.2 (2.10) | 475.7 (2.10) | 1.5 |  |  |
|  |  | v4 | 520.4 (1.92) | 539.0 (1.86) | 18.5 | +11.2 % | gain |
| 32,768 | 16,388 on 16,316 | v3 | 327.6 (3.05) | 326.0 (3.07) | 1.6 |  |  |
|  |  | v4 | 338.6 (2.95) | 342.2 (2.92) | 3.6 | +4.2 % | gain |
| 49,152 | 16,388 on 32,700 | v3 | 213.3 (4.69) | 214.1 (4.67) | 0.8 |  |  |
|  |  | v4 | 221.2 (4.52) | 222.0 (4.50) | 0.8 | +3.7 % | gain |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---|---:|---|---|---:|---:|---|
| v3 | 53,200 | 312.7 (170.1 s) | 312.0 (170.5 s) | 0.7 |  |  |
| v4 | 53,200 | 329.9 (161.3 s) | 334.1 (159.2 s) | 4.2 | +6.3 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `v31` | v3 | 1,435 | 940 | 940 | 293 | +0 | 148 | 7.45 / 6.36 % | did not act |
| `v32` | v3 | 1,432 | 952 | 952 | 294 | +0 | 206 | 16.21 / 15.22 % | did not act |
| `v41` | v4 | 1,420 | 1,149 | 1,149 | 294 | +0 | 203 | 21.26 / 18.77 % | did not act |
| `v42` | v4 | 1,427 | 1,130 | 1,130 | 295 | +0 | 220 | 12.31 / 11.20 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v31` | v3 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v32` | v3 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v41` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v42` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `v31` | `v32` | `v41` | `v42` |
|---:|---|---|---|---|
| 0 | 47.10, 54.86; 21.2, 18.2 ms; 0/0; 576dc4 | 43.05, 54.90; 23.2, 18.2 ms; 0/0; 576dc4 | 46.91, 55.28; 21.3, 18.1 ms; 0/0; 576dc4 | 46.22, 55.27; 21.6, 18.1 ms; 0/0; 576dc4 |
| 4,096 | 73.97, 79.55; 13.5, 12.6 ms; 0/0; df16a0 | 74.03, 79.41; 13.5, 12.6 ms; 0/0; df16a0 | 75.09, 80.44; 13.3, 12.4 ms; 0/0; df16a0 | 74.99, 80.43; 13.3, 12.4 ms; 0/0; df16a0 |
| 8,192 | 46.28, 50.81; 21.6, 19.7 ms; 0/0; e20d7b | 46.55, 50.80; 21.5, 19.7 ms; 0/0; e20d7b | 47.08, 51.33; 21.2, 19.5 ms; 0/0; e20d7b | 47.71, 51.28; 21.0, 19.5 ms; 0/0; e20d7b |
| 16,384 | 64.17, 69.82; 15.6, 14.3 ms; 0/0; f8f10f | 64.31, 69.83; 15.6, 14.3 ms; 0/0; f8f10f | 65.00, 70.44; 15.4, 14.2 ms; 0/0; f8f10f | 65.06, 70.37; 15.4, 14.2 ms; 0/0; f8f10f |
| 32,768 | 25.79, 26.78; 38.8, 37.3 ms; 0/0; 76d9bb | 25.79, 26.79; 38.8, 37.3 ms; 0/0; 76d9bb | 26.05, 27.02; 38.4, 37.0 ms; 0/0; 76d9bb | 26.03, 27.00; 38.4, 37.0 ms; 0/0; 76d9bb |
| 49,152 | 23.02, 23.86; 43.4, 41.9 ms; 0/0; 8cdee9 | 23.01, 23.87; 43.5, 41.9 ms; 0/0; 8cdee9 | 23.15, 24.02; 43.2, 41.6 ms; 0/0; 8cdee9 | 23.15, 24.02; 43.2, 41.6 ms; 0/0; 8cdee9 |
