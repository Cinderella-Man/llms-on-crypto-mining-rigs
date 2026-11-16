### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v4` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v4` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v4 | 86.4 (11.57) | 87.1 (11.48) | 0.7 |  |  |
|  |  | v4gf | 85.5 (11.69) | 86.4 (11.57) | 0.9 | −0.9 % | negligible |
|  |  | v4hs | 88.7 (11.27) | 89.4 (11.19) | 0.7 | +2.6 % | gain |
| 4,096 | 4,032 on 0 | v4 | 519.0 (1.93) | 523.4 (1.91) | 4.4 |  |  |
|  |  | v4gf | 541.9 (1.85) | 464.8 (2.15) | 77.0 | −3.4 % | negligible |
|  |  | v4hs | 532.6 (1.88) | 447.6 (2.23) | 84.9 | −6.0 % | negligible |
| 8,192 | 4,100 on 4,028 | v4 | 478.2 (2.09) | 479.2 (2.09) | 1.0 |  |  |
|  |  | v4gf | 498.5 (2.01) | 483.6 (2.07) | 15.0 | +2.6 % | negligible |
|  |  | v4hs | 446.1 (2.24) | 437.4 (2.29) | 8.6 | −7.7 % | loss |
| 16,384 | 12,292 on 4,028 | v4 | 493.8 (2.03) | 483.1 (2.07) | 10.7 |  |  |
|  |  | v4gf | 516.4 (1.94) | 514.5 (1.94) | 1.9 | +5.5 % | gain |
|  |  | v4hs | 498.5 (2.01) | 498.4 (2.01) | 0.1 | +2.1 % | negligible |
| 32,768 | 16,388 on 16,316 | v4 | 338.6 (2.95) | 332.6 (3.01) | 6.1 |  |  |
|  |  | v4gf | 334.7 (2.99) | 334.5 (2.99) | 0.3 | −0.3 % | negligible |
|  |  | v4hs | 335.1 (2.98) | 334.5 (2.99) | 0.5 | −0.2 % | negligible |
| 49,152 | 16,388 on 32,700 | v4 | 220.0 (4.54) | 218.2 (4.58) | 1.8 |  |  |
|  |  | v4gf | 218.8 (4.57) | 217.9 (4.59) | 0.9 | −0.4 % | negligible |
|  |  | v4hs | 217.9 (4.59) | 218.3 (4.58) | 0.3 | −0.5 % | negligible |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v4` | Class |
|---|---:|---|---|---:|---:|---|
| v4 | 53,200 | 324.2 (164.1 s) | 320.3 (166.1 s) | 3.8 |  |  |
| v4gf | 53,200 | 325.7 (163.3 s) | 321.9 (165.3 s) | 3.8 | +0.5 % | negligible |
| v4hs | 53,200 | 321.4 (165.5 s) | 318.3 (167.1 s) | 3.1 | −0.7 % | negligible |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `v41` | v4 | 1,440 | 958 | 958 | 293 | +0 | 221 | 8.95 / 8.40 % | did not act |
| `v42` | v4 | 1,438 | 960 | 960 | 293 | +0 | 220 | 9.31 / 8.32 % | did not act |
| `gf1` | v4gf | 1,436 | 952 | 952 | 293 | +0 | 218 | 4.85 / 3.88 % | did not act |
| `gf2` | v4gf | 1,436 | 949 | 949 | 294 | +0 | 197 | 16.74 / 15.85 % | did not act |
| `hs1` | v4hs | 1,414 | 937 | 937 | 293 | +0 | 217 | 6.42 / 5.77 % | did not act |
| `hs2` | v4hs | 1,417 | 931 | 931 | 293 | +0 | 215 | 20.51 / 19.35 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v41` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v42` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `gf1` | v4gf | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `gf2` | v4gf | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `hs1` | v4hs | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `hs2` | v4hs | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `v41` | `v42` | `gf1` | `gf2` | `hs1` | `hs2` |
|---:|---|---|---|---|---|---|
| 0 | 46.41, 55.50; 21.5, 18.0 ms; 0/0; 576dc4 | 43.74, 55.79; 22.9, 17.9 ms; 0/0; 576dc4 | 46.93, 54.76; 21.3, 18.3 ms; 0/0; 576dc4 | 46.71, 55.08; 21.4, 18.2 ms; 0/0; 576dc4 | 48.32, 57.02; 20.7, 17.5 ms; 0/0; 576dc4 | 48.77, 56.91; 20.5, 17.6 ms; 0/0; 576dc4 |
| 4,096 | 74.56, 80.33; 13.4, 12.4 ms; 0/0; df16a0 | 74.67, 80.43; 13.4, 12.4 ms; 0/0; df16a0 | 74.66, 80.34; 13.4, 12.4 ms; 0/0; df16a0 | 74.77, 80.30; 13.4, 12.5 ms; 0/0; df16a0 | 76.05, 81.44; 13.1, 12.3 ms; 0/0; df16a0 | 76.15, 81.52; 13.1, 12.3 ms; 0/0; df16a0 |
| 8,192 | 46.75, 51.22; 21.4, 19.5 ms; 0/0; e20d7b | 47.81, 51.23; 20.9, 19.5 ms; 0/0; e20d7b | 47.44, 51.23; 21.1, 19.5 ms; 0/0; e20d7b | 47.73, 51.23; 21.0, 19.5 ms; 0/0; e20d7b | 47.35, 51.78; 21.1, 19.3 ms; 0/0; e20d7b | 47.16, 51.76; 21.2, 19.3 ms; 0/0; e20d7b |
| 16,384 | 64.59, 70.21; 15.5, 14.2 ms; 0/0; f8f10f | 64.89, 70.27; 15.4, 14.2 ms; 0/0; f8f10f | 64.87, 70.30; 15.4, 14.2 ms; 0/0; f8f10f | 64.88, 70.30; 15.4, 14.2 ms; 0/0; f8f10f | 65.37, 71.03; 15.3, 14.1 ms; 0/0; f8f10f | 65.63, 71.10; 15.2, 14.1 ms; 0/0; f8f10f |
| 32,768 | 25.89, 26.91; 38.6, 37.2 ms; 0/0; 76d9bb | 26.01, 27.00; 38.4, 37.0 ms; 0/0; 76d9bb | 25.98, 26.96; 38.5, 37.1 ms; 0/0; 76d9bb | 26.01, 27.00; 38.4, 37.0 ms; 0/0; 76d9bb | 26.16, 27.22; 38.2, 36.7 ms; 0/0; 76d9bb | 26.17, 27.25; 38.2, 36.7 ms; 0/0; 76d9bb |
| 49,152 | 23.12, 23.95; 43.3, 41.8 ms; 0/0; 8cdee9 | 23.16, 24.00; 43.2, 41.7 ms; 0/0; 8cdee9 | 23.08, 23.97; 43.3, 41.7 ms; 0/0; 8cdee9 | 23.09, 23.99; 43.3, 41.7 ms; 0/0; 8cdee9 | 23.30, 24.17; 42.9, 41.4 ms; 0/0; 8cdee9 | 23.31, 24.18; 42.9, 41.4 ms; 0/0; 8cdee9 |
