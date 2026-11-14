### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v3` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v3 | 85.1 (11.74) | 85.2 (11.73) | 0.1 |  |  |
|  |  | v4 | 84.5 (11.83) | 84.0 (11.90) | 0.5 | −1.1 % | loss |
|  |  | v4s | 111.1 (9.00) | 110.2 (9.08) | 0.9 | +29.9 % | gain |
| 4,096 | 4,032 on 0 | v3 | 477.5 (2.09) | 478.2 (2.09) | 0.8 |  |  |
|  |  | v4 | 503.3 (1.99) | 515.7 (1.94) | 12.3 | +6.6 % | gain |
|  |  | v4s | 539.5 (1.85) | 542.0 (1.84) | 2.5 | +13.2 % | gain |
| 8,192 | 4,100 on 4,028 | v3 | 442.6 (2.26) | 446.7 (2.24) | 4.0 |  |  |
|  |  | v4 | 443.6 (2.25) | 476.3 (2.10) | 32.7 | +3.4 % | negligible |
|  |  | v4s | 457.4 (2.19) | 511.4 (1.96) | 54.0 | +8.9 % | negligible |
| 16,384 | 12,292 on 4,028 | v3 | 472.6 (2.12) | 475.4 (2.10) | 2.8 |  |  |
|  |  | v4 | 488.5 (2.05) | 490.8 (2.04) | 2.3 | +3.3 % | gain |
|  |  | v4s | 462.1 (2.16) | 466.7 (2.14) | 4.5 | −2.0 % | loss |
| 32,768 | 16,388 on 16,316 | v3 | 321.9 (3.11) | 322.3 (3.10) | 0.4 |  |  |
|  |  | v4 | 332.7 (3.01) | 332.2 (3.01) | 0.5 | +3.2 % | gain |
|  |  | v4s | 263.8 (3.79) | 264.0 (3.79) | 0.2 | −18.1 % | loss |
| 49,152 | 16,388 on 32,700 | v3 | 212.7 (4.70) | 213.6 (4.68) | 0.9 |  |  |
|  |  | v4 | 218.0 (4.59) | 219.2 (4.56) | 1.2 | +2.6 % | gain |
|  |  | v4s | 180.8 (5.53) | 180.7 (5.53) | 0.1 | −15.2 % | loss |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---|---:|---|---|---:|---:|---|
| v3 | 53,200 | 309.9 (171.7 s) | 311.0 (171.1 s) | 1.2 |  |  |
| v4 | 53,200 | 318.9 (166.8 s) | 321.3 (165.6 s) | 2.5 | +3.1 % | gain |
| v4s | 53,200 | 271.7 (195.8 s) | 273.5 (194.5 s) | 1.7 | −12.2 % | loss |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `v31` | v3 | 1,453 | 959 | 959 | 293 | +0 | 215 | 7.83 / 6.56 % | did not act |
| `v32` | v3 | 1,453 | 943 | 943 | 294 | +0 | 209 | 12.22 / 11.05 % | did not act |
| `v41` | v4 | 1,437 | 942 | 942 | 295 | +0 | 217 | 5.77 / 5.12 % | did not act |
| `v42` | v4 | 1,436 | 954 | 954 | 295 | +0 | 217 | 8.39 / 7.42 % | did not act |
| `s1` | v4s | 1,413 | 1,136 | 1,136 | 295 | +0 | 218 | 9.03 / 7.81 % | did not act |
| `s2` | v4s | 1,412 | 1,119 | 1,119 | 295 | +0 | 203 | 6.08 / 5.17 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v31` | v3 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v32` | v3 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v41` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `v42` | v4 | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `s1` | v4s | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |
| `s2` | v4s | 5,383 | 5,923 | 5,081 | 5,591 | 5,923 | 5,081 | 5,545 | 5,293 | 4,793 | 48,613 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `v31` | `v32` | `v41` | `v42` | `s1` | `s2` |
|---:|---|---|---|---|---|---|
| 0 | 46.02, 54.59; 21.7, 18.3 ms; 0/0; 576dc4 | 46.14, 54.46; 21.7, 18.4 ms; 0/0; 576dc4 | 45.23, 55.34; 22.1, 18.1 ms; 0/0; 576dc4 | 46.87, 55.42; 21.3, 18.0 ms; 0/0; 576dc4 | 47.61, 55.78; 21.0, 17.9 ms; 0/0; 576dc4 | 49.14, 55.74; 20.4, 17.9 ms; 0/0; 576dc4 |
| 4,096 | 73.94, 79.26; 13.5, 12.6 ms; 0/0; df16a0 | 73.99, 79.27; 13.5, 12.6 ms; 0/0; df16a0 | 74.66, 80.16; 13.4, 12.5 ms; 0/0; df16a0 | 74.68, 80.17; 13.4, 12.5 ms; 0/0; df16a0 | 73.32, 79.79; 13.6, 12.5 ms; 0/0; df16a0 | 73.25, 79.82; 13.7, 12.5 ms; 0/0; df16a0 |
| 8,192 | 46.30, 50.63; 21.6, 19.8 ms; 0/0; e20d7b | 46.31, 50.62; 21.6, 19.8 ms; 0/0; e20d7b | 46.63, 51.12; 21.4, 19.6 ms; 0/0; e20d7b | 46.44, 51.02; 21.5, 19.6 ms; 0/0; e20d7b | 45.84, 50.59; 21.8, 19.8 ms; 0/0; e20d7b | 46.01, 50.56; 21.7, 19.8 ms; 0/0; e20d7b |
| 16,384 | 63.90, 69.46; 15.6, 14.4 ms; 0/0; f8f10f | 64.14, 69.56; 15.6, 14.4 ms; 0/0; f8f10f | 64.11, 69.96; 15.6, 14.3 ms; 0/0; f8f10f | 64.62, 69.99; 15.5, 14.3 ms; 0/0; f8f10f | 63.34, 69.11; 15.8, 14.5 ms; 0/0; f8f10f | 63.36, 69.15; 15.8, 14.5 ms; 0/0; f8f10f |
| 32,768 | 25.70, 26.67; 38.9, 37.5 ms; 0/0; 76d9bb | 25.67, 26.66; 39.0, 37.5 ms; 0/0; 76d9bb | 25.85, 26.87; 38.7, 37.2 ms; 0/0; 76d9bb | 25.87, 26.87; 38.7, 37.2 ms; 0/0; 76d9bb | 25.31, 26.32; 39.5, 38.0 ms; 0/0; 76d9bb | 25.35, 26.33; 39.5, 38.0 ms; 0/0; 76d9bb |
| 49,152 | 22.87, 23.72; 43.7, 42.2 ms; 0/0; 8cdee9 | 22.90, 23.75; 43.7, 42.1 ms; 0/0; 8cdee9 | 23.03, 23.87; 43.4, 41.9 ms; 0/0; 8cdee9 | 23.02, 23.87; 43.4, 41.9 ms; 0/0; 8cdee9 | 22.28, 23.15; 44.9, 43.2 ms; 0/0; 8cdee9 | 22.33, 23.17; 44.8, 43.2 ms; 0/0; 8cdee9 |
