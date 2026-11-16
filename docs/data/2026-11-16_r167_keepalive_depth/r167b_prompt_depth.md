### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v4r` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v4r` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v4r | 53.5 (18.70) | 51.7 (19.35) | 1.8 |  |  |
|  |  | v4nk | 47.2 (21.19) | 48.5 (20.61) | 1.3 | −9.0 % | loss |
|  |  | v4ne | 53.3 (18.75) | 53.9 (18.57) | 0.5 | +1.9 % | negligible |
| 4,096 | 4,032 on 0 | v4r | 198.0 (5.05) | 197.3 (5.07) | 0.7 |  |  |
|  |  | v4nk | 198.1 (5.05) | 200.1 (5.00) | 2.0 | +0.7 % | negligible |
|  |  | v4ne | 198.8 (5.03) | 197.6 (5.06) | 1.2 | +0.3 % | negligible |
| 8,192 | 4,612 on 3,516 | v4r | 207.9 (4.81) | 220.0 (4.54) | 12.1 |  |  |
|  |  | v4nk | 221.9 (4.51) | 218.9 (4.57) | 3.0 | +3.0 % | negligible |
|  |  | v4ne | 221.6 (4.51) | 223.4 (4.48) | 1.8 | +4.0 % | negligible |
| 16,384 | 12,804 on 3,516 | v4r | 275.7 (3.63) | 281.1 (3.56) | 5.3 |  |  |
|  |  | v4nk | 290.7 (3.44) | 282.6 (3.54) | 8.1 | +3.0 % | gain |
|  |  | v4ne | 284.7 (3.51) | 279.5 (3.58) | 5.1 | +1.3 % | negligible |
| 32,768 | 16,900 on 15,804 | v4r | 221.3 (4.52) | 224.6 (4.45) | 3.4 |  |  |
|  |  | v4nk | 230.1 (4.35) | 230.0 (4.35) | 0.1 | +3.2 % | gain |
|  |  | v4ne | 225.7 (4.43) | 225.4 (4.44) | 0.2 | +1.2 % | negligible |
| 49,152 | 16,900 on 32,188 | v4r | 178.7 (5.60) | 179.0 (5.59) | 0.3 |  |  |
|  |  | v4nk | 181.5 (5.51) | 181.4 (5.51) | 0.0 | +1.4 % | gain |
|  |  | v4ne | 174.7 (5.72) | 180.4 (5.54) | 5.8 | −0.7 % | negligible |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v4r` | Class |
|---|---:|---|---|---:|---:|---|
| v4r | 55,248 | 212.6 (259.9 s) | 215.3 (256.6 s) | 2.8 |  |  |
| v4nk | 55,248 | 219.4 (251.8 s) | 218.2 (253.2 s) | 1.2 | +2.3 % | gain |
| v4ne | 55,248 | 214.4 (257.7 s) | 216.2 (255.5 s) | 1.9 | +0.6 % | negligible |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `r1` | v4r | 866 | 247 | 247 | 45 | +14 | 1 | 0.36 / 0.26 % | did not act |
| `r2` | v4r | 840 | 268 | 268 | 52 | +5 | 1 | 0.33 / 0.18 % | did not act |
| `n1` | v4nk | 862 | 307 | 307 | 891 | +819 | 125 | 4.78 / 2.76 % | did not act |
| `n2` | v4nk | 874 | 288 | 288 | 910 | +726 | 137 | 5.61 / 3.01 % | did not act |
| `e1` | v4ne | 866 | 241 | 241 | 47 | +0 | 0 | 0.26 / 0.18 % | did not act |
| `e2` | v4ne | 854 | 275 | 275 | 672 | +528 | 100 | 3.21 / 1.51 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1` | v4r | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `r2` | v4r | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `n1` | v4nk | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `n2` | v4nk | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `e1` | v4ne | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `e2` | v4ne | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `r1` | `r2` | `n1` | `n2` | `e1` | `e2` |
|---:|---|---|---|---|---|---|
| 0 | 23.06, 24.10; 101.8, 98.3 ms; 295/639; 2aadc9 | 22.87, 24.09; 102.7, 98.4 ms; 295/639; 2aadc9 | 23.02, 23.73; 102.0, 99.9 ms; 295/639; 2aadc9 | 23.40, 24.16; 100.4, 98.1 ms; 295/639; 2aadc9 | 23.30, 24.10; 100.8, 98.3 ms; 295/639; 2aadc9 | 23.34, 24.10; 100.6, 98.3 ms; 295/639; 2aadc9 |
| 4,096 | 28.42, 29.54; 114.0, 109.7 ms; 354/462; 765bab | 28.45, 29.54; 113.9, 109.7 ms; 354/462; 765bab | 30.83, 30.45; 105.1, 106.4 ms; 354/462; 765bab | 30.85, 30.83; 105.0, 105.1 ms; 354/462; 765bab | 23.66, 29.56; 137.0, 109.6 ms; 354/462; 765bab | 23.73, 29.46; 136.6, 110.0 ms; 354/462; 765bab |
| 8,192 | 30.00, 31.59; 118.5, 114.1 ms; 369/421; 79c04b | 29.51, 31.60; 120.5, 114.1 ms; 369/421; 79c04b | 31.63, 30.87; 112.4, 116.8 ms; 369/421; 79c04b | 32.34, 32.31; 110.0, 111.6 ms; 369/421; 79c04b | 30.86, 31.61; 115.2, 114.1 ms; 369/421; 79c04b | 30.87, 31.61; 115.2, 114.0 ms; 369/421; 79c04b |
| 16,384 | 30.33, 31.37; 122.3, 118.3 ms; 374/406; 95387c | 30.39, 31.39; 122.1, 118.2 ms; 374/406; 95387c | 29.37, 31.83; 126.3, 116.6 ms; 374/406; 95387c | 28.55, 29.43; 129.9, 126.1 ms; 374/406; 95387c | 31.14, 31.24; 119.2, 118.8 ms; 374/406; 95387c | 31.15, 31.24; 119.1, 118.8 ms; 374/406; 95387c |
| 32,768 | 21.23, 22.05; 131.1, 127.6 ms; 329/543; 39f0dc | 21.20, 22.05; 131.3, 127.6 ms; 329/543; 39f0dc | 20.39, 22.40; 136.5, 125.6 ms; 329/543; 39f0dc | 21.32, 21.63; 130.5, 130.1 ms; 329/543; 39f0dc | 21.43, 22.02; 129.8, 127.8 ms; 329/543; 39f0dc | 21.00, 22.02; 132.5, 127.8 ms; 329/543; 39f0dc |
| 49,152 | 23.89, 24.45; 141.0, 137.7 ms; 360/450; fd3759 | 23.94, 24.49; 140.7, 137.5 ms; 360/450; fd3759 | 22.36, 20.96; 150.6, 160.7 ms; 360/450; fd3759 | 22.53, 21.43; 149.5, 157.2 ms; 360/450; fd3759 | 24.14, 24.26; 139.5, 138.8 ms; 360/450; fd3759 | 24.28, 24.48; 138.7, 137.6 ms; 360/450; fd3759 |
