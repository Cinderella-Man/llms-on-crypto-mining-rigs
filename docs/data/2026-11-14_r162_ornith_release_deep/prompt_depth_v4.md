### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v4` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v4` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v4 | 24.6 (40.67) | 24.6 (40.70) | 0.0 |  |  |
|  |  | v3 | 24.6 (40.62) | 24.3 (41.14) | 0.3 | −0.5 % | negligible |
|  |  | v4s | 40.6 (24.64) | 40.6 (24.66) | 0.0 | +65.1 % | gain |
| 4,096 | 4,032 on 0 | v4 | 133.8 (7.47) | 133.9 (7.47) | 0.1 |  |  |
|  |  | v3 | 132.4 (7.55) | 132.2 (7.56) | 0.2 | −1.1 % | loss |
|  |  | v4s | 169.4 (5.90) | 168.8 (5.92) | 0.6 | +26.4 % | gain |
| 8,192 | 4,612 on 3,516 | v4 | 143.2 (6.99) | 143.6 (6.96) | 0.4 |  |  |
|  |  | v3 | 144.8 (6.91) | 144.5 (6.92) | 0.3 | +0.9 % | gain |
|  |  | v4s | 182.3 (5.49) | 182.2 (5.49) | 0.1 | +27.1 % | gain |
| 16,384 | 12,804 on 3,516 | v4 | 191.9 (5.21) | 191.7 (5.22) | 0.1 |  |  |
|  |  | v3 | 193.8 (5.16) | 193.4 (5.17) | 0.3 | +0.9 % | gain |
|  |  | v4s | 219.8 (4.55) | 219.4 (4.56) | 0.4 | +14.5 % | gain |
| 32,768 | 16,900 on 15,804 | v4 | 152.3 (6.57) | 152.3 (6.57) | 0.0 |  |  |
|  |  | v3 | 155.5 (6.43) | 155.5 (6.43) | 0.0 | +2.1 % | gain |
|  |  | v4s | 168.8 (5.92) | 168.6 (5.93) | 0.2 | +10.8 % | gain |
| 48,640 | 16,388 on 32,188 | v4 | 119.7 (8.36) | 119.2 (8.39) | 0.4 |  |  |
|  |  | v3 | 120.7 (8.29) | 120.5 (8.30) | 0.1 | +1.0 % | gain |
|  |  | v4s | 136.7 (7.31) | 136.4 (7.33) | 0.4 | +14.3 % | gain |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v4` | Class |
|---|---:|---|---|---:|---:|---|
| v4 | 54,736 | 145.2 (377.0 s) | 145.0 (377.4 s) | 0.1 |  |  |
| v3 | 54,736 | 146.8 (372.8 s) | 146.7 (373.2 s) | 0.2 | +1.1 % | gain |
| v4s | 54,736 | 167.2 (327.3 s) | 166.9 (328.0 s) | 0.3 | +15.1 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `v41` | v4 | 1,528 | 651 | 651 | 336 | +0 | 259 | 7.06 / 6.18 % | did not act |
| `v42` | v4 | 1,528 | 655 | 655 | 336 | +0 | 221 | 7.02 / 5.82 % | did not act |
| `v31` | v3 | 1,538 | 672 | 672 | 335 | +0 | 256 | 9.41 / 8.36 % | did not act |
| `v32` | v3 | 1,538 | 661 | 661 | 335 | +0 | 242 | 5.94 / 4.81 % | did not act |
| `s1` | v4s | 1,516 | 1,238 | 1,238 | 337 | +0 | 243 | 7.16 / 5.89 % | did not act |
| `s2` | v4s | 1,512 | 1,254 | 1,254 | 336 | +0 | 241 | 9.32 / 7.57 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v41` | v4 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `v42` | v4 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `v31` | v3 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `v32` | v3 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `s1` | v4s | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `s2` | v4s | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `v41` | `v42` | `v31` | `v32` | `s1` | `s2` |
|---:|---|---|---|---|---|---|
| 0 | 18.18, 18.59; 55.0, 53.8 ms; 0/0; b82797 | 18.19, 18.60; 55.0, 53.8 ms; 0/0; b82797 | 18.17, 18.51; 55.0, 54.0 ms; 0/0; b82797 | 18.16, 18.50; 55.1, 54.1 ms; 0/0; b82797 | 18.24, 18.66; 54.8, 53.6 ms; 0/0; b82797 | 18.23, 18.64; 54.8, 53.6 ms; 0/0; b82797 |
| 4,096 | 16.81, 16.92; 59.5, 59.1 ms; 0/0; 5eb4b3 | 16.81, 16.92; 59.5, 59.1 ms; 0/0; 5eb4b3 | 16.79, 16.84; 59.6, 59.4 ms; 0/0; 5eb4b3 | 16.77, 16.82; 59.6, 59.4 ms; 0/0; 5eb4b3 | 16.88, 17.00; 59.2, 58.8 ms; 0/0; 5eb4b3 | 16.87, 16.99; 59.3, 58.9 ms; 0/0; 5eb4b3 |
| 8,192 | 15.51, 15.58; 64.5, 64.2 ms; 0/0; eea8be | 15.50, 15.57; 64.5, 64.2 ms; 0/0; eea8be | 15.47, 15.45; 64.6, 64.7 ms; 0/0; eea8be | 15.45, 15.45; 64.7, 64.7 ms; 0/0; eea8be | 15.50, 15.58; 64.5, 64.2 ms; 0/0; 5e66d7 | 15.53, 15.60; 64.4, 64.1 ms; 0/0; 5e66d7 |
| 16,384 | 13.37, 13.41; 74.8, 74.6 ms; 0/0; d7e7c7 | 13.36, 13.40; 74.8, 74.6 ms; 0/0; d7e7c7 | 13.33, 13.34; 75.0, 74.9 ms; 0/0; d7e7c7 | 13.33, 13.33; 75.0, 75.0 ms; 0/0; d7e7c7 | 13.34, 13.39; 74.9, 74.7 ms; 0/0; a15fa4 | 13.35, 13.39; 74.9, 74.7 ms; 0/0; a15fa4 |
| 32,768 | 10.44, 10.47; 95.8, 95.5 ms; 0/0; 576f56 | 10.42, 10.45; 95.9, 95.7 ms; 0/0; 576f56 | 10.42, 10.43; 95.9, 95.9 ms; 0/0; 576f56 | 10.41, 10.43; 96.0, 95.9 ms; 0/0; 576f56 | 10.43, 10.46; 95.9, 95.6 ms; 0/0; f1d8fe | 10.42, 10.44; 96.0, 95.8 ms; 0/0; f1d8fe |
| 48,640 | 8.62, 8.63; 116.0, 115.9 ms; 0/0; d340d8 | 8.61, 8.63; 116.1, 115.9 ms; 0/0; d340d8 | 8.61, 8.61; 116.1, 116.1 ms; 0/0; d340d8 | 8.61, 8.61; 116.2, 116.2 ms; 0/0; d340d8 | 8.64, 8.66; 115.7, 115.5 ms; 0/0; d340d8 | 8.62, 8.64; 116.0, 115.8 ms; 0/0; d340d8 |
