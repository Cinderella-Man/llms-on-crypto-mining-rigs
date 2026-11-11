### Prompt reading: tok/s (ms a prompt token)

Per load the tokens its prompt request read over their time. Class of each arm against `draft` by the deep lane's rule. The draft adds: the arm's mean ms a prompt token minus `nodraft`'s. Taken away: the share of what the draft adds in `draft` that the arm does not add.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `draft` | Class | The draft adds, ms a prompt token | Taken away |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 0 | 64 on 0 | draft | 33.6 (29.76) | 32.8 (30.51) | 0.8 |  |  | +0.65 |  |
|  |  | kvonly | 33.5 (29.86) | 32.9 (30.43) | 0.6 | −0.0 % | negligible | +0.66 | −1.2 % |
|  |  | nodraft | 34.1 (29.29) | 33.7 (29.68) | 0.4 | +2.2 % | negligible |  |  |
| 8,192 | 8,128 on 0 | draft | 138.1 (7.24) | 138.6 (7.22) | 0.5 |  |  | +1.70 |  |
|  |  | kvonly | 146.4 (6.83) | 147.9 (6.76) | 1.5 | +6.4 % | gain | +1.26 | 25.5 % |
|  |  | nodraft | 180.1 (5.55) | 181.4 (5.51) | 1.3 | +30.7 % | gain |  |  |
| 16,384 | 8,708 on 7,612 | draft | 114.8 (8.71) | 115.1 (8.69) | 0.3 |  |  | +2.75 |  |
|  |  | kvonly | 122.9 (8.14) | 123.6 (8.09) | 0.7 | +7.2 % | gain | +2.16 | 21.3 % |
|  |  | nodraft | 167.9 (5.96) | 168.1 (5.95) | 0.2 | +46.1 % | gain |  |  |
| 32,768 | 16,388 on 16,316 | draft | 93.6 (10.68) | 94.4 (10.59) | 0.8 |  |  | +4.00 |  |
|  |  | kvonly | 103.2 (9.69) | 103.7 (9.64) | 0.5 | +10.0 % | gain | +3.03 | 24.2 % |
|  |  | nodraft | 150.8 (6.63) | 150.7 (6.63) | 0.1 | +60.3 % | gain |  |  |
| 49,152 | 16,900 on 32,188 | draft | 72.2 (13.85) | 72.3 (13.82) | 0.2 |  |  | +5.59 |  |
|  |  | kvonly | 80.8 (12.38) | 80.8 (12.37) | 0.1 | +11.8 % | gain | +4.13 | 26.2 % |
|  |  | nodraft | 121.6 (8.22) | 120.9 (8.27) | 0.7 | +67.8 % | gain |  |  |
| all primes | 50,124 tokens | draft | 92.2 (10.85) | 92.6 (10.80) | 0.4 |  |  | +3.95 |  |
|  |  | kvonly | 101.4 (9.86) | 101.8 (9.83) | 0.4 | +10.0 % | gain | +2.96 | 24.9 % |
|  |  | nodraft | 145.4 (6.88) | 145.2 (6.88) | 0.2 | +57.4 % | gain |  |  |

The primes of a load together, seconds: `draft` 543.8 and 541.5; `kvonly` 494.4 and 492.5; `nodraft` 344.7 and 345.1.

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved: the change in swap in use over the load, and the sum of its rises during the prime requests. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved over the load | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `draft1` | draft | 907 | 154 | 154 | 848 | +469 | +691 | 252 | 5.47 / 3.84 % | did not act |
| `draft2` | draft | 870 | 249 | 249 | 988 | +1 | +812 | 217 | 6.33 / 4.60 % | did not act |
| `kvonly1` | kvonly | 899 | 93 | 93 | 467 | +1 | +406 | 145 | 2.49 / 1.71 % | did not act |
| `kvonly2` | kvonly | 905 | 66 | 66 | 55 | +0 | +0 | 5 | 1.51 / 1.35 % | did not act |
| `nodraft1` | nodraft | 1,006 | 665 | 665 | 934 | +2 | +622 | 339 | 12.67 / 10.52 % | did not act |
| `nodraft2` | nodraft | 1,009 | 548 | 548 | 917 | +594 | +593 | 229 | 13.59 / 13.26 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `draft1` | draft | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,647 | 44,101 |
| `draft2` | draft | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,647 | 44,101 |
| `kvonly1` | kvonly | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,647 | 44,101 |
| `kvonly2` | kvonly | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,647 | 44,101 |
| `nodraft1` | nodraft | 5,013 | 5,061 | 4,917 | 5,077 | 4,191 | 4,933 | 4,355 | 4,945 | 2,411 | 40,903 |
| `nodraft2` | nodraft | 5,013 | 5,061 | 4,917 | 5,077 | 4,191 | 4,933 | 4,355 | 4,945 | 2,411 | 40,903 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2, the draft tokens accepted of drafted on both, and the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `draft1` | `draft2` | `kvonly1` | `kvonly2` | `nodraft1` | `nodraft2` |
|---:|---|---|---|---|---|---|
| 0 | 7.29, 16.00; 14/46; 9da71b | 7.15, 15.98; 14/46; 9da71b | 7.18, 16.23; 14/46; 9da71b | 7.17, 16.23; 14/46; 9da71b | 19.74, 20.37; 9da71b | 19.83, 20.39; 9da71b |
| 8,192 | 6.92, 7.29; 14/44; 652878 | 6.48, 7.29; 14/44; 652878 | 6.99, 7.38; 14/44; 652878 | 6.98, 7.38; 14/44; 652878 | 5.64, 17.44; 652878 | 6.29, 11.92; 652878 |
| 16,384 | 10.94, 15.81; 18/36; d4dde6 | 10.80, 17.31; 18/36; d4dde6 | 16.41, 17.63; 18/36; d4dde6 | 16.49, 17.65; 18/36; d4dde6 | 5.42, 17.66; d4dde6 | 9.33, 13.59; d4dde6 |
| 32,768 | 8.62, 14.42; 16/32; 0795e3 | 12.65, 14.74; 16/32; 0795e3 | 14.41, 15.04; 16/32; 0795e3 | 14.43, 14.98; 16/32; 0795e3 | 15.39, 16.33; 1ca520 | 8.12, 16.26; 1ca520 |
| 49,152 | 11.78, 17.26; 20/28; fc1933 | 16.60, 17.29; 20/28; fc1933 | 10.22, 17.52; 20/28; fc1933 | 17.00, 17.56; 20/28; fc1933 | 14.75, 15.16; fc1933 | 13.43, 15.16; fc1933 |
