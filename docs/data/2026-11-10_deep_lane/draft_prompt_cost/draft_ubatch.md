### Prompt reading: tok/s (ms a prompt token)

Per load the tokens its prompt request read over their time. Class of each arm against `draft` by the deep lane's rule. The draft adds: the arm's mean ms a prompt token minus `nodraft`'s. Taken away: the share of what the draft adds in `draft` that the arm does not add.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `draft` | Class | The draft adds, ms a prompt token | Taken away |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 0 | 64 on 0 | draft | 32.8 (30.52) | 32.8 (30.45) | 0.1 |  |  | +0.80 |  |
|  |  | dub256 | 32.8 (30.53) | 31.8 (31.47) | 1.0 | −1.6 % | negligible | +1.31 | −64.6 % |
|  |  | dub512 | 32.9 (30.40) | 32.6 (30.66) | 0.3 | −0.1 % | negligible | +0.84 | −5.1 % |
|  |  | nodraft | 33.5 (29.82) | 33.8 (29.57) | 0.3 | +2.7 % | gain |  |  |
| 8,192 | 8,128 on 0 | draft | 122.5 (8.17) | 129.2 (7.74) | 6.8 |  |  | +2.31 |  |
|  |  | dub256 | 130.2 (7.68) | 129.0 (7.75) | 1.2 | +3.0 % | negligible | +2.08 | 10.2 % |
|  |  | dub512 | 132.8 (7.53) | 131.9 (7.58) | 0.9 | +5.2 % | negligible | +1.92 | 17.1 % |
|  |  | nodraft | 177.1 (5.65) | 177.5 (5.63) | 0.4 | +40.9 % | gain |  |  |
| 16,384 | 8,708 on 7,612 | draft | 107.8 (9.28) | 108.9 (9.18) | 1.2 |  |  | +3.23 |  |
|  |  | dub256 | 110.7 (9.03) | 110.8 (9.02) | 0.1 | +2.2 % | gain | +3.03 | 6.2 % |
|  |  | dub512 | 113.2 (8.84) | 112.7 (8.87) | 0.5 | +4.2 % | gain | +2.85 | 11.6 % |
|  |  | nodraft | 167.0 (5.99) | 166.2 (6.02) | 0.8 | +53.8 % | gain |  |  |
| 32,768 | 16,388 on 16,316 | draft | 90.6 (11.04) | 90.4 (11.07) | 0.2 |  |  | +4.32 |  |
|  |  | dub256 | 92.4 (10.83) | 92.1 (10.86) | 0.3 | +1.9 % | gain | +4.11 | 4.8 % |
|  |  | dub512 | 93.8 (10.66) | 93.6 (10.69) | 0.2 | +3.6 % | gain | +3.94 | 8.8 % |
|  |  | nodraft | 148.5 (6.73) | 148.7 (6.73) | 0.1 | +64.2 % | gain |  |  |
| 49,152 | 16,900 on 32,188 | draft | 70.0 (14.29) | 70.1 (14.26) | 0.1 |  |  | +5.91 |  |
|  |  | dub256 | 71.1 (14.07) | 70.9 (14.10) | 0.2 | +1.4 % | gain | +5.72 | 3.2 % |
|  |  | dub512 | 72.4 (13.81) | 72.2 (13.85) | 0.2 | +3.2 % | gain | +5.46 | 7.6 % |
|  |  | nodraft | 119.5 (8.37) | 119.6 (8.36) | 0.1 | +70.7 % | gain |  |  |
| all primes | 50,124 tokens | draft | 88.0 (11.36) | 88.7 (11.28) | 0.7 |  |  | +4.34 |  |
|  |  | dub256 | 90.1 (11.10) | 89.8 (11.13) | 0.3 | +1.8 % | gain | +4.14 | 4.7 % |
|  |  | dub512 | 91.8 (10.90) | 91.5 (10.93) | 0.3 | +3.7 % | gain | +3.94 | 9.3 % |
|  |  | nodraft | 143.3 (6.98) | 143.3 (6.98) | 0.0 | +62.2 % | gain |  |  |

The primes of a load together, seconds: `draft` 569.6 and 565.2; `dub256` 556.2 and 557.9; `dub512` 546.2 and 548.0; `nodraft` 349.8 and 349.7.

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved: the change in swap in use over the load, and the sum of its rises during the prime requests. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved over the load | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `draft1` | draft | 884 | 71 | 71 | 512 | +16 | +281 | 275 | 19.50 / 17.67 % | did not act |
| `draft2` | draft | 930 | 81 | 81 | 1,012 | +688 | +754 | 208 | 6.17 / 4.48 % | did not act |
| `dub256_1` | dub256 | 905 | 80 | 80 | 1,131 | +767 | +882 | 228 | 15.18 / 13.40 % | did not act |
| `dub256_2` | dub256 | 904 | 79 | 79 | 1,147 | +0 | +969 | 206 | 11.15 / 9.77 % | did not act |
| `dub512_1` | dub512 | 871 | 263 | 263 | 1,064 | +730 | +905 | 353 | 5.80 / 3.71 % | did not act |
| `dub512_2` | dub512 | 873 | 238 | 238 | 1,097 | +734 | +880 | 284 | 10.29 / 8.28 % | did not act |
| `nodraft1` | nodraft | 984 | 268 | 268 | 44 | +0 | +0 | 0 | 5.54 / 5.33 % | did not act |
| `nodraft2` | nodraft | 1,025 | 257 | 257 | 46 | +0 | +0 | 0 | 0.36 / 0.29 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `draft1` | draft | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `draft2` | draft | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `dub256_1` | dub256 | 5,059 | 5,095 | 4,963 | 5,111 | 4,877 | 4,979 | 4,381 | 4,991 | 4,613 | 44,069 |
| `dub256_2` | dub256 | 5,059 | 5,095 | 4,963 | 5,111 | 4,877 | 4,979 | 4,381 | 4,991 | 4,613 | 44,069 |
| `dub512_1` | dub512 | 5,059 | 5,095 | 4,963 | 5,111 | 4,879 | 4,979 | 4,381 | 4,991 | 4,711 | 44,169 |
| `dub512_2` | dub512 | 5,059 | 5,095 | 4,963 | 5,111 | 4,879 | 4,979 | 4,381 | 4,991 | 4,711 | 44,169 |
| `nodraft1` | nodraft | 5,013 | 5,061 | 4,917 | 5,077 | 4,191 | 4,933 | 4,355 | 4,945 | 2,411 | 40,903 |
| `nodraft2` | nodraft | 5,013 | 5,061 | 4,917 | 5,077 | 4,191 | 4,933 | 4,355 | 4,945 | 2,411 | 40,903 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2, the draft tokens accepted of drafted on both, and the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `draft1` | `draft2` | `dub256_1` | `dub256_2` | `dub512_1` | `dub512_2` | `nodraft1` | `nodraft2` |
|---:|---|---|---|---|---|---|---|---|
| 0 | 7.24, 15.63; 14/46; 9da71b | 7.10, 15.74; 14/46; 9da71b | 7.11, 15.74; 14/46; 9da71b | 7.10, 15.73; 14/46; 9da71b | 7.09, 15.72; 14/46; 9da71b | 7.10, 15.69; 14/46; 9da71b | 19.63, 20.30; 9da71b | 19.66, 20.30; 9da71b |
| 8,192 | 4.16, 13.74; 15/41; 41e2ec | 4.95, 20.89; 20/26; 1931fa | 3.83, 12.09; 13/47; 6eb4e3 | 3.07, 15.15; 15/41; 41e2ec | 5.11, 10.11; 13/49; 6e8fef | 4.32, 13.39; 16/36; 0b6c15 | 8.68, 18.51; 907d1c | 8.68, 18.51; 907d1c |
| 16,384 | 6.18, 18.15; 18/30; db55ac | 5.22, 13.26; 18/30; db55ac | 4.84, 20.38; 20/28; 983ba3 | 13.28, 19.80; 20/28; 983ba3 | 13.89, 18.93; 20/28; 983ba3 | 12.10, 17.83; 20/28; 983ba3 | 16.72, 17.65; 983ba3 | 16.70, 17.63; 983ba3 |
| 32,768 | 15.11, 15.68; 18/34; 2b721e | 11.87, 13.75; 18/28; f46ec6 | 5.78, 17.31; 18/26; 1ca520 | 5.79, 15.63; 18/26; 25092c | 6.55, 14.59; 16/32; 1ca520 | 14.17, 14.71; 16/32; 1ca520 | 15.43, 16.28; f46ec6 | 15.51, 16.27; 9622a9 |
| 49,152 | 14.21, 14.74; 18/32; 89fc4f | 11.76, 13.20; 16/30; ce4127 | 16.41, 17.01; 20/28; fc1933 | 16.01, 16.99; 20/28; fc1933 | 16.47, 17.20; 20/28; fc1933 | 16.64, 17.23; 20/28; fc1933 | 14.43, 15.12; fc1933 | 14.46, 15.12; fc1933 |
