### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `v3` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | v3 | 31.9 (31.40) | 32.2 (31.04) | 0.4 |  |  |
|  |  | v4r | 51.8 (19.29) | 52.4 (19.09) | 0.5 | +62.7 % | gain |
| 4,096 | 4,032 on 0 | v3 | 123.7 (8.09) | 123.6 (8.09) | 0.1 |  |  |
|  |  | v4r | 198.6 (5.04) | 195.9 (5.10) | 2.7 | +59.6 % | gain |
| 8,192 | 4,612 on 3,516 | v3 | 119.1 (8.40) | 119.0 (8.40) | 0.1 |  |  |
|  |  | v4r | 219.9 (4.55) | 223.0 (4.48) | 3.1 | +86.0 % | gain |
| 16,384 | 12,804 on 3,516 | v3 | 119.5 (8.37) | 118.5 (8.44) | 1.0 |  |  |
|  |  | v4r | 289.3 (3.46) | 290.7 (3.44) | 1.3 | +143.7 % | gain |
| 32,768 | 16,900 on 15,804 | v3 | 90.4 (11.06) | 91.1 (10.98) | 0.7 |  |  |
|  |  | v4r | 229.1 (4.37) | 229.1 (4.36) | 0.0 | +152.4 % | gain |
| 49,152 | 16,900 on 32,188 | v3 | 70.3 (14.22) | 70.6 (14.17) | 0.2 |  |  |
|  |  | v4r | 180.7 (5.53) | 180.8 (5.53) | 0.1 | +156.5 % | gain |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `v3` | Class |
|---|---:|---|---|---:|---:|---|
| v3 | 55,248 | 91.2 (605.6 s) | 91.4 (604.3 s) | 0.2 |  |  |
| v4r | 55,248 | 218.5 (252.8 s) | 218.8 (252.6 s) | 0.2 | +139.4 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `v31` | v3 | 906 | 88 | 88 | 851 | +671 | 343 | 14.99 / 12.52 % | did not act |
| `v32` | v3 | 910 | 76 | 76 | 1,265 | +998 | 227 | 11.27 / 9.77 % | did not act |
| `v4r1` | v4r | 850 | 339 | 339 | 955 | +851 | 131 | 8.58 / 7.56 % | did not act |
| `v4r2` | v4r | 875 | 289 | 289 | 764 | +688 | 153 | 3.88 / 3.70 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v31` | v3 | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `v32` | v3 | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `v4r1` | v4r | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |
| `v4r2` | v4r | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 43,983 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `v31` | `v32` | `v4r1` | `v4r2` |
|---:|---|---|---|---|
| 0 | 25.74, 27.75; 127.5, 123.0 ms; 359/449; 08569d | 25.84, 27.75; 127.0, 123.0 ms; 359/449; 08569d | 22.90, 24.13; 102.5, 98.2 ms; 295/639; 2aadc9 | 22.89, 24.12; 102.6, 98.3 ms; 295/639; 2aadc9 |
| 4,096 | 14.85, 25.63; 236.1, 136.8 ms; 366/426; e99208 | 15.16, 24.51; 222.2, 137.4 ms; 360/446; 56caf0 | 28.13, 29.58; 115.2, 109.6 ms; 354/462; 765bab | 28.44, 29.60; 113.9, 109.5 ms; 354/462; 765bab |
| 8,192 | 24.72, 25.07; 154.6, 152.4 ms; 378/392; 5be5f3 | 20.80, 23.80; 171.0, 151.5 ms; 369/417; 8b3879 | 30.06, 31.59; 118.3, 114.1 ms; 369/421; 79c04b | 30.02, 31.63; 118.4, 114.0 ms; 369/421; 79c04b |
| 16,384 | 17.71, 20.16; 206.6, 181.4 ms; 372/408; 41cfc7 | 17.57, 19.76; 205.3, 182.5 ms; 370/420; aa1e65 | 30.39, 31.41; 122.1, 118.1 ms; 374/406; 95387c | 30.39, 31.39; 122.1, 118.2 ms; 374/406; 95387c |
| 32,768 | 11.21, 11.49; 248.3, 242.1 ms; 328/542; 49f0bb | 12.19, 12.60; 247.0, 241.9 ms; 343/501; d60347 | 21.27, 22.07; 130.8, 127.5 ms; 329/543; 39f0dc | 21.21, 22.07; 131.2, 127.5 ms; 329/543; 39f0dc |
| 49,152 | 10.58, 10.75; 306.3, 301.4 ms; 354/466; 2cf14f | 10.47, 11.55; 330.5, 299.5 ms; 364/432; a92b03 | 24.01, 24.51; 140.3, 137.4 ms; 360/450; fd3759 | 23.80, 24.49; 141.5, 137.5 ms; 360/450; fd3759 |
