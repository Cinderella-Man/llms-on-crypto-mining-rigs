### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `ub512` by the deep lane's rule. Beside: the loads of another run that read the same chunk on the same cached tokens, not classed. Gap closed: the share of the gap between `ub512` and `nodraft (the fourth run)` in ms a prompt token that the arm closes.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `ub512` | Class | Gap closed |
|---:|---|---|---|---|---:|---:|---|---:|
| 0 | 64 on 0 | ub512 | 32.7 (30.57) | 28.0 (35.70) | 4.7 |  |  |  |
|  |  | ub128 | 32.4 (30.84) | 32.7 (30.62) | 0.2 | +7.2 % | negligible |  |
|  |  | sp128 | 32.6 (30.69) | 32.1 (31.14) | 0.5 | +6.6 % | negligible |  |
|  |  | draft (the fourth run) (beside) | 32.8 (30.52) | 32.8 (30.45) | 0.1 | +8.0 % | beside |  |
|  |  | nodraft (the fourth run) (beside) | 33.5 (29.82) | 33.8 (29.57) | 0.3 | +10.9 % | beside |  |
| 16,384 | 16,320 on 0 | ub512 | 123.8 (8.07) | 123.7 (8.08) | 0.1 |  |  |  |
|  |  | ub128 | 118.2 (8.46) | 118.6 (8.43) | 0.3 | −4.3 % | loss |  |
|  |  | sp128 | 128.2 (7.80) | 128.3 (7.79) | 0.1 | +3.6 % | gain |  |
| 32,768 | 16,900 on 15,804 | ub512 | 90.2 (11.09) | 90.3 (11.08) | 0.1 |  |  |  |
|  | 16,516 on 16,188 | ub128 | 98.5 (10.15) | 98.5 (10.15) | 0.0 | +9.1 % | gain |  |
|  | 16,516 on 16,188 | sp128 | 133.8 (7.47) | 133.9 (7.47) | 0.0 | +48.3 % | gain |  |
| 49,152 | 16,900 on 32,188 | ub512 | 70.2 (14.25) | 70.0 (14.29) | 0.2 |  |  |  |
|  | 16,516 on 32,572 | ub128 | 80.5 (12.42) | 80.7 (12.38) | 0.2 | +15.1 % | gain | 31.6 % |
|  | 16,516 on 32,572 | sp128 | 131.3 (7.62) | 131.9 (7.58) | 0.6 | +87.8 % | gain | 112.9 % |
|  |  | draft (the fourth run) (beside) | 70.0 (14.29) | 70.1 (14.26) | 0.1 | −0.0 % | beside |  |
|  |  | nodraft (the fourth run) (beside) | 119.5 (8.37) | 119.6 (8.36) | 0.1 | +70.6 % | beside |  |

Not shown, as they read another chunk: draft (the fourth run) at 16,384 (8,708 on 7,612); nodraft (the fourth run) at 16,384 (8,708 on 7,612); draft (the fourth run) at 32,768 (16,388 on 16,316); nodraft (the fourth run) at 32,768 (16,388 on 16,316).

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `ub512` | Class |
|---|---:|---|---|---:|---:|---|
| ub512 | 50,120 | 89.5 (559.9 s) | 89.4 (560.6 s) | 0.1 |  |  |
| ub128 | 49,352 | 96.6 (510.8 s) | 96.8 (509.9 s) | 0.2 | +8.1 % | gain |
| sp128 | 49,352 | 131.1 (376.5 s) | 131.3 (375.8 s) | 0.2 | +46.7 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `ub512_1` | ub512 | 896 | 220 | 220 | 1,061 | +929 | 242 | 10.05 / 8.96 % | did not act |
| `ub512_2` | ub512 | 911 | 622 | 622 | 1,036 | +833 | 319 | 12.70 / 11.09 % | did not act |
| `ub128_1` | ub128 | 996 | 67 | 67 | 282 | +232 | 129 | 3.42 / 2.62 % | did not act |
| `ub128_2` | ub128 | 1,016 | 246 | 246 | 1,075 | +835 | 262 | 6.49 / 4.97 % | did not act |
| `sp128_1` | sp128 | 1,013 | 69 | 69 | 635 | +590 | 240 | 9.53 / 9.31 % | did not act |
| `sp128_2` | sp128 | 970 | 92 | 92 | 331 | +116 | 156 | 6.69 / 5.86 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ub512_1` | ub512 | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `ub512_2` | ub512 | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 44,019 |
| `ub128_1` | ub128 | 4,811 | 4,895 | 4,763 | 4,911 | 4,675 | 4,779 | 4,181 | 4,791 | 4,365 | 42,171 |
| `ub128_2` | ub128 | 4,811 | 4,895 | 4,763 | 4,911 | 4,675 | 4,779 | 4,181 | 4,791 | 4,365 | 42,171 |
| `sp128_1` | sp128 | 4,813 | 4,897 | 4,765 | 4,913 | 4,675 | 4,781 | 4,183 | 4,793 | 4,365 | 42,185 |
| `sp128_2` | sp128 | 4,813 | 4,897 | 4,765 | 4,913 | 4,675 | 4,781 | 4,183 | 4,793 | 4,365 | 42,185 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `ub512_1` | `ub512_2` | `ub128_1` | `ub128_2` | `sp128_1` | `sp128_2` |
|---:|---|---|---|---|---|---|
| 0 | 7.24, 15.72; 245.4, 113.1 ms; 14/46; 9da71b | 5.49, 15.71; 323.9, 113.1 ms; 14/46; 9da71b | 7.08, 15.72; 250.9, 113.1 ms; 14/46; 9da71b | 7.11, 15.71; 250.2, 113.2 ms; 14/46; 9da71b | 7.72, 8.10; 259.2, 282.4 ms; 17/35; 95afbd | 7.72, 8.09; 259.2, 282.5 ms; 17/35; 95afbd |
| 16,384 | 5.84, 15.60; 391.2, 146.6 ms; 18/30; db55ac | 5.58, 15.05; 409.4, 151.9 ms; 18/30; db55ac | 18.89, 20.49; 141.2, 130.2 ms; 20/30; af9d32 | 11.84, 12.73; 193.1, 179.6 ms; 18/32; db55ac | 18.80, 20.45; 141.9, 130.4 ms; 20/30; af9d32 | 13.57, 20.38; 196.5, 130.9 ms; 20/28; 983ba3 |
| 32,768 | 11.70, 13.42; 195.3, 170.3 ms; 18/30; 1ca520* | 15.07, 15.67; 151.7, 145.8 ms; 18/34; 2b721e | 5.41, 13.96; 369.5, 143.3 ms; 16/31; 325496 | 4.62, 11.89; 494.7, 192.2 ms; 18/29; 1ca520* | 7.83, 14.14; 255.6, 141.5 ms; 16/34; f46ec6 | 6.64, 7.82; 267.6, 255.7 ms; 15/40; 9622a9 |
| 49,152 | 16.59, 17.19; 160.8, 155.1 ms; 20/28; fc1933 | 16.68, 17.20; 159.9, 155.0 ms; 20/28; fc1933 | 16.45, 17.09; 162.1, 156.0 ms; 20/28; fc1933 | 14.79, 15.74; 180.3, 169.4 ms; 20/28; fc1933 | 11.17, 17.60; 238.7, 151.5 ms; 20/26; 89fc4f | 16.55, 17.17; 161.1, 155.3 ms; 20/28; fc1933 |
