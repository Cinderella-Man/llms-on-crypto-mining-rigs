### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm of this run is classed against `o4` by the deep lane's rule.

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `o4` | Class |
|---:|---|---|---|---|---:|---:|---|
| 0 | 64 on 0 | o4 | 40.9 (24.45) | 40.8 (24.50) | 0.1 |  |  |
|  |  | o4nk | 33.9 (29.46) | 33.2 (30.13) | 0.8 | −17.8 % | loss |
|  |  | o4ne | 40.8 (24.53) | 40.7 (24.55) | 0.0 | −0.3 % | loss |
| 4,096 | 4,032 on 0 | o4 | 170.7 (5.86) | 168.6 (5.93) | 2.1 |  |  |
|  |  | o4nk | 173.8 (5.75) | 172.5 (5.80) | 1.3 | +2.1 % | gain |
|  |  | o4ne | 172.5 (5.80) | 172.6 (5.79) | 0.1 | +1.7 % | gain |
| 8,192 | 4,612 on 3,516 | o4 | 158.2 (6.32) | 158.8 (6.30) | 0.6 |  |  |
|  |  | o4nk | 160.3 (6.24) | 161.3 (6.20) | 0.9 | +1.5 % | gain |
|  |  | o4ne | 162.4 (6.16) | 162.4 (6.16) | 0.0 | +2.5 % | gain |
| 16,384 | 12,804 on 3,516 | o4 | 203.0 (4.93) | 203.1 (4.92) | 0.1 |  |  |
|  |  | o4nk | 206.5 (4.84) | 206.1 (4.85) | 0.5 | +1.6 % | gain |
|  |  | o4ne | 206.3 (4.85) | 206.1 (4.85) | 0.1 | +1.6 % | gain |
| 32,768 | 16,900 on 15,804 | o4 | 157.8 (6.34) | 157.8 (6.34) | 0.0 |  |  |
|  |  | o4nk | 162.5 (6.15) | 162.8 (6.14) | 0.3 | +3.1 % | gain |
|  |  | o4ne | 162.3 (6.16) | 162.1 (6.17) | 0.1 | +2.8 % | gain |
| 48,640 | 16,388 on 32,188 | o4 | 137.2 (7.29) | 137.2 (7.29) | 0.0 |  |  |
|  |  | o4nk | 141.5 (7.07) | 141.7 (7.06) | 0.3 | +3.2 % | gain |
|  |  | o4ne | 140.6 (7.11) | 140.6 (7.11) | 0.0 | +2.5 % | gain |

All primes of a load together: the tokens they read, and tok/s (seconds):

| Arm | Tokens read | Load 1 | Load 2 | Spread, tok/s | Against `o4` | Class |
|---|---:|---|---|---:|---:|---|
| o4 | 54,736 | 159.9 (342.3 s) | 159.8 (342.6 s) | 0.1 |  |  |
| o4nk | 54,736 | 164.0 (333.8 s) | 164.1 (333.5 s) | 0.1 | +2.6 % | gain |
| o4ne | 54,736 | 163.6 (334.5 s) | 163.6 (334.6 s) | 0.1 | +2.4 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second, over the whole load (from the server's start to its stop); usable is MemAvailable minus CmaFree. Swap moved during the primes: the sum of the rises in swap in use while a prime request ran. The most written to swap in 10 s is one of the watchdog's own limits (by default it stops the server above 600 MiB).

| Load | Arm | MemAvailable after loading, MiB | Lowest MemAvailable | Lowest usable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full | The watchdog |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `o1` | o4 | 1,506 | 1,228 | 1,228 | 338 | +0 | 257 | 9.78 / 8.17 % | did not act |
| `o2` | o4 | 1,485 | 1,248 | 1,248 | 338 | +0 | 232 | 25.08 / 22.27 % | did not act |
| `n1` | o4nk | 1,494 | 1,236 | 1,236 | 338 | +0 | 249 | 9.31 / 7.84 % | did not act |
| `n2` | o4nk | 1,504 | 1,240 | 1,240 | 338 | +0 | 228 | 7.49 / 6.77 % | did not act |
| `e1` | o4ne | 1,497 | 1,245 | 1,245 | 337 | +0 | 252 | 7.73 / 6.56 % | did not act |
| `e2` | o4ne | 1,508 | 1,237 | 1,237 | 339 | +0 | 229 | 18.88 / 17.35 % | did not act |

Each card's memory after loading, MiB:

| Load | Arm | Card 0 | Card 1 | Card 2 | Card 3 | Card 4 | Card 5 | Card 6 | Card 7 | Card 8 | All |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `o1` | o4 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `o2` | o4 | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `n1` | o4nk | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `n2` | o4nk | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `e1` | o4ne | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |
| `e2` | o4ne | 5,603 | 5,635 | 5,549 | 5,635 | 5,243 | 5,351 | 5,513 | 5,601 | 5,009 | 49,139 |

### Generation, a check

Per load and depth: tok/s on pass 1 and pass 2; ms a step on pass 1 and pass 2; the draft tokens accepted of drafted on both passes; the first six characters of the reply's hash on pass 1 (a star when pass 2 wrote another reply).

| Depth (tokens) | `o1` | `o2` | `n1` | `n2` | `e1` | `e2` |
|---:|---|---|---|---|---|---|
| 0 | 18.34, 18.69; 54.5, 53.5 ms; 0/0; b82797 | 18.28, 18.71; 54.7, 53.5 ms; 0/0; b82797 | 17.11, 17.24; 58.4, 58.0 ms; 0/0; b82797 | 17.18, 17.37; 58.2, 57.6 ms; 0/0; b82797 | 18.36, 18.70; 54.5, 53.5 ms; 0/0; b82797 | 18.35, 18.71; 54.5, 53.5 ms; 0/0; b82797 |
| 4,096 | 16.93, 17.03; 59.1, 58.7 ms; 0/0; 5eb4b3 | 16.92, 17.04; 59.1, 58.7 ms; 0/0; 5eb4b3 | 16.88, 14.14; 59.2, 70.7 ms; 0/0; 5eb4b3 | 17.39, 14.59; 57.5, 68.5 ms; 0/0; 5eb4b3 | 16.97, 17.05; 58.9, 58.7 ms; 0/0; 5eb4b3 | 16.99, 17.04; 58.9, 58.7 ms; 0/0; 5eb4b3 |
| 8,192 | 15.56, 15.63; 64.3, 64.0 ms; 0/0; eea8be | 15.58, 15.65; 64.2, 63.9 ms; 0/0; eea8be | 16.32, 15.37; 61.3, 65.1 ms; 0/0; eea8be | 16.17, 15.31; 61.8, 65.3 ms; 0/0; eea8be | 15.63, 15.64; 64.0, 63.9 ms; 0/0; eea8be | 15.63, 15.64; 64.0, 63.9 ms; 0/0; eea8be |
| 16,384 | 13.41, 13.45; 74.6, 74.3 ms; 0/0; d7e7c7 | 13.40, 13.44; 74.6, 74.4 ms; 0/0; d7e7c7 | 13.23, 12.14; 75.6, 82.4 ms; 0/0; d7e7c7 | 13.53, 12.31; 73.9, 81.2 ms; 0/0; d7e7c7 | 13.45, 13.45; 74.4, 74.4 ms; 0/0; d7e7c7 | 13.44, 13.44; 74.4, 74.4 ms; 0/0; d7e7c7 |
| 32,768 | 10.45, 10.48; 95.7, 95.4 ms; 0/0; 576f56 | 10.46, 10.49; 95.6, 95.3 ms; 0/0; 576f56 | 10.99, 9.32; 91.0, 107.3 ms; 0/0; 576f56 | 11.20, 9.23; 89.3, 108.4 ms; 0/0; 576f56 | 10.49, 10.50; 95.3, 95.2 ms; 0/0; 576f56 | 10.49, 10.50; 95.3, 95.2 ms; 0/0; 576f56 |
| 48,640 | 8.65, 8.67; 115.6, 115.4 ms; 0/0; d340d8 | 8.67, 8.68; 115.4, 115.2 ms; 0/0; d340d8 | 9.44, 7.76; 105.9, 128.9 ms; 0/0; d340d8 | 9.80, 7.91; 102.0, 126.5 ms; 0/0; d340d8 | 8.69, 8.69; 115.1, 115.1 ms; 0/0; d340d8 | 8.69, 8.69; 115.1, 115.1 ms; 0/0; d340d8 |
