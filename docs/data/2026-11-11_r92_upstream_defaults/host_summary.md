## `host` against `base`, decode and prompts

Timed loads, in their order: base1, host1, nopf1, base2, host2, nopf2, trace. Every value is one load.

- `base`: the preset as it is
- `host`: `LLAMA_MTP_H_DEV=0`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | base1 | 26.20, 28.14 → **27.14** | 36.85 | 125.25, 121.28 | 0.7996 | 3.346 | — | 31.9 (64 on 0) |
|  | base2 | 26.34, 28.04 → **27.16** | 36.81 | 124.61, 121.71 | 0.7996 | 3.346 | — | 32.8 (64 on 0) |
|  | host1 | 25.87, 27.72 → **26.76** | 37.37 | 126.88, 123.15 | 0.7996 | 3.346 | — | 32.9 (64 on 0) |
|  | host2 | 25.87, 27.77 → **26.79** | 37.33 | 126.86, 122.89 | 0.7996 | 3.346 | — | 32.1 (64 on 0) |
| 16,384 | base1 | 17.74, 18.43 → **18.08** | 55.32 | 185.03, 180.40 | 0.7778 | 3.303 | — | 131.8 (16,320 on 0) |
|  | base2 | 17.73, 18.38 → **18.05** | 55.40 | 185.11, 180.85 | 0.7778 | 3.303 | — | 132.1 (16,320 on 0) |
|  | host1 | 15.29, 18.26 → **16.64** | 60.08 | 214.65, 182.06 | 0.7778 | 3.303 | — | 126.3 (16,320 on 0) |
|  | host2 | 14.82, 18.24 → **16.35** | 61.14 | 221.46, 182.23 | 0.7778 | 3.303 | — | 124.5 (16,320 on 0) |
| 49,152 | base1 | 10.72, 11.43 → **11.06** | 90.39 | 318.49, 298.57 | 0.8265 | 3.413 | — | 82.8 (33,284 on 15,804) |
|  | base2 | 9.95, 11.45 → **10.65** | 93.91 | 343.01, 298.06 | 0.8265 | 3.413 | — | 82.8 (33,284 on 15,804) |
|  | host1 | 9.84, 11.37 → **10.55** | 94.77 | 346.83, 300.14 | 0.8265 | 3.413 | — | 80.4 (33,284 on 15,804) |
|  | host2 | 9.99, 11.37 → **10.64** | 94.01 | 341.62, 300.19 | 0.8265 | 3.413 | — | 80.4 (33,284 on 15,804) |

### `host` against `base`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `base` loads | `host` loads | Spread | Against `base` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 27.14, 27.16 | 26.76, 26.79 | 0.03 | -0.38 tok/s | -1.39 % | loss |
|  | decode ms a token | 36.85, 36.81 | 37.37, 37.33 | 0.04 | +0.52 ms | +1.41 % | loss |
|  | the step on pass 1, ms | 125.25, 124.61 | 126.88, 126.86 | 0.64 | +1.94 ms | +1.55 % | loss |
|  | the step on pass 2 (the repeated request), ms | 121.28, 121.71 | 123.15, 122.89 | 0.43 | +1.52 ms | +1.25 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.7996, 0.7996 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.346, 3.346 | 3.346, 3.346 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 31.9, 32.8 | 32.9, 32.1 | 0.9 | +0.2 tok/s | +0.60 % | negligible |
| 16,384 | decode tok/s (greedy, both passes) | 18.08, 18.05 | 16.64, 16.35 | 0.29 | -1.56 tok/s | -8.66 % | loss |
|  | decode ms a token | 55.32, 55.40 | 60.08, 61.14 | 1.06 | +5.25 ms | +9.49 % | loss |
|  | the step on pass 1, ms | 185.03, 185.11 | 214.65, 221.46 | 6.81 | +32.99 ms | +17.82 % | loss |
|  | the step on pass 2 (the repeated request), ms | 180.40, 180.85 | 182.06, 182.23 | 0.44 | +1.52 ms | +0.84 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.7778, 0.7778 | 0.7778, 0.7778 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.303, 3.303 | 3.303, 3.303 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 131.8, 132.1 | 126.3, 124.5 | 1.8 | -6.6 tok/s | -5.01 % | loss |
| 49,152 | decode tok/s (greedy, both passes) | 11.06, 10.65 | 10.55, 10.64 | 0.41 | -0.26 tok/s | -2.41 % | negligible |
|  | decode ms a token | 90.39, 93.91 | 94.77, 94.01 | 3.52 | +2.24 ms | +2.43 % | negligible |
|  | the step on pass 1, ms | 318.49, 343.01 | 346.83, 341.62 | 24.52 | +13.47 ms | +4.07 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 298.57, 298.06 | 300.14, 300.19 | 0.51 | +1.84 ms | +0.62 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.8265, 0.8265 | 0.8265, 0.8265 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.413, 3.413 | 3.413, 3.413 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 82.8, 82.8 | 80.4, 80.4 | 0.0 | -2.4 tok/s | -2.88 % | loss |
| all depths | decode tok/s (greedy, both passes) | 16.43, 16.12 | 15.61, 15.59 | 0.31 | -0.68 tok/s | -4.18 % | loss |
|  | decode ms a token | 60.85, 62.04 | 64.07, 64.16 | 1.19 | +2.67 ms | +4.35 % | loss |
|  | the step on pass 1, ms | 208.18, 215.95 | 227.93, 228.53 | 7.77 | +16.17 ms | +7.62 % | loss |
|  | the step on pass 2 (the repeated request), ms | 199.91, 200.04 | 201.61, 201.60 | 0.12 | +1.63 ms | +0.81 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.8009, 0.8009 | 0.8009, 0.8009 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.354, 3.354 | 3.354, 3.354 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 94.3, 94.4 | 91.3, 91.0 | 0.3 | -3.2 tok/s | -3.39 % | loss |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| base1 | 901 | 97 | 45 | +4 | 0.35 |
| base2 | 903 | 222 | 788 | +0 | 1.72 |
| host1 | 926 | 80 | 841 | +536 | 5.43 |
| host2 | 927 | 113 | 970 | +0 | 5.56 |
