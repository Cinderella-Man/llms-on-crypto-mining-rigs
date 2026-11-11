## `nopf` against `base`, decode and prompts

Timed loads, in their order: base1, host1, nopf1, base2, host2, nopf2, trace. Every value is one load.

- `base`: the preset as it is
- `nopf`: `GGML_SCHED_PREFETCH=0`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | base1 | 26.20, 28.14 → **27.14** | 36.85 | 125.25, 121.28 | 0.7996 | 3.346 | — | 31.9 (64 on 0) |
|  | base2 | 26.34, 28.04 → **27.16** | 36.81 | 124.61, 121.71 | 0.7996 | 3.346 | — | 32.8 (64 on 0) |
|  | nopf1 | 26.05, 28.05 → **27.01** | 37.02 | 126.01, 121.71 | 0.7996 | 3.346 | — | 31.4 (64 on 0) |
|  | nopf2 | 26.18, 28.15 → **27.13** | 36.86 | 125.37, 121.25 | 0.7996 | 3.346 | — | 32.6 (64 on 0) |
| 16,384 | base1 | 17.74, 18.43 → **18.08** | 55.32 | 185.03, 180.40 | 0.7778 | 3.303 | — | 131.8 (16,320 on 0) |
|  | base2 | 17.73, 18.38 → **18.05** | 55.40 | 185.11, 180.85 | 0.7778 | 3.303 | — | 132.1 (16,320 on 0) |
|  | nopf1 | 17.77, 18.39 → **18.08** | 55.32 | 184.69, 180.76 | 0.7778 | 3.303 | — | 132.1 (16,320 on 0) |
|  | nopf2 | 17.72, 18.40 → **18.05** | 55.40 | 185.24, 180.71 | 0.7778 | 3.303 | — | 131.5 (16,320 on 0) |
| 49,152 | base1 | 10.72, 11.43 → **11.06** | 90.39 | 318.49, 298.57 | 0.8265 | 3.413 | — | 82.8 (33,284 on 15,804) |
|  | base2 | 9.95, 11.45 → **10.65** | 93.91 | 343.01, 298.06 | 0.8265 | 3.413 | — | 82.8 (33,284 on 15,804) |
|  | nopf1 | 10.75, 11.44 → **11.08** | 90.23 | 317.59, 298.41 | 0.8265 | 3.413 | — | 83.3 (33,284 on 15,804) |
|  | nopf2 | 10.70, 11.44 → **11.06** | 90.44 | 319.13, 298.30 | 0.8265 | 3.413 | — | 82.6 (33,284 on 15,804) |

### `nopf` against `base`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `base` loads | `nopf` loads | Spread | Against `base` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 27.14, 27.16 | 27.01, 27.13 | 0.12 | -0.08 tok/s | -0.30 % | negligible |
|  | decode ms a token | 36.85, 36.81 | 37.02, 36.86 | 0.17 | +0.11 ms | +0.30 % | negligible |
|  | the step on pass 1, ms | 125.25, 124.61 | 126.01, 125.37 | 0.64 | +0.75 ms | +0.60 % | loss |
|  | the step on pass 2 (the repeated request), ms | 121.28, 121.71 | 121.71, 121.25 | 0.46 | -0.02 ms | -0.02 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.7996, 0.7996 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.346, 3.346 | 3.346, 3.346 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 31.9, 32.8 | 31.4, 32.6 | 1.2 | -0.3 tok/s | -1.02 % | negligible |
| 16,384 | decode tok/s (greedy, both passes) | 18.08, 18.05 | 18.08, 18.05 | 0.03 | -0.00 tok/s | -0.00 % | negligible |
|  | decode ms a token | 55.32, 55.40 | 55.32, 55.40 | 0.08 | +0.00 ms | +0.00 % | negligible |
|  | the step on pass 1, ms | 185.03, 185.11 | 184.69, 185.24 | 0.55 | -0.10 ms | -0.06 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 180.40, 180.85 | 180.76, 180.71 | 0.44 | +0.11 ms | +0.06 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.7778, 0.7778 | 0.7778, 0.7778 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.303, 3.303 | 3.303, 3.303 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 131.8, 132.1 | 132.1, 131.5 | 0.6 | -0.2 tok/s | -0.12 % | negligible |
| 49,152 | decode tok/s (greedy, both passes) | 11.06, 10.65 | 11.08, 11.06 | 0.41 | +0.21 tok/s | +1.97 % | negligible |
|  | decode ms a token | 90.39, 93.91 | 90.23, 90.44 | 3.52 | -1.81 ms | -1.96 % | negligible |
|  | the step on pass 1, ms | 318.49, 343.01 | 317.59, 319.13 | 24.52 | -12.39 ms | -3.75 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 298.57, 298.06 | 298.41, 298.30 | 0.51 | +0.04 ms | +0.01 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.8265, 0.8265 | 0.8265, 0.8265 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.413, 3.413 | 3.413, 3.413 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 82.8, 82.8 | 83.3, 82.6 | 0.7 | +0.2 tok/s | +0.18 % | negligible |
| all depths | decode tok/s (greedy, both passes) | 16.43, 16.12 | 16.43, 16.42 | 0.31 | +0.15 tok/s | +0.92 % | negligible |
|  | decode ms a token | 60.85, 62.04 | 60.86, 60.90 | 1.19 | -0.57 ms | -0.92 % | negligible |
|  | the step on pass 1, ms | 208.18, 215.95 | 208.03, 208.49 | 7.77 | -3.80 ms | -1.79 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 199.91, 200.04 | 200.12, 199.91 | 0.21 | +0.04 ms | +0.02 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.8009, 0.8009 | 0.8009, 0.8009 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.354, 3.354 | 3.354, 3.354 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 94.3, 94.4 | 94.8, 94.1 | 0.7 | +0.1 tok/s | +0.11 % | negligible |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| base1 | 901 | 97 | 45 | +4 | 0.35 |
| base2 | 903 | 222 | 788 | +0 | 1.72 |
| nopf1 | 903 | 752 | 848 | +0 | 4.34 |
| nopf2 | 900 | 117 | 44 | +7 | 0.30 |
