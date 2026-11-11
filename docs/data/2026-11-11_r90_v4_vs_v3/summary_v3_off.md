## `off` against `v3`, decode and prompts

Timed loads, in their order: v31, gains1, v32, gains2, off1, off2. Every value is one load.

- `v3`: the preset as it is
- `off`: the preset as it is

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | v31 | 26.27, 27.60 → **26.92** | 37.15 | 124.96, 123.66 | 0.7996 | 3.346 | 17.03, 0.4506, 2.327, 136.65 | 32.8 (64 on 0) |
|  | v32 | 25.71, 27.59 → **26.62** | 37.57 | 127.65, 123.70 | 0.7996 | 3.346 | 16.12, 0.4506, 2.327, 144.40 | 31.7 (64 on 0) |
|  | off1 | 26.62, 28.06 → **27.32** | 36.60 | 123.27, 121.64 | 0.7996 | 3.346 | 16.65, 0.4506, 2.327, 139.75 | 33.1 (64 on 0) |
|  | off2 | 26.07, 27.97 → **26.99** | 37.05 | 125.89, 122.03 | 0.7996 | 3.346 | 16.65, 0.4506, 2.327, 139.74 | 32.8 (64 on 0) |
| 8,192 | v31 | 16.27, 24.11 → **19.43** | 51.47 | 234.86, 158.48 | 0.9643 | 3.821 | 19.33, 0.7160, 3.122, 161.49 | 124.6 (8,128 on 0) |
|  | v32 | 16.32, 23.62 → **19.30** | 51.80 | 217.84, 152.65 | 0.8849 | 3.580 | 17.11, 0.6007, 2.783, 162.61 | 125.7 (8,128 on 0) |
|  | off1 | 14.28, 24.37 → **18.01** | 55.52 | 256.03, 152.27 | 0.9075 | 3.683 | 16.10, 0.5288, 2.560, 159.00 | 133.0 (8,128 on 0) |
|  | off2 | 20.92, 24.29 → **22.48** | 44.49 | 174.85, 152.73 | 0.9075 | 3.683 | 15.83, 0.5288, 2.560, 161.71 | 137.7 (8,128 on 0) |
| 16,384 | v31 | 17.28, 19.59 → **18.36** | 54.46 | 208.69, 184.07 | 0.8810 | 3.606 | 17.57, 0.7706, 3.282, 186.82 | 108.5 (8,708 on 7,612) |
|  | v32 | 14.37, 19.68 → **16.61** | 60.19 | 254.45, 185.79 | 0.8986 | 3.657 | 16.10, 0.7597, 3.241, 201.23 | 108.4 (8,708 on 7,612) |
|  | off1 | 15.29, 18.34 → **16.68** | 59.96 | 214.60, 181.30 | 0.7778 | 3.303 | 17.86, 0.7807, 3.282, 183.76 | 113.3 (8,708 on 7,612) |
|  | off2 | 14.15, 18.35 → **15.98** | 62.58 | 231.94, 181.18 | 0.7778 | 3.303 | 17.53, 0.7807, 3.282, 187.25 | 113.9 (8,708 on 7,612) |
| 32,768 | v31 | 11.13, 12.06 → **11.58** | 86.38 | 264.26, 244.09 | 0.6576 | 2.943 | 10.42, 0.5505, 2.612, 250.65 | 90.0 (16,388 on 16,316) |
|  | v32 | 10.42, 11.28 → **10.83** | 92.32 | 261.33, 244.11 | 0.5898 | 2.738 | 8.73, 0.4017, 2.188, 250.63 | 90.1 (16,388 on 16,316) |
|  | off1 | 10.79, 12.22 → **11.46** | 87.25 | 269.51, 240.89 | 0.6595 | 2.926 | 10.68, 0.5521, 2.639, 247.08 | 93.8 (16,388 on 16,316) |
|  | off2 | 10.96, 12.21 → **11.55** | 86.59 | 265.46, 241.08 | 0.6595 | 2.926 | 10.67, 0.5521, 2.639, 247.28 | 93.3 (16,388 on 16,316) |
| 49,152 | v31 | 11.60, 11.76 → **11.68** | 85.60 | 306.48, 302.25 | 0.8762 | 3.556 | 8.71, 0.5771, 2.695, 309.56 | 69.3 (16,900 on 32,188) |
|  | v32 | 10.02, 10.72 → **10.36** | 96.54 | 319.27, 302.37 | 0.7591 | 3.220 | 10.57, 0.7629, 3.241, 306.58 | 69.5 (16,900 on 32,188) |
|  | off1 | 11.24, 11.40 → **11.32** | 88.32 | 303.56, 299.38 | 0.8265 | 3.413 | 10.91, 0.7885, 3.325, 304.66 | 71.8 (16,900 on 32,188) |
|  | off2 | 11.22, 11.40 → **11.31** | 88.45 | 304.26, 299.53 | 0.8265 | 3.413 | 10.92, 0.7885, 3.325, 304.60 | 71.5 (16,900 on 32,188) |

### `off` against `v3`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `v3` loads | `off` loads | Spread | Against `v3` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 26.92, 26.62 | 27.32, 26.99 | 0.34 | +0.39 tok/s | +1.45 % | gain |
|  | decode ms a token | 37.15, 37.57 | 36.60, 37.05 | 0.45 | -0.53 ms | -1.43 % | gain |
|  | the step on pass 1, ms | 124.96, 127.65 | 123.27, 125.89 | 2.69 | -1.72 ms | -1.36 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 123.66, 123.70 | 121.64, 122.03 | 0.38 | -1.85 ms | -1.49 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.7996, 0.7996 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.346, 3.346 | 3.346, 3.346 | 0.000 | +0.000 | +0.00 % | negligible |
|  | sampler pass tok/s | 17.03, 16.12 | 16.65, 16.65 | 0.91 | +0.08 tok/s | +0.48 % | negligible |
|  | sampler pass share accepted | 0.4506, 0.4506 | 0.4506, 0.4506 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | sampler pass tokens a step | 2.327, 2.327 | 2.327, 2.327 | 0.000 | +0.000 | +0.00 % | negligible |
|  | sampler pass ms a step | 136.65, 144.40 | 139.75, 139.74 | 7.75 | -0.78 ms | -0.55 % | negligible |
|  | prompt tok/s (the prime) | 32.8, 31.7 | 33.1, 32.8 | 1.1 | +0.7 tok/s | +2.17 % | negligible |
| 8,192 | decode tok/s (greedy, both passes) | 19.43, 19.30 | 18.01, 22.48 | 4.47 | +0.88 tok/s | +4.53 % | negligible |
|  | decode ms a token | 51.47, 51.80 | 55.52, 44.49 | 11.04 | -1.63 ms | -3.16 % | negligible |
|  | the step on pass 1, ms | 234.86, 217.84 | 256.03, 174.85 | 81.19 | -10.91 ms | -4.82 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 158.48, 152.65 | 152.27, 152.73 | 5.83 | -3.07 ms | -1.97 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.9643, 0.8849 | 0.9075, 0.9075 | 0.0794 | -0.0170 | -1.84 % | negligible |
|  | tokens a step (greedy) | 3.821, 3.580 | 3.683, 3.683 | 0.240 | -0.017 | -0.46 % | negligible |
|  | sampler pass tok/s | 19.33, 17.11 | 16.10, 15.83 | 2.22 | -2.26 tok/s | -12.38 % | loss |
|  | sampler pass share accepted | 0.7160, 0.6007 | 0.5288, 0.5288 | 0.1153 | -0.1296 | -19.68 % | loss |
|  | sampler pass tokens a step | 3.122, 2.783 | 2.560, 2.560 | 0.339 | -0.392 | -13.29 % | loss |
|  | sampler pass ms a step | 161.49, 162.61 | 159.00, 161.71 | 2.71 | -1.70 ms | -1.05 % | negligible |
|  | prompt tok/s (the prime) | 124.6, 125.7 | 133.0, 137.7 | 4.7 | +10.2 tok/s | +8.16 % | gain |
| 16,384 | decode tok/s (greedy, both passes) | 18.36, 16.61 | 16.68, 15.98 | 1.75 | -1.16 tok/s | -6.63 % | negligible |
|  | decode ms a token | 54.46, 60.19 | 59.96, 62.58 | 5.72 | +3.94 ms | +6.88 % | negligible |
|  | the step on pass 1, ms | 208.69, 254.45 | 214.60, 231.94 | 45.76 | -8.30 ms | -3.58 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 184.07, 185.79 | 181.30, 181.18 | 1.72 | -3.69 ms | -2.00 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8810, 0.8986 | 0.7778, 0.7778 | 0.0176 | -0.1120 | -12.58 % | loss |
|  | tokens a step (greedy) | 3.606, 3.657 | 3.303, 3.303 | 0.052 | -0.328 | -9.04 % | loss |
|  | sampler pass tok/s | 17.57, 16.10 | 17.86, 17.53 | 1.47 | +0.86 tok/s | +5.10 % | negligible |
|  | sampler pass share accepted | 0.7706, 0.7597 | 0.7807, 0.7807 | 0.0109 | +0.0156 | +2.04 % | gain |
|  | sampler pass tokens a step | 3.282, 3.241 | 3.282, 3.282 | 0.042 | +0.021 | +0.64 % | negligible |
|  | sampler pass ms a step | 186.82, 201.23 | 183.76, 187.25 | 14.42 | -8.52 ms | -4.39 % | negligible |
|  | prompt tok/s (the prime) | 108.5, 108.4 | 113.3, 113.9 | 0.6 | +5.1 tok/s | +4.69 % | gain |
| 32,768 | decode tok/s (greedy, both passes) | 11.58, 10.83 | 11.46, 11.55 | 0.74 | +0.30 tok/s | +2.68 % | negligible |
|  | decode ms a token | 86.38, 92.32 | 87.25, 86.59 | 5.94 | -2.43 ms | -2.72 % | negligible |
|  | the step on pass 1, ms | 264.26, 261.33 | 269.51, 265.46 | 4.05 | +4.69 ms | +1.78 % | loss |
|  | the step on pass 2 (the repeated request), ms | 244.09, 244.11 | 240.89, 241.08 | 0.19 | -3.11 ms | -1.28 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.6576, 0.5898 | 0.6595, 0.6595 | 0.0678 | +0.0358 | +5.74 % | negligible |
|  | tokens a step (greedy) | 2.943, 2.738 | 2.926, 2.926 | 0.205 | +0.085 | +3.01 % | negligible |
|  | sampler pass tok/s | 10.42, 8.73 | 10.68, 10.67 | 1.69 | +1.10 tok/s | +11.50 % | negligible |
|  | sampler pass share accepted | 0.5505, 0.4017 | 0.5521, 0.5521 | 0.1488 | +0.0760 | +15.95 % | negligible |
|  | sampler pass tokens a step | 2.612, 2.188 | 2.639, 2.639 | 0.424 | +0.239 | +9.96 % | negligible |
|  | sampler pass ms a step | 250.65, 250.63 | 247.08, 247.28 | 0.20 | -3.46 ms | -1.38 % | gain |
|  | prompt tok/s (the prime) | 90.0, 90.1 | 93.8, 93.3 | 0.5 | +3.5 tok/s | +3.86 % | gain |
| 49,152 | decode tok/s (greedy, both passes) | 11.68, 10.36 | 11.32, 11.31 | 1.32 | +0.29 tok/s | +2.67 % | negligible |
|  | decode ms a token | 85.60, 96.54 | 88.32, 88.45 | 10.94 | -2.69 ms | -2.95 % | negligible |
|  | the step on pass 1, ms | 306.48, 319.27 | 303.56, 304.26 | 12.79 | -8.97 ms | -2.87 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 302.25, 302.37 | 299.38, 299.53 | 0.15 | -2.86 ms | -0.94 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8762, 0.7591 | 0.8265, 0.8265 | 0.1171 | +0.0088 | +1.08 % | negligible |
|  | tokens a step (greedy) | 3.556, 3.220 | 3.413, 3.413 | 0.335 | +0.025 | +0.75 % | negligible |
|  | sampler pass tok/s | 8.71, 10.57 | 10.91, 10.92 | 1.86 | +1.28 tok/s | +13.24 % | negligible |
|  | sampler pass share accepted | 0.5771, 0.7629 | 0.7885, 0.7885 | 0.1859 | +0.1186 | +17.69 % | negligible |
|  | sampler pass tokens a step | 2.695, 3.241 | 3.325, 3.325 | 0.546 | +0.357 | +12.03 % | negligible |
|  | sampler pass ms a step | 309.56, 306.58 | 304.66, 304.60 | 2.98 | -3.44 ms | -1.12 % | gain |
|  | prompt tok/s (the prime) | 69.3, 69.5 | 71.8, 71.5 | 0.3 | +2.2 tok/s | +3.23 % | gain |
| all depths | decode tok/s (greedy, both passes) | 15.87, 14.77 | 15.26, 15.67 | 1.09 | +0.14 tok/s | +0.92 % | negligible |
|  | decode ms a token | 63.01, 67.68 | 65.53, 63.83 | 4.67 | -0.67 ms | -1.02 % | negligible |
|  | the step on pass 1, ms | 227.62, 237.46 | 233.32, 221.93 | 11.39 | -4.92 ms | -2.11 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 204.19, 205.44 | 201.05, 201.25 | 1.25 | -3.67 ms | -1.79 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8260, 0.7744 | 0.7884, 0.7884 | 0.0516 | -0.0118 | -1.48 % | negligible |
|  | tokens a step (greedy) | 3.427, 3.274 | 3.316, 3.316 | 0.153 | -0.034 | -1.02 % | negligible |
|  | sampler pass tok/s | 13.22, 12.76 | 13.76, 13.68 | 0.46 | +0.73 tok/s | +5.59 % | gain |
|  | sampler pass share accepted | 0.5990, 0.5703 | 0.6006, 0.6006 | 0.0287 | +0.0159 | +2.73 % | negligible |
|  | sampler pass tokens a step | 2.765, 2.683 | 2.771, 2.771 | 0.081 | +0.047 | +1.71 % | negligible |
|  | sampler pass ms a step | 209.11, 210.24 | 201.36, 202.57 | 1.21 | -7.71 ms | -3.68 % | gain |
|  | prompt tok/s (the prime) | 87.7, 88.0 | 91.5, 91.5 | 0.2 | +3.7 tok/s | +4.17 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| v31 | 1178 | 207 | 975 | +20 | 11.80 |
| v32 | 1193 | 80 | 1,151 | +796 | 17.22 |
| off1 | 1157 | 310 | 1,085 | +785 | 22.45 |
| off2 | 1148 | 81 | 207 | +168 | 2.91 |
