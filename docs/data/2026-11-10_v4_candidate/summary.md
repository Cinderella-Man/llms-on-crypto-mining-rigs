## `v4c` against `rel`, decode and prompts

Timed loads, in their order: rel1, v4c1, rel2, v4c2. Every value is one load.

- `rel`: the preset as it is
- `v4c`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | rel1 | 26.18, 27.68 → **26.91** | 37.16 | 125.38, 123.31 | 0.7996 | 3.346 | 17.04, 0.4506, 2.327, 136.59 | 32.9 (64 on 0) |
|  | rel2 | 23.35, 27.59 → **25.29** | 39.54 | 140.58, 123.74 | 0.7996 | 3.346 | 15.22, 0.4506, 2.327, 152.91 | 31.2 (64 on 0) |
|  | v4c1 | 25.73, 27.60 → **26.63** | 37.55 | 127.54, 123.68 | 0.7996 | 3.346 | 15.22, 0.3943, 2.169, 142.54 | 48.1 (64 on 0) |
|  | v4c2 | 25.74, 27.59 → **26.64** | 37.54 | 127.49, 123.70 | 0.7996 | 3.346 | 15.27, 0.3943, 2.169, 142.10 | 48.3 (64 on 0) |
| 8,192 | rel1 | 12.47, 20.05 → **15.37** | 65.05 | 233.36, 153.86 | 0.6779 | 2.994 | 17.34, 0.6693, 2.977, 171.71 | 128.5 (8,128 on 0) |
|  | rel2 | 22.45, 25.01 → **23.66** | 42.26 | 170.19, 152.76 | 0.9643 | 3.821 | 17.83, 0.6326, 2.876, 161.36 | 130.1 (8,128 on 0) |
|  | v4c1 | 20.08, 24.72 → **22.16** | 45.13 | 174.62, 143.85 | 0.8555 | 3.531 | 19.89, 0.6840, 3.012, 151.42 | 127.9 (8,128 on 0) |
|  | v4c2 | 18.48, 23.31 → **20.62** | 48.50 | 177.58, 144.48 | 0.7851 | 3.325 | 18.87, 0.7149, 3.084, 163.48 | 130.0 (8,128 on 0) |
| 16,384 | rel1 | 16.35, 19.42 → **17.75** | 56.33 | 214.52, 183.08 | 0.8635 | 3.531 | 16.49, 0.7033, 3.084, 187.04 | 109.6 (8,708 on 7,612) |
|  | rel2 | 19.16, 19.39 → **19.27** | 51.88 | 185.60, 183.34 | 0.8638 | 3.556 | 18.98, 0.8472, 3.507, 184.74 | 110.0 (8,708 on 7,612) |
|  | v4c1 | 20.14, 22.66 → **21.32** | 46.90 | 169.51, 150.66 | 0.8265 | 3.413 | 24.11, 0.9118, 3.657, 151.66 | 123.9 (8,708 on 7,612) |
|  | v4c2 | 23.87, 24.11 → **23.99** | 41.68 | 153.21, 151.67 | 0.9029 | 3.657 | 21.93, 0.8036, 3.368, 153.56 | 128.6 (8,708 on 7,612) |
| 32,768 | rel1 | 11.65, 12.29 → **11.96** | 83.60 | 249.66, 245.10 | 0.6686 | 2.960 | 11.27, 0.6074, 2.783, 247.00 | 90.8 (16,388 on 16,316) |
|  | rel2 | 11.01, 11.66 → **11.32** | 88.32 | 250.08, 244.01 | 0.6104 | 2.798 | 10.11, 0.5168, 2.510, 248.20 | 90.5 (16,388 on 16,316) |
|  | v4c1 | 16.34, 17.00 → **16.66** | 60.01 | 170.26, 165.51 | 0.6173 | 2.798 | 15.12, 0.5448, 2.612, 172.75 | 103.6 (16,388 on 16,316) |
|  | v4c2 | 14.67, 15.42 → **15.03** | 66.51 | 172.80, 166.02 | 0.5245 | 2.547 | 16.22, 0.6029, 2.783, 171.51 | 103.7 (16,388 on 16,316) |
| 49,152 | rel1 | 10.27, 10.54 → **10.40** | 96.11 | 307.76, 303.51 | 0.7358 | 3.180 | 9.65, 0.6627, 2.943, 304.92 | 69.9 (16,900 on 32,188) |
|  | rel2 | 10.55, 10.85 → **10.70** | 93.48 | 307.18, 302.47 | 0.7701 | 3.261 | 9.78, 0.6786, 3.012, 308.03 | 69.8 (16,900 on 32,188) |
|  | v4c1 | 18.93, 19.25 → **19.08** | 52.40 | 185.29, 182.21 | 0.8472 | 3.507 | 17.84, 0.7807, 3.282, 183.96 | 83.5 (16,900 on 32,188) |
|  | v4c2 | 18.14, 18.54 → **18.34** | 54.53 | 185.72, 181.66 | 0.8036 | 3.368 | 18.56, 0.8227, 3.413, 183.90 | 83.1 (16,900 on 32,188) |

### `v4c` against `rel`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `rel` loads | `v4c` loads | Spread | Against `rel` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 26.91, 25.29 | 26.63, 26.64 | 1.62 | +0.54 tok/s | +2.06 % | negligible |
|  | decode ms a token | 37.16, 39.54 | 37.55, 37.54 | 2.38 | -0.81 ms | -2.11 % | negligible |
|  | the step on pass 1, ms | 125.38, 140.58 | 127.54, 127.49 | 15.21 | -5.47 ms | -4.11 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 123.31, 123.74 | 123.68, 123.70 | 0.43 | +0.17 ms | +0.13 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.7996, 0.7996 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.346, 3.346 | 3.346, 3.346 | 0.000 | +0.000 | +0.00 % | negligible |
|  | sampler pass tok/s | 17.04, 15.22 | 15.22, 15.27 | 1.82 | -0.88 tok/s | -5.49 % | negligible |
|  | sampler pass share accepted | 0.4506, 0.4506 | 0.3943, 0.3943 | 0.0000 | -0.0563 | -12.50 % | loss |
|  | sampler pass tokens a step | 2.327, 2.327 | 2.169, 2.169 | 0.000 | -0.158 | -6.78 % | loss |
|  | sampler pass ms a step | 136.59, 152.91 | 142.54, 142.10 | 16.32 | -2.44 ms | -1.68 % | negligible |
|  | prompt tok/s (the prime) | 32.9, 31.2 | 48.1, 48.3 | 1.6 | +16.2 tok/s | +50.37 % | gain |
| 8,192 | decode tok/s (greedy, both passes) | 15.37, 23.66 | 22.16, 20.62 | 8.29 | +1.87 tok/s | +9.59 % | negligible |
|  | decode ms a token | 65.05, 42.26 | 45.13, 48.50 | 22.79 | -6.84 ms | -12.75 % | negligible |
|  | the step on pass 1, ms | 233.36, 170.19 | 174.62, 177.58 | 63.17 | -25.68 ms | -12.72 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 153.86, 152.76 | 143.85, 144.48 | 1.10 | -9.14 ms | -5.96 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.6779, 0.9643 | 0.8555, 0.7851 | 0.2864 | -0.0008 | -0.10 % | negligible |
|  | tokens a step (greedy) | 2.994, 3.821 | 3.531, 3.325 | 0.827 | +0.020 | +0.60 % | negligible |
|  | sampler pass tok/s | 17.34, 17.83 | 19.89, 18.87 | 1.02 | +1.80 tok/s | +10.23 % | gain |
|  | sampler pass share accepted | 0.6693, 0.6326 | 0.6840, 0.7149 | 0.0367 | +0.0485 | +7.45 % | gain |
|  | sampler pass tokens a step | 2.977, 2.876 | 3.012, 3.084 | 0.100 | +0.121 | +4.15 % | gain |
|  | sampler pass ms a step | 171.71, 161.36 | 151.42, 163.48 | 12.07 | -9.09 ms | -5.46 % | negligible |
|  | prompt tok/s (the prime) | 128.5, 130.1 | 127.9, 130.0 | 2.1 | -0.4 tok/s | -0.29 % | negligible |
| 16,384 | decode tok/s (greedy, both passes) | 17.75, 19.27 | 21.32, 23.99 | 2.67 | +4.14 tok/s | +22.38 % | gain |
|  | decode ms a token | 56.33, 51.88 | 46.90, 41.68 | 5.22 | -9.82 ms | -18.14 % | gain |
|  | the step on pass 1, ms | 214.52, 185.60 | 169.51, 153.21 | 28.92 | -38.70 ms | -19.35 % | gain |
|  | the step on pass 2 (the repeated request), ms | 183.08, 183.34 | 150.66, 151.67 | 1.01 | -32.05 ms | -17.49 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8635, 0.8638 | 0.8265, 0.9029 | 0.0764 | +0.0010 | +0.12 % | negligible |
|  | tokens a step (greedy) | 3.531, 3.556 | 3.413, 3.657 | 0.244 | -0.008 | -0.23 % | negligible |
|  | sampler pass tok/s | 16.49, 18.98 | 24.11, 21.93 | 2.49 | +5.29 tok/s | +29.81 % | gain |
|  | sampler pass share accepted | 0.7033, 0.8472 | 0.9118, 0.8036 | 0.1440 | +0.0824 | +10.63 % | negligible |
|  | sampler pass tokens a step | 3.084, 3.507 | 3.657, 3.368 | 0.423 | +0.217 | +6.59 % | negligible |
|  | sampler pass ms a step | 187.04, 184.74 | 151.66, 153.56 | 2.30 | -33.28 ms | -17.90 % | gain |
|  | prompt tok/s (the prime) | 109.6, 110.0 | 123.9, 128.6 | 4.7 | +16.5 tok/s | +15.01 % | gain |
| 32,768 | decode tok/s (greedy, both passes) | 11.96, 11.32 | 16.66, 15.03 | 1.63 | +4.21 tok/s | +36.13 % | gain |
|  | decode ms a token | 83.60, 88.32 | 60.01, 66.51 | 6.50 | -22.70 ms | -26.40 % | gain |
|  | the step on pass 1, ms | 249.66, 250.08 | 170.26, 172.80 | 2.54 | -78.34 ms | -31.35 % | gain |
|  | the step on pass 2 (the repeated request), ms | 245.10, 244.01 | 165.51, 166.02 | 1.09 | -78.79 ms | -32.22 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.6686, 0.6104 | 0.6173, 0.5245 | 0.0928 | -0.0687 | -10.74 % | negligible |
|  | tokens a step (greedy) | 2.960, 2.798 | 2.798, 2.547 | 0.251 | -0.206 | -7.16 % | negligible |
|  | sampler pass tok/s | 11.27, 10.11 | 15.12, 16.22 | 1.15 | +4.98 tok/s | +46.63 % | gain |
|  | sampler pass share accepted | 0.6074, 0.5168 | 0.5448, 0.6029 | 0.0906 | +0.0118 | +2.10 % | negligible |
|  | sampler pass tokens a step | 2.783, 2.510 | 2.612, 2.783 | 0.273 | +0.051 | +1.94 % | negligible |
|  | sampler pass ms a step | 247.00, 248.20 | 172.75, 171.51 | 1.24 | -75.47 ms | -30.48 % | gain |
|  | prompt tok/s (the prime) | 90.8, 90.5 | 103.6, 103.7 | 0.2 | +13.0 tok/s | +14.36 % | gain |
| 49,152 | decode tok/s (greedy, both passes) | 10.40, 10.70 | 19.08, 18.34 | 0.75 | +8.16 tok/s | +77.34 % | gain |
|  | decode ms a token | 96.11, 93.48 | 52.40, 54.53 | 2.64 | -41.33 ms | -43.60 % | gain |
|  | the step on pass 1, ms | 307.76, 307.18 | 185.29, 185.72 | 0.58 | -121.96 ms | -39.67 % | gain |
|  | the step on pass 2 (the repeated request), ms | 303.51, 302.47 | 182.21, 181.66 | 1.03 | -121.05 ms | -39.95 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7358, 0.7701 | 0.8472, 0.8036 | 0.0437 | +0.0724 | +9.62 % | gain |
|  | tokens a step (greedy) | 3.180, 3.261 | 3.507, 3.368 | 0.138 | +0.217 | +6.74 % | gain |
|  | sampler pass tok/s | 9.65, 9.78 | 17.84, 18.56 | 0.72 | +8.49 tok/s | +87.37 % | gain |
|  | sampler pass share accepted | 0.6627, 0.6786 | 0.7807, 0.8227 | 0.0420 | +0.1311 | +19.54 % | gain |
|  | sampler pass tokens a step | 2.943, 3.012 | 3.282, 3.413 | 0.131 | +0.371 | +12.45 % | gain |
|  | sampler pass ms a step | 304.92, 308.03 | 183.96, 183.90 | 3.10 | -122.55 ms | -39.99 % | gain |
|  | prompt tok/s (the prime) | 69.9, 69.8 | 83.5, 83.1 | 0.4 | +13.4 tok/s | +19.17 % | gain |
| all depths | decode tok/s (greedy, both passes) | 14.78, 15.85 | 20.66, 20.10 | 1.07 | +5.07 tok/s | +33.08 % | gain |
|  | decode ms a token | 67.65, 63.10 | 48.40, 49.75 | 4.56 | -16.30 ms | -24.93 % | gain |
|  | the step on pass 1, ms | 227.63, 214.03 | 165.21, 163.99 | 13.61 | -56.23 ms | -25.46 % | gain |
|  | the step on pass 2 (the repeated request), ms | 203.33, 204.89 | 153.62, 154.36 | 1.56 | -50.12 ms | -24.56 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7442, 0.7891 | 0.7817, 0.7464 | 0.0450 | -0.0026 | -0.34 % | negligible |
|  | tokens a step (greedy) | 3.188, 3.320 | 3.295, 3.200 | 0.132 | -0.007 | -0.21 % | negligible |
|  | sampler pass tok/s | 13.54, 13.31 | 17.88, 17.88 | 0.23 | +4.46 tok/s | +33.22 % | gain |
|  | sampler pass share accepted | 0.6093, 0.6064 | 0.6286, 0.6391 | 0.0106 | +0.0260 | +4.28 % | gain |
|  | sampler pass tokens a step | 2.795, 2.789 | 2.851, 2.883 | 0.032 | +0.075 | +2.69 % | gain |
|  | sampler pass ms a step | 206.48, 209.51 | 159.43, 161.21 | 3.03 | -47.68 ms | -22.92 % | gain |
|  | prompt tok/s (the prime) | 88.7, 88.7 | 101.4, 101.9 | 0.6 | +12.9 tok/s | +14.55 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| rel1 | 1180 | 172 | 997 | +15 | 4.50 |
| rel2 | 1173 | 395 | 802 | +0 | 21.93 |
| v4c1 | 1017 | 106 | 800 | +3 | 2.65 |
| v4c2 | 1018 | 211 | 1,034 | +757 | 5.77 |
