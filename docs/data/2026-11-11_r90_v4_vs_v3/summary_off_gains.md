## `gains` against `off`, decode and prompts

Timed loads, in their order: v31, gains1, v32, gains2, off1, off2. Every value is one load.

- `off`: the preset as it is
- `gains`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on --spec-draft-n-ubatch 512`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | off1 | 26.62, 28.06 → **27.32** | 36.60 | 123.27, 121.64 | 0.7996 | 3.346 | 16.65, 0.4506, 2.327, 139.75 | 33.1 (64 on 0) |
|  | off2 | 26.07, 27.97 → **26.99** | 37.05 | 125.89, 122.03 | 0.7996 | 3.346 | 16.65, 0.4506, 2.327, 139.74 | 32.8 (64 on 0) |
|  | gains1 | 26.54, 27.78 → **27.15** | 36.84 | 112.16, 108.42 | 0.6779 | 2.994 | 19.08, 0.4189, 2.246, 117.71 | 53.6 (64 on 0) |
|  | gains2 | 26.54, 27.68 → **27.10** | 36.90 | 112.15, 108.80 | 0.6779 | 2.994 | 19.09, 0.4189, 2.246, 117.64 | 52.1 (64 on 0) |
| 8,192 | off1 | 14.28, 24.37 → **18.01** | 55.52 | 256.03, 152.27 | 0.9075 | 3.683 | 16.10, 0.5288, 2.560, 159.00 | 133.0 (8,128 on 0) |
|  | off2 | 20.92, 24.29 → **22.48** | 44.49 | 174.85, 152.73 | 0.9075 | 3.683 | 15.83, 0.5288, 2.560, 161.71 | 137.7 (8,128 on 0) |
|  | gains1 | 18.47, 23.96 → **20.86** | 47.94 | 155.69, 127.22 | 0.6660 | 2.960 | 24.66, 0.7426, 3.200, 129.78 | 142.0 (8,128 on 0) |
|  | gains2 | 18.48, 24.03 → **20.90** | 47.86 | 155.62, 126.80 | 0.6660 | 2.960 | 24.64, 0.7426, 3.200, 129.86 | 142.0 (8,128 on 0) |
| 16,384 | off1 | 15.29, 18.34 → **16.68** | 59.96 | 214.60, 181.30 | 0.7778 | 3.303 | 17.86, 0.7807, 3.282, 183.76 | 113.3 (8,708 on 7,612) |
|  | off2 | 14.15, 18.35 → **15.98** | 62.58 | 231.94, 181.18 | 0.7778 | 3.303 | 17.53, 0.7807, 3.282, 187.25 | 113.9 (8,708 on 7,612) |
|  | gains1 | 24.74, 26.24 → **25.47** | 39.26 | 134.40, 130.06 | 0.8072 | 3.368 | 27.15, 0.8852, 3.606, 132.79 | 125.8 (8,708 on 7,612) |
|  | gains2 | 24.62, 26.19 → **25.38** | 39.40 | 135.02, 130.33 | 0.8072 | 3.368 | 27.18, 0.8852, 3.606, 132.65 | 125.0 (8,708 on 7,612) |
| 32,768 | off1 | 10.79, 12.22 → **11.46** | 87.25 | 269.51, 240.89 | 0.6595 | 2.926 | 10.68, 0.5521, 2.639, 247.08 | 93.8 (16,388 on 16,316) |
|  | off2 | 10.96, 12.21 → **11.55** | 86.59 | 265.46, 241.08 | 0.6595 | 2.926 | 10.67, 0.5521, 2.639, 247.28 | 93.3 (16,388 on 16,316) |
|  | gains1 | 19.15, 20.31 → **19.71** | 50.73 | 145.29, 140.05 | 0.6111 | 2.813 | 20.39, 0.6786, 3.012, 147.69 | 98.5 (16,388 on 16,316) |
|  | gains2 | 19.14, 20.23 → **19.67** | 50.84 | 145.39, 140.58 | 0.6111 | 2.813 | 20.46, 0.6786, 3.012, 147.23 | 98.7 (16,388 on 16,316) |
| 49,152 | off1 | 11.24, 11.40 → **11.32** | 88.32 | 303.56, 299.38 | 0.8265 | 3.413 | 10.91, 0.7885, 3.325, 304.66 | 71.8 (16,900 on 32,188) |
|  | off2 | 11.22, 11.40 → **11.31** | 88.45 | 304.26, 299.53 | 0.8265 | 3.413 | 10.92, 0.7885, 3.325, 304.60 | 71.5 (16,900 on 32,188) |
|  | gains1 | 19.80, 22.48 → **21.06** | 47.49 | 170.14, 149.81 | 0.8108 | 3.368 | 19.75, 0.7090, 3.084, 156.19 | 75.5 (16,900 on 32,188) |
|  | gains2 | 18.12, 22.44 → **20.05** | 49.88 | 185.90, 150.10 | 0.8108 | 3.368 | 19.97, 0.7090, 3.084, 154.43 | 76.5 (16,900 on 32,188) |

### `gains` against `off`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `off` loads | `gains` loads | Spread | Against `off` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 27.32, 26.99 | 27.15, 27.10 | 0.34 | -0.03 tok/s | -0.12 % | negligible |
|  | decode ms a token | 36.60, 37.05 | 36.84, 36.90 | 0.45 | +0.04 ms | +0.12 % | negligible |
|  | the step on pass 1, ms | 123.27, 125.89 | 112.16, 112.15 | 2.62 | -12.43 ms | -9.98 % | gain |
|  | the step on pass 2 (the repeated request), ms | 121.64, 122.03 | 108.42, 108.80 | 0.38 | -13.23 ms | -10.86 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.6779, 0.6779 | 0.0000 | -0.1216 | -15.21 % | loss |
|  | tokens a step (greedy) | 3.346, 3.346 | 2.994, 2.994 | 0.000 | -0.352 | -10.53 % | loss |
|  | sampler pass tok/s | 16.65, 16.65 | 19.08, 19.09 | 0.01 | +2.43 tok/s | +14.59 % | gain |
|  | sampler pass share accepted | 0.4506, 0.4506 | 0.4189, 0.4189 | 0.0000 | -0.0317 | -7.04 % | loss |
|  | sampler pass tokens a step | 2.327, 2.327 | 2.246, 2.246 | 0.000 | -0.082 | -3.51 % | loss |
|  | sampler pass ms a step | 139.75, 139.74 | 117.71, 117.64 | 0.08 | -22.07 ms | -15.79 % | gain |
|  | prompt tok/s (the prime) | 33.1, 32.8 | 53.6, 52.1 | 1.6 | +19.9 tok/s | +60.54 % | gain |
| 8,192 | decode tok/s (greedy, both passes) | 18.01, 22.48 | 20.86, 20.90 | 4.47 | +0.63 tok/s | +3.14 % | negligible |
|  | decode ms a token | 55.52, 44.49 | 47.94, 47.86 | 11.04 | -2.11 ms | -4.22 % | negligible |
|  | the step on pass 1, ms | 256.03, 174.85 | 155.69, 155.62 | 81.19 | -59.78 ms | -27.75 % | negligible |
|  | the step on pass 2 (the repeated request), ms | 152.27, 152.73 | 127.22, 126.80 | 0.46 | -25.49 ms | -16.71 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.9075, 0.9075 | 0.6660, 0.6660 | 0.0000 | -0.2415 | -26.61 % | loss |
|  | tokens a step (greedy) | 3.683, 3.683 | 2.960, 2.960 | 0.000 | -0.724 | -19.65 % | loss |
|  | sampler pass tok/s | 16.10, 15.83 | 24.66, 24.64 | 0.27 | +8.68 tok/s | +54.39 % | gain |
|  | sampler pass share accepted | 0.5288, 0.5288 | 0.7426, 0.7426 | 0.0000 | +0.2138 | +40.43 % | gain |
|  | sampler pass tokens a step | 2.560, 2.560 | 3.200, 3.200 | 0.000 | +0.640 | +25.00 % | gain |
|  | sampler pass ms a step | 159.00, 161.71 | 129.78, 129.86 | 2.71 | -30.53 ms | -19.04 % | gain |
|  | prompt tok/s (the prime) | 133.0, 137.7 | 142.0, 142.0 | 4.7 | +6.6 tok/s | +4.89 % | gain |
| 16,384 | decode tok/s (greedy, both passes) | 16.68, 15.98 | 25.47, 25.38 | 0.70 | +9.10 tok/s | +55.71 % | gain |
|  | decode ms a token | 59.96, 62.58 | 39.26, 39.40 | 2.63 | -21.94 ms | -35.81 % | gain |
|  | the step on pass 1, ms | 214.60, 231.94 | 134.40, 135.02 | 17.34 | -88.56 ms | -39.66 % | gain |
|  | the step on pass 2 (the repeated request), ms | 181.30, 181.18 | 130.06, 130.33 | 0.27 | -51.04 ms | -28.16 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7778, 0.7778 | 0.8072, 0.8072 | 0.0000 | +0.0294 | +3.78 % | gain |
|  | tokens a step (greedy) | 3.303, 3.303 | 3.368, 3.368 | 0.000 | +0.065 | +1.97 % | gain |
|  | sampler pass tok/s | 17.86, 17.53 | 27.15, 27.18 | 0.33 | +9.47 tok/s | +53.54 % | gain |
|  | sampler pass share accepted | 0.7807, 0.7807 | 0.8852, 0.8852 | 0.0000 | +0.1045 | +13.38 % | gain |
|  | sampler pass tokens a step | 3.282, 3.282 | 3.606, 3.606 | 0.000 | +0.324 | +9.86 % | gain |
|  | sampler pass ms a step | 183.76, 187.25 | 132.79, 132.65 | 3.49 | -52.79 ms | -28.45 % | gain |
|  | prompt tok/s (the prime) | 113.3, 113.9 | 125.8, 125.0 | 0.8 | +11.8 tok/s | +10.42 % | gain |
| 32,768 | decode tok/s (greedy, both passes) | 11.46, 11.55 | 19.71, 19.67 | 0.09 | +8.19 tok/s | +71.17 % | gain |
|  | decode ms a token | 87.25, 86.59 | 50.73, 50.84 | 0.66 | -36.14 ms | -41.58 % | gain |
|  | the step on pass 1, ms | 269.51, 265.46 | 145.29, 145.39 | 4.05 | -122.14 ms | -45.66 % | gain |
|  | the step on pass 2 (the repeated request), ms | 240.89, 241.08 | 140.05, 140.58 | 0.53 | -100.67 ms | -41.77 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.6595, 0.6595 | 0.6111, 0.6111 | 0.0000 | -0.0484 | -7.34 % | loss |
|  | tokens a step (greedy) | 2.926, 2.926 | 2.813, 2.813 | 0.000 | -0.113 | -3.85 % | loss |
|  | sampler pass tok/s | 10.68, 10.67 | 20.39, 20.46 | 0.06 | +9.75 tok/s | +91.29 % | gain |
|  | sampler pass share accepted | 0.5521, 0.5521 | 0.6786, 0.6786 | 0.0000 | +0.1265 | +22.91 % | gain |
|  | sampler pass tokens a step | 2.639, 2.639 | 3.012, 3.012 | 0.000 | +0.373 | +14.12 % | gain |
|  | sampler pass ms a step | 247.08, 247.28 | 147.69, 147.23 | 0.45 | -99.72 ms | -40.34 % | gain |
|  | prompt tok/s (the prime) | 93.8, 93.3 | 98.5, 98.7 | 0.5 | +5.0 tok/s | +5.39 % | gain |
| 49,152 | decode tok/s (greedy, both passes) | 11.32, 11.31 | 21.06, 20.05 | 1.01 | +9.24 tok/s | +81.66 % | gain |
|  | decode ms a token | 88.32, 88.45 | 47.49, 49.88 | 2.38 | -39.70 ms | -44.92 % | gain |
|  | the step on pass 1, ms | 303.56, 304.26 | 170.14, 185.90 | 15.76 | -125.89 ms | -41.42 % | gain |
|  | the step on pass 2 (the repeated request), ms | 299.38, 299.53 | 149.81, 150.10 | 0.29 | -149.50 ms | -49.92 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8265, 0.8265 | 0.8108, 0.8108 | 0.0000 | -0.0157 | -1.90 % | loss |
|  | tokens a step (greedy) | 3.413, 3.413 | 3.368, 3.368 | 0.000 | -0.045 | -1.32 % | loss |
|  | sampler pass tok/s | 10.91, 10.92 | 19.75, 19.97 | 0.23 | +8.95 tok/s | +81.97 % | gain |
|  | sampler pass share accepted | 0.7885, 0.7885 | 0.7090, 0.7090 | 0.0000 | -0.0795 | -10.09 % | loss |
|  | sampler pass tokens a step | 3.325, 3.325 | 3.084, 3.084 | 0.000 | -0.240 | -7.23 % | loss |
|  | sampler pass ms a step | 304.66, 304.60 | 156.19, 154.43 | 1.77 | -149.32 ms | -49.02 % | gain |
|  | prompt tok/s (the prime) | 71.8, 71.5 | 75.5, 76.5 | 1.0 | +4.4 tok/s | +6.07 % | gain |
| all depths | decode tok/s (greedy, both passes) | 15.26, 15.67 | 22.50, 22.24 | 0.41 | +6.90 tok/s | +44.64 % | gain |
|  | decode ms a token | 65.53, 63.83 | 44.45, 44.97 | 1.70 | -19.97 ms | -30.87 % | gain |
|  | the step on pass 1, ms | 233.32, 221.93 | 143.21, 146.18 | 11.39 | -82.93 ms | -36.43 % | gain |
|  | the step on pass 2 (the repeated request), ms | 201.05, 201.25 | 130.85, 131.06 | 0.21 | -70.20 ms | -34.90 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7884, 0.7884 | 0.7084, 0.7084 | 0.0000 | -0.0799 | -10.14 % | loss |
|  | tokens a step (greedy) | 3.316, 3.316 | 3.084, 3.084 | 0.000 | -0.232 | -6.99 % | loss |
|  | sampler pass tok/s | 13.76, 13.68 | 21.79, 21.86 | 0.08 | +8.11 tok/s | +59.10 % | gain |
|  | sampler pass share accepted | 0.6006, 0.6006 | 0.6612, 0.6612 | 0.0000 | +0.0606 | +10.09 % | gain |
|  | sampler pass tokens a step | 2.771, 2.771 | 2.956, 2.956 | 0.000 | +0.186 | +6.70 % | gain |
|  | sampler pass ms a step | 201.36, 202.57 | 135.67, 135.22 | 1.21 | -66.52 ms | -32.94 % | gain |
|  | prompt tok/s (the prime) | 91.5, 91.5 | 97.0, 97.6 | 0.6 | +5.8 tok/s | +6.31 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| off1 | 1157 | 310 | 1,085 | +785 | 22.45 |
| off2 | 1148 | 81 | 207 | +168 | 2.91 |
| gains1 | 1016 | 304 | 40 | +0 | 0.20 |
| gains2 | 1019 | 313 | 474 | +0 | 1.42 |
