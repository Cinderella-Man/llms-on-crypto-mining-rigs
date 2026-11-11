## `gains` against `v3`, decode and prompts

Timed loads, in their order: v31, gains1, v32, gains2, off1, off2. Every value is one load.

- `v3`: the preset as it is
- `gains`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on --spec-draft-n-ubatch 512`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | v31 | 26.27, 27.60 → **26.92** | 37.15 | 124.96, 123.66 | 0.7996 | 3.346 | 17.03, 0.4506, 2.327, 136.65 | 32.8 (64 on 0) |
|  | v32 | 25.71, 27.59 → **26.62** | 37.57 | 127.65, 123.70 | 0.7996 | 3.346 | 16.12, 0.4506, 2.327, 144.40 | 31.7 (64 on 0) |
|  | gains1 | 26.54, 27.78 → **27.15** | 36.84 | 112.16, 108.42 | 0.6779 | 2.994 | 19.08, 0.4189, 2.246, 117.71 | 53.6 (64 on 0) |
|  | gains2 | 26.54, 27.68 → **27.10** | 36.90 | 112.15, 108.80 | 0.6779 | 2.994 | 19.09, 0.4189, 2.246, 117.64 | 52.1 (64 on 0) |
| 8,192 | v31 | 16.27, 24.11 → **19.43** | 51.47 | 234.86, 158.48 | 0.9643 | 3.821 | 19.33, 0.7160, 3.122, 161.49 | 124.6 (8,128 on 0) |
|  | v32 | 16.32, 23.62 → **19.30** | 51.80 | 217.84, 152.65 | 0.8849 | 3.580 | 17.11, 0.6007, 2.783, 162.61 | 125.7 (8,128 on 0) |
|  | gains1 | 18.47, 23.96 → **20.86** | 47.94 | 155.69, 127.22 | 0.6660 | 2.960 | 24.66, 0.7426, 3.200, 129.78 | 142.0 (8,128 on 0) |
|  | gains2 | 18.48, 24.03 → **20.90** | 47.86 | 155.62, 126.80 | 0.6660 | 2.960 | 24.64, 0.7426, 3.200, 129.86 | 142.0 (8,128 on 0) |
| 16,384 | v31 | 17.28, 19.59 → **18.36** | 54.46 | 208.69, 184.07 | 0.8810 | 3.606 | 17.57, 0.7706, 3.282, 186.82 | 108.5 (8,708 on 7,612) |
|  | v32 | 14.37, 19.68 → **16.61** | 60.19 | 254.45, 185.79 | 0.8986 | 3.657 | 16.10, 0.7597, 3.241, 201.23 | 108.4 (8,708 on 7,612) |
|  | gains1 | 24.74, 26.24 → **25.47** | 39.26 | 134.40, 130.06 | 0.8072 | 3.368 | 27.15, 0.8852, 3.606, 132.79 | 125.8 (8,708 on 7,612) |
|  | gains2 | 24.62, 26.19 → **25.38** | 39.40 | 135.02, 130.33 | 0.8072 | 3.368 | 27.18, 0.8852, 3.606, 132.65 | 125.0 (8,708 on 7,612) |
| 32,768 | v31 | 11.13, 12.06 → **11.58** | 86.38 | 264.26, 244.09 | 0.6576 | 2.943 | 10.42, 0.5505, 2.612, 250.65 | 90.0 (16,388 on 16,316) |
|  | v32 | 10.42, 11.28 → **10.83** | 92.32 | 261.33, 244.11 | 0.5898 | 2.738 | 8.73, 0.4017, 2.188, 250.63 | 90.1 (16,388 on 16,316) |
|  | gains1 | 19.15, 20.31 → **19.71** | 50.73 | 145.29, 140.05 | 0.6111 | 2.813 | 20.39, 0.6786, 3.012, 147.69 | 98.5 (16,388 on 16,316) |
|  | gains2 | 19.14, 20.23 → **19.67** | 50.84 | 145.39, 140.58 | 0.6111 | 2.813 | 20.46, 0.6786, 3.012, 147.23 | 98.7 (16,388 on 16,316) |
| 49,152 | v31 | 11.60, 11.76 → **11.68** | 85.60 | 306.48, 302.25 | 0.8762 | 3.556 | 8.71, 0.5771, 2.695, 309.56 | 69.3 (16,900 on 32,188) |
|  | v32 | 10.02, 10.72 → **10.36** | 96.54 | 319.27, 302.37 | 0.7591 | 3.220 | 10.57, 0.7629, 3.241, 306.58 | 69.5 (16,900 on 32,188) |
|  | gains1 | 19.80, 22.48 → **21.06** | 47.49 | 170.14, 149.81 | 0.8108 | 3.368 | 19.75, 0.7090, 3.084, 156.19 | 75.5 (16,900 on 32,188) |
|  | gains2 | 18.12, 22.44 → **20.05** | 49.88 | 185.90, 150.10 | 0.8108 | 3.368 | 19.97, 0.7090, 3.084, 154.43 | 76.5 (16,900 on 32,188) |

### `gains` against `v3`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `v3` loads | `gains` loads | Spread | Against `v3` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 26.92, 26.62 | 27.15, 27.10 | 0.30 | +0.35 tok/s | +1.33 % | gain |
|  | decode ms a token | 37.15, 37.57 | 36.84, 36.90 | 0.42 | -0.49 ms | -1.31 % | gain |
|  | the step on pass 1, ms | 124.96, 127.65 | 112.16, 112.15 | 2.69 | -14.15 ms | -11.20 % | gain |
|  | the step on pass 2 (the repeated request), ms | 123.66, 123.70 | 108.42, 108.80 | 0.38 | -15.07 ms | -12.19 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.6779, 0.6779 | 0.0000 | -0.1216 | -15.21 % | loss |
|  | tokens a step (greedy) | 3.346, 3.346 | 2.994, 2.994 | 0.000 | -0.352 | -10.53 % | loss |
|  | sampler pass tok/s | 17.03, 16.12 | 19.08, 19.09 | 0.91 | +2.51 tok/s | +15.14 % | gain |
|  | sampler pass share accepted | 0.4506, 0.4506 | 0.4189, 0.4189 | 0.0000 | -0.0317 | -7.04 % | loss |
|  | sampler pass tokens a step | 2.327, 2.327 | 2.246, 2.246 | 0.000 | -0.082 | -3.51 % | loss |
|  | sampler pass ms a step | 136.65, 144.40 | 117.71, 117.64 | 7.75 | -22.85 ms | -16.26 % | gain |
|  | prompt tok/s (the prime) | 32.8, 31.7 | 53.6, 52.1 | 1.6 | +20.6 tok/s | +64.03 % | gain |
| 8,192 | decode tok/s (greedy, both passes) | 19.43, 19.30 | 20.86, 20.90 | 0.12 | +1.51 tok/s | +7.81 % | gain |
|  | decode ms a token | 51.47, 51.80 | 47.94, 47.86 | 0.33 | -3.74 ms | -7.25 % | gain |
|  | the step on pass 1, ms | 234.86, 217.84 | 155.69, 155.62 | 17.02 | -70.69 ms | -31.23 % | gain |
|  | the step on pass 2 (the repeated request), ms | 158.48, 152.65 | 127.22, 126.80 | 5.83 | -28.55 ms | -18.35 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.9643, 0.8849 | 0.6660, 0.6660 | 0.0794 | -0.2586 | -27.97 % | loss |
|  | tokens a step (greedy) | 3.821, 3.580 | 2.960, 2.960 | 0.240 | -0.741 | -20.03 % | loss |
|  | sampler pass tok/s | 19.33, 17.11 | 24.66, 24.64 | 2.22 | +6.43 tok/s | +35.27 % | gain |
|  | sampler pass share accepted | 0.7160, 0.6007 | 0.7426, 0.7426 | 0.1153 | +0.0842 | +12.79 % | negligible |
|  | sampler pass tokens a step | 3.122, 2.783 | 3.200, 3.200 | 0.339 | +0.248 | +8.39 % | negligible |
|  | sampler pass ms a step | 161.49, 162.61 | 129.78, 129.86 | 1.12 | -32.23 ms | -19.89 % | gain |
|  | prompt tok/s (the prime) | 124.6, 125.7 | 142.0, 142.0 | 1.1 | +16.8 tok/s | +13.45 % | gain |
| 16,384 | decode tok/s (greedy, both passes) | 18.36, 16.61 | 25.47, 25.38 | 1.75 | +7.94 tok/s | +45.39 % | gain |
|  | decode ms a token | 54.46, 60.19 | 39.26, 39.40 | 5.72 | -18.00 ms | -31.39 % | gain |
|  | the step on pass 1, ms | 208.69, 254.45 | 134.40, 135.02 | 45.76 | -96.85 ms | -41.83 % | gain |
|  | the step on pass 2 (the repeated request), ms | 184.07, 185.79 | 130.06, 130.33 | 1.72 | -54.74 ms | -29.60 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8810, 0.8986 | 0.8072, 0.8072 | 0.0176 | -0.0826 | -9.28 % | loss |
|  | tokens a step (greedy) | 3.606, 3.657 | 3.368, 3.368 | 0.052 | -0.263 | -7.24 % | loss |
|  | sampler pass tok/s | 17.57, 16.10 | 27.15, 27.18 | 1.47 | +10.33 tok/s | +61.36 % | gain |
|  | sampler pass share accepted | 0.7706, 0.7597 | 0.8852, 0.8852 | 0.0109 | +0.1201 | +15.69 % | gain |
|  | sampler pass tokens a step | 3.282, 3.241 | 3.606, 3.606 | 0.042 | +0.344 | +10.56 % | gain |
|  | sampler pass ms a step | 186.82, 201.23 | 132.79, 132.65 | 14.42 | -61.30 ms | -31.60 % | gain |
|  | prompt tok/s (the prime) | 108.5, 108.4 | 125.8, 125.0 | 0.8 | +16.9 tok/s | +15.60 % | gain |
| 32,768 | decode tok/s (greedy, both passes) | 11.58, 10.83 | 19.71, 19.67 | 0.74 | +8.49 tok/s | +75.76 % | gain |
|  | decode ms a token | 86.38, 92.32 | 50.73, 50.84 | 5.94 | -38.57 ms | -43.17 % | gain |
|  | the step on pass 1, ms | 264.26, 261.33 | 145.29, 145.39 | 2.94 | -117.45 ms | -44.69 % | gain |
|  | the step on pass 2 (the repeated request), ms | 244.09, 244.11 | 140.05, 140.58 | 0.53 | -103.78 ms | -42.52 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.6576, 0.5898 | 0.6111, 0.6111 | 0.0678 | -0.0126 | -2.02 % | negligible |
|  | tokens a step (greedy) | 2.943, 2.738 | 2.813, 2.813 | 0.205 | -0.027 | -0.95 % | negligible |
|  | sampler pass tok/s | 10.42, 8.73 | 20.39, 20.46 | 1.69 | +10.85 tok/s | +113.29 % | gain |
|  | sampler pass share accepted | 0.5505, 0.4017 | 0.6786, 0.6786 | 0.1488 | +0.2024 | +42.52 % | gain |
|  | sampler pass tokens a step | 2.612, 2.188 | 3.012, 3.012 | 0.424 | +0.612 | +25.48 % | gain |
|  | sampler pass ms a step | 250.65, 250.63 | 147.69, 147.23 | 0.45 | -103.18 ms | -41.17 % | gain |
|  | prompt tok/s (the prime) | 90.0, 90.1 | 98.5, 98.7 | 0.2 | +8.5 tok/s | +9.45 % | gain |
| 49,152 | decode tok/s (greedy, both passes) | 11.68, 10.36 | 21.06, 20.05 | 1.32 | +9.53 tok/s | +86.51 % | gain |
|  | decode ms a token | 85.60, 96.54 | 47.49, 49.88 | 10.94 | -42.39 ms | -46.54 % | gain |
|  | the step on pass 1, ms | 306.48, 319.27 | 170.14, 185.90 | 15.76 | -134.86 ms | -43.10 % | gain |
|  | the step on pass 2 (the repeated request), ms | 302.25, 302.37 | 149.81, 150.10 | 0.29 | -152.36 ms | -50.40 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8762, 0.7591 | 0.8108, 0.8108 | 0.1171 | -0.0069 | -0.84 % | negligible |
|  | tokens a step (greedy) | 3.556, 3.220 | 3.368, 3.368 | 0.335 | -0.019 | -0.57 % | negligible |
|  | sampler pass tok/s | 8.71, 10.57 | 19.75, 19.97 | 1.86 | +10.22 tok/s | +106.07 % | gain |
|  | sampler pass share accepted | 0.5771, 0.7629 | 0.7090, 0.7090 | 0.1859 | +0.0390 | +5.82 % | negligible |
|  | sampler pass tokens a step | 2.695, 3.241 | 3.084, 3.084 | 0.546 | +0.117 | +3.93 % | negligible |
|  | sampler pass ms a step | 309.56, 306.58 | 156.19, 154.43 | 2.98 | -152.76 ms | -49.59 % | gain |
|  | prompt tok/s (the prime) | 69.3, 69.5 | 75.5, 76.5 | 1.0 | +6.6 tok/s | +9.50 % | gain |
| all depths | decode tok/s (greedy, both passes) | 15.87, 14.77 | 22.50, 22.24 | 1.09 | +7.04 tok/s | +45.97 % | gain |
|  | decode ms a token | 63.01, 67.68 | 44.45, 44.97 | 4.67 | -20.64 ms | -31.58 % | gain |
|  | the step on pass 1, ms | 227.62, 237.46 | 143.21, 146.18 | 9.84 | -87.84 ms | -37.78 % | gain |
|  | the step on pass 2 (the repeated request), ms | 204.19, 205.44 | 130.85, 131.06 | 1.25 | -73.87 ms | -36.06 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8260, 0.7744 | 0.7084, 0.7084 | 0.0516 | -0.0917 | -11.47 % | loss |
|  | tokens a step (greedy) | 3.427, 3.274 | 3.084, 3.084 | 0.153 | -0.266 | -7.94 % | loss |
|  | sampler pass tok/s | 13.22, 12.76 | 21.79, 21.86 | 0.46 | +8.83 tok/s | +67.99 % | gain |
|  | sampler pass share accepted | 0.5990, 0.5703 | 0.6612, 0.6612 | 0.0287 | +0.0766 | +13.10 % | gain |
|  | sampler pass tokens a step | 2.765, 2.683 | 2.956, 2.956 | 0.081 | +0.232 | +8.52 % | gain |
|  | sampler pass ms a step | 209.11, 210.24 | 135.67, 135.22 | 1.13 | -74.23 ms | -35.40 % | gain |
|  | prompt tok/s (the prime) | 87.7, 88.0 | 97.0, 97.6 | 0.6 | +9.4 tok/s | +10.74 % | gain |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| v31 | 1178 | 207 | 975 | +20 | 11.80 |
| v32 | 1193 | 80 | 1,151 | +796 | 17.22 |
| gains1 | 1016 | 304 | 40 | +0 | 0.20 |
| gains2 | 1019 | 313 | 474 | +0 | 1.42 |
