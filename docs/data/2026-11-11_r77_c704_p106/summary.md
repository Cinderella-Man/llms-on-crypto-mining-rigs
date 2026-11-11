## `rows` against `v4c`, decode and prompts

Timed loads, in their order: v4c1, rows1, v4c2, rows2, prof. Every value is one load.

- `v4c`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512`
- `rows`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 LLAMA_QSA_INDEXER_ROWS=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | v4c1 | 26.09, 28.07 → **27.04** | 36.98 | 125.81, 121.58 | 0.7996 | 3.346 | — | 49.0 (64 on 0) |
|  | v4c2 | 26.12, 28.06 → **27.06** | 36.96 | 125.66, 121.62 | 0.7996 | 3.346 | — | 49.5 (64 on 0) |
|  | rows1 | 26.14, 28.06 → **27.07** | 36.94 | 125.54, 121.62 | 0.7996 | 3.346 | — | 49.6 (64 on 0) |
|  | rows2 | 26.18, 28.08 → **27.09** | 36.91 | 125.39, 121.56 | 0.7996 | 3.346 | — | 49.6 (64 on 0) |
| 16,384 | v4c1 | 23.84, 24.15 → **24.00** | 41.67 | 151.23, 149.29 | 0.8810 | 3.606 | — | 135.4 (16,320 on 0) |
|  | v4c2 | 23.87, 24.12 → **24.00** | 41.67 | 151.07, 149.46 | 0.8810 | 3.606 | — | 134.6 (16,320 on 0) |
|  | rows1 | 24.51, 24.81 → **24.66** | 40.55 | 147.11, 145.34 | 0.8810 | 3.606 | — | 134.2 (16,320 on 0) |
|  | rows2 | 24.58, 24.88 → **24.73** | 40.44 | 146.70, 144.94 | 0.8810 | 3.606 | — | 134.5 (16,320 on 0) |
| 32,768 | v4c1 | 14.53, 16.08 → **15.26** | 65.52 | 181.69, 164.16 | 0.5540 | 2.639 | — | 100.2 (16,900 on 15,804) |
|  | v4c2 | 14.55, 16.07 → **15.27** | 65.49 | 181.41, 164.28 | 0.5540 | 2.639 | — | 100.3 (16,900 on 15,804) |
|  | rows1 | 15.14, 16.83 → **15.94** | 62.72 | 174.26, 156.79 | 0.5540 | 2.639 | — | 100.3 (16,900 on 15,804) |
|  | rows2 | 15.16, 16.76 → **15.92** | 62.80 | 174.06, 157.42 | 0.5540 | 2.639 | — | 100.2 (16,900 on 15,804) |
| 49,152 | v4c1 | 18.11, 18.51 → **18.31** | 54.62 | 183.62, 179.58 | 0.7885 | 3.325 | — | 77.4 (16,900 on 32,188) |
|  | v4c2 | 18.10, 18.46 → **18.27** | 54.72 | 183.73, 180.15 | 0.7885 | 3.325 | — | 77.3 (16,900 on 32,188) |
|  | rows1 | 19.19, 19.68 → **19.43** | 51.46 | 173.22, 168.94 | 0.7885 | 3.325 | — | 77.5 (16,900 on 32,188) |
|  | rows2 | 19.21, 19.65 → **19.43** | 51.47 | 173.10, 169.16 | 0.7885 | 3.325 | — | 77.4 (16,900 on 32,188) |

### `rows` against `v4c`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `v4c` loads | `rows` loads | Spread | Against `v4c` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 27.04, 27.06 | 27.07, 27.09 | 0.02 | +0.03 tok/s | +0.12 % | gain |
|  | decode ms a token | 36.98, 36.96 | 36.94, 36.91 | 0.03 | -0.04 ms | -0.12 % | gain |
|  | the step on pass 1, ms | 125.81, 125.66 | 125.54, 125.39 | 0.15 | -0.27 ms | -0.22 % | gain |
|  | the step on pass 2 (the repeated request), ms | 121.58, 121.62 | 121.62, 121.56 | 0.06 | -0.01 ms | -0.01 % | negligible |
|  | share of drafted tokens accepted (greedy) | 0.7996, 0.7996 | 0.7996, 0.7996 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.346, 3.346 | 3.346, 3.346 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 49.0, 49.5 | 49.6, 49.6 | 0.5 | +0.3 tok/s | +0.68 % | negligible |
| 16,384 | decode tok/s (greedy, both passes) | 24.00, 24.00 | 24.66, 24.73 | 0.07 | +0.70 tok/s | +2.91 % | gain |
|  | decode ms a token | 41.67, 41.67 | 40.55, 40.44 | 0.11 | -1.18 ms | -2.82 % | gain |
|  | the step on pass 1, ms | 151.23, 151.07 | 147.11, 146.70 | 0.41 | -4.25 ms | -2.81 % | gain |
|  | the step on pass 2 (the repeated request), ms | 149.29, 149.46 | 145.34, 144.94 | 0.41 | -4.24 ms | -2.84 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.8810, 0.8810 | 0.8810, 0.8810 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.606, 3.606 | 3.606, 3.606 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 135.4, 134.6 | 134.2, 134.5 | 0.8 | -0.7 tok/s | -0.52 % | negligible |
| 32,768 | decode tok/s (greedy, both passes) | 15.26, 15.27 | 15.94, 15.92 | 0.02 | +0.67 tok/s | +4.38 % | gain |
|  | decode ms a token | 65.52, 65.49 | 62.72, 62.80 | 0.08 | -2.75 ms | -4.19 % | gain |
|  | the step on pass 1, ms | 181.69, 181.41 | 174.26, 174.06 | 0.27 | -7.39 ms | -4.07 % | gain |
|  | the step on pass 2 (the repeated request), ms | 164.16, 164.28 | 156.79, 157.42 | 0.63 | -7.11 ms | -4.33 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.5540, 0.5540 | 0.5540, 0.5540 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 2.639, 2.639 | 2.639, 2.639 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 100.2, 100.3 | 100.3, 100.2 | 0.2 | +0.0 tok/s | +0.03 % | negligible |
| 49,152 | decode tok/s (greedy, both passes) | 18.31, 18.27 | 19.43, 19.43 | 0.03 | +1.14 tok/s | +6.23 % | gain |
|  | decode ms a token | 54.62, 54.72 | 51.46, 51.47 | 0.10 | -3.21 ms | -5.87 % | gain |
|  | the step on pass 1, ms | 183.62, 183.73 | 173.22, 173.10 | 0.12 | -10.51 ms | -5.72 % | gain |
|  | the step on pass 2 (the repeated request), ms | 179.58, 180.15 | 168.94, 169.16 | 0.57 | -10.81 ms | -6.01 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7885, 0.7885 | 0.7885, 0.7885 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.325, 3.325 | 3.325, 3.325 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 77.4, 77.3 | 77.5, 77.4 | 0.1 | +0.1 tok/s | +0.09 % | negligible |
| all depths | decode tok/s (greedy, both passes) | 20.12, 20.12 | 20.87, 20.87 | 0.01 | +0.75 tok/s | +3.74 % | gain |
|  | decode ms a token | 49.70, 49.71 | 47.92, 47.91 | 0.01 | -1.79 ms | -3.61 % | gain |
|  | the step on pass 1, ms | 161.96, 161.83 | 156.28, 156.06 | 0.22 | -5.73 ms | -3.54 % | gain |
|  | the step on pass 2 (the repeated request), ms | 154.59, 154.81 | 148.93, 149.07 | 0.22 | -5.70 ms | -3.68 % | gain |
|  | share of drafted tokens accepted (greedy) | 0.7406, 0.7406 | 0.7406, 0.7406 | 0.0000 | +0.0000 | +0.00 % | negligible |
|  | tokens a step (greedy) | 3.185, 3.185 | 3.185, 3.185 | 0.000 | +0.000 | +0.00 % | negligible |
|  | prompt tok/s (the prime) | 98.7, 98.6 | 98.6, 98.6 | 0.1 | -0.1 tok/s | -0.07 % | negligible |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| v4c1 | 895 | 443 | 9 | +0 | 0.00 |
| v4c2 | 895 | 439 | 9 | +0 | 0.00 |
| rows1 | 892 | 442 | 9 | +0 | 0.00 |
| rows2 | 892 | 441 | 9 | +0 | 0.00 |
