## `gmix` against `pmix`, decode and prompts

Timed loads, in their order: pmix1, gmix1, pmix2, gmix2. Every value is one load.

- `pmix`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 --spec-draft-n-ubatch 512`
- `gmix`: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 --spec-draft-n-ubatch 512 -m Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf`

### Per depth and load

| Depth (tokens) | Load | Decode tok/s: pass 1, pass 2 → load | ms a token | Step pass 1, pass 2, ms | Share accepted | Tokens a step | Sampler pass: tok/s, share, tokens a step, ms a step | Prompt tok/s (read on cached) |
|---:|---|---|---:|---|---:|---:|---|---|
| 0 | pmix1 | 26.90, 28.00 → **27.44** | 36.45 | 102.34, 99.37 | 0.6000 | 2.768 | 20.69, 0.5270, 2.560, 123.71 | 54.4 (64 on 0) |
|  | pmix2 | 25.96, 27.94 → **26.91** | 37.15 | 106.02, 99.60 | 0.6000 | 2.768 | 20.04, 0.5270, 2.560, 127.75 | 53.0 (64 on 0) |
|  | gmix1 | 17.22, 19.25 → **18.18** | 55.00 | 125.98, 118.71 | 0.4147 | 2.226 | 20.22, 0.6398, 2.876, 142.25 | 51.3 (64 on 0) |
|  | gmix2 | 17.24, 19.27 → **18.19** | 54.96 | 125.87, 118.63 | 0.4147 | 2.226 | 20.61, 0.6398, 2.876, 139.56 | 51.2 (64 on 0) |
| 8,192 | pmix1 | 20.24, 26.62 → **23.00** | 43.48 | 150.56, 117.26 | 0.7033 | 3.084 | 23.29, 0.6350, 2.876, 123.50 | 152.2 (8,128 on 0) |
|  | pmix2 | 20.73, 26.70 → **23.34** | 42.84 | 146.98, 116.94 | 0.7033 | 3.084 | 23.18, 0.6350, 2.876, 124.07 | 150.6 (8,128 on 0) |
|  | gmix1 | 16.29, 20.90 → **18.31** | 54.62 | 170.84, 134.59 | 0.6127 | 2.798 | 19.68, 0.6157, 2.813, 142.93 | 122.5 (8,128 on 0) |
|  | gmix2 | 16.29, 20.88 → **18.30** | 54.64 | 170.81, 134.75 | 0.6127 | 2.798 | 19.68, 0.6157, 2.813, 142.92 | 127.5 (8,128 on 0) |

### `gmix` against `pmix`: classes

Per metric: each load by value (the control's, then the arm's), the spread (the larger of the two arms' differences between their two loads), the arm against the control and the class. For a time less is better.

| Depth (tokens) | Metric | `pmix` loads | `gmix` loads | Spread | Against `pmix` | % | Class |
|---:|---|---|---|---:|---:|---:|---|
| 0 | decode tok/s (greedy, both passes) | 27.44, 26.91 | 18.18, 18.19 | 0.52 | -8.99 tok/s | -33.07 % | loss |
|  | decode ms a token | 36.45, 37.15 | 55.00, 54.96 | 0.71 | +18.18 ms | +49.41 % | loss |
|  | the step on pass 1, ms | 102.34, 106.02 | 125.98, 125.87 | 3.67 | +21.74 ms | +20.87 % | loss |
|  | the step on pass 2 (the repeated request), ms | 99.37, 99.60 | 118.71, 118.63 | 0.23 | +19.19 ms | +19.28 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.6000, 0.6000 | 0.4147, 0.4147 | 0.0000 | -0.1853 | -30.88 % | loss |
|  | tokens a step (greedy) | 2.768, 2.768 | 2.226, 2.226 | 0.000 | -0.541 | -19.57 % | loss |
|  | sampler pass tok/s | 20.69, 20.04 | 20.22, 20.61 | 0.65 | +0.05 tok/s | +0.24 % | negligible |
|  | sampler pass share accepted | 0.5270, 0.5270 | 0.6398, 0.6398 | 0.0000 | +0.1128 | +21.41 % | gain |
|  | sampler pass tokens a step | 2.560, 2.560 | 2.876, 2.876 | 0.000 | +0.316 | +12.36 % | gain |
|  | sampler pass ms a step | 123.71, 127.75 | 142.25, 139.56 | 4.04 | +15.18 ms | +12.07 % | loss |
|  | prompt tok/s (the prime) | 54.4, 53.0 | 51.3, 51.2 | 1.4 | -2.5 tok/s | -4.58 % | loss |
| 8,192 | decode tok/s (greedy, both passes) | 23.00, 23.34 | 18.31, 18.30 | 0.34 | -4.87 tok/s | -21.00 % | loss |
|  | decode ms a token | 43.48, 42.84 | 54.62, 54.64 | 0.64 | +11.47 ms | +26.57 % | loss |
|  | the step on pass 1, ms | 150.56, 146.98 | 170.84, 170.81 | 3.58 | +22.05 ms | +14.82 % | loss |
|  | the step on pass 2 (the repeated request), ms | 117.26, 116.94 | 134.59, 134.75 | 0.31 | +17.57 ms | +15.01 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.7033, 0.7033 | 0.6127, 0.6127 | 0.0000 | -0.0906 | -12.88 % | loss |
|  | tokens a step (greedy) | 3.084, 3.084 | 2.798, 2.798 | 0.000 | -0.287 | -9.29 % | loss |
|  | sampler pass tok/s | 23.29, 23.18 | 19.68, 19.68 | 0.11 | -3.56 tok/s | -15.30 % | loss |
|  | sampler pass share accepted | 0.6350, 0.6350 | 0.6157, 0.6157 | 0.0000 | -0.0193 | -3.04 % | loss |
|  | sampler pass tokens a step | 2.876, 2.876 | 2.813, 2.813 | 0.000 | -0.063 | -2.20 % | loss |
|  | sampler pass ms a step | 123.50, 124.07 | 142.93, 142.92 | 0.57 | +19.14 ms | +15.47 % | loss |
|  | prompt tok/s (the prime) | 152.2, 150.6 | 122.5, 127.5 | 4.9 | -26.4 tok/s | -17.43 % | loss |
| all depths | decode tok/s (greedy, both passes) | 25.02, 25.00 | 18.24, 18.25 | 0.02 | -6.77 tok/s | -27.05 % | loss |
|  | decode ms a token | 39.96, 40.00 | 54.81, 54.80 | 0.04 | +14.83 ms | +37.08 % | loss |
|  | the step on pass 1, ms | 125.23, 125.46 | 145.63, 145.56 | 0.23 | +20.25 ms | +16.16 % | loss |
|  | the step on pass 2 (the repeated request), ms | 107.80, 107.77 | 125.83, 125.86 | 0.03 | +18.06 ms | +16.75 % | loss |
|  | share of drafted tokens accepted (greedy) | 0.6490, 0.6490 | 0.5021, 0.5021 | 0.0000 | -0.1469 | -22.64 % | loss |
|  | tokens a step (greedy) | 2.917, 2.917 | 2.479, 2.479 | 0.000 | -0.438 | -15.01 % | loss |
|  | sampler pass tok/s | 21.92, 21.50 | 19.95, 20.14 | 0.42 | -1.66 tok/s | -7.67 % | loss |
|  | sampler pass share accepted | 0.5778, 0.5778 | 0.6276, 0.6276 | 0.0000 | +0.0498 | +8.62 % | gain |
|  | sampler pass tokens a step | 2.709, 2.709 | 2.844, 2.844 | 0.000 | +0.135 | +5.00 % | gain |
|  | sampler pass ms a step | 123.61, 126.02 | 142.60, 141.26 | 2.41 | +17.12 ms | +13.71 % | loss |
|  | prompt tok/s (the prime) | 152.2, 150.6 | 122.5, 127.5 | 4.9 | -26.4 tok/s | -17.43 % | loss |

### The host's memory per load

From the RAM watchdog's samples, one a second: the lowest MemAvailable, the most swap in use and its change over the load, the highest memory pressure (psi full, 10 s average, %).

| Load | Samples | Lowest MemAvailable, MiB | Most swap in use, MiB | Swap change, MiB | Highest psi full |
|---|---:|---:|---:|---:|---:|
| pmix1 | 437 | 379 | 21 | +16 | 0.54 |
| pmix2 | 437 | 377 | 22 | +1 | 0.00 |
| gmix1 | 506 | 328 | 21 | +0 | 0.00 |
| gmix2 | 503 | 330 | 22 | +0 | 0.00 |
