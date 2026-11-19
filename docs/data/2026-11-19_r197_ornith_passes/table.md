## The deep lane: w1h1, w2b1, w2r1, w3b1, w3r1, w4b1, w4r1, w2r2, w2b2, w3r2, w3b2, w4r2, w4b2

Model file `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: w1h1, w2b1, w2r1, w3b1, w3r1, w4b1, w4r1, w2r2, w2b2, w3r2, w3b2, w4r2, w4b2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| w1h1 (control) | w1h1 | off | 49152 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w1h1.csv LLAMA_STEP_TRACE=<OUT>/st-w1h1.jsonl` |
| w2b1 | w2b1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w2b1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w2r1 | w2r1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w2r1.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w3b1 | w3b1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w3b1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w3r1 | w3r1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w3r1.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w4b1 | w4b1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w4b1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |
| w4r1 | w4r1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w4r1.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |
| w2r2 | w2r2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w2r2.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w2b2 | w2b2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w2b2.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w3r2 | w3r2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w3r2.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w3b2 | w3b2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w3b2.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w4r2 | w4r2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w4r2.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |
| w4b2 | w4b2 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w4b2.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `w1h1`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 21.37, 21.92 → **21.64** (46.21 ms a token) | — | control | | |
|  | w2b1 | w2b1: 15.59, 15.89 → **15.74** (63.53 ms a token) | — | -5.90 tok/s | -27.3 % | no class (one load) |
|  | w2r1 | w2r1: 16.09, 16.32 → **16.21** (61.70 ms a token) | — | -5.43 tok/s | -25.1 % | no class (one load) |
|  | w3b1 | w3b1: 16.67, 16.98 → **16.82** (59.45 ms a token) | — | -4.82 tok/s | -22.3 % | no class (one load) |
|  | w3r1 | w3r1: 17.96, 18.19 → **18.08** (55.32 ms a token) | — | -3.56 tok/s | -16.5 % | no class (one load) |
|  | w4b1 | w4b1: 15.66, 15.86 → **15.75** (63.47 ms a token) | — | -5.89 tok/s | -27.2 % | no class (one load) |
|  | w4r1 | w4r1: 17.63, 17.72 → **17.67** (56.58 ms a token) | — | -3.97 tok/s | -18.3 % | no class (one load) |
|  | w2r2 | w2r2: 16.10, 16.30 → **16.20** (61.74 ms a token) | — | -5.44 tok/s | -25.2 % | no class (one load) |
|  | w2b2 | w2b2: 15.58, 15.89 → **15.74** (63.55 ms a token) | — | -5.90 tok/s | -27.3 % | no class (one load) |
|  | w3r2 | w3r2: 18.00, 18.18 → **18.09** (55.28 ms a token) | — | -3.55 tok/s | -16.4 % | no class (one load) |
|  | w3b2 | w3b2: 16.68, 16.96 → **16.82** (59.46 ms a token) | — | -4.82 tok/s | -22.3 % | no class (one load) |
|  | w4r2 | w4r2: 17.60, 17.73 → **17.66** (56.61 ms a token) | — | -3.98 tok/s | -18.4 % | no class (one load) |
|  | w4b2 | w4b2: 15.67, 15.85 → **15.76** (63.46 ms a token) | — | -5.88 tok/s | -27.2 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 18.08, 18.16 → **18.12** (55.19 ms a token) | — | control | | |
|  | w2b1 | w2b1: 14.10, 14.08 → **14.09** (70.96 ms a token) | — | -4.03 tok/s | -22.2 % | no class (one load) |
|  | w2r1 | w2r1: 12.90, 12.88 → **12.89** (77.59 ms a token) | — | -5.23 tok/s | -28.9 % | no class (one load) |
|  | w3b1 | w3b1: 15.00, 14.96 → **14.98** (66.77 ms a token) | — | -3.14 tok/s | -17.3 % | no class (one load) |
|  | w3r1 | w3r1: 12.14, 12.12 → **12.13** (82.45 ms a token) | — | -5.99 tok/s | -33.1 % | no class (one load) |
|  | w4b1 | w4b1: 14.89, 14.88 → **14.89** (67.18 ms a token) | — | -3.23 tok/s | -17.8 % | no class (one load) |
|  | w4r1 | w4r1: 13.55, 13.55 → **13.55** (73.79 ms a token) | — | -4.57 tok/s | -25.2 % | no class (one load) |
|  | w2r2 | w2r2: 12.85, 12.86 → **12.85** (77.80 ms a token) | — | -5.27 tok/s | -29.1 % | no class (one load) |
|  | w2b2 | w2b2: 14.10, 14.07 → **14.08** (71.01 ms a token) | — | -4.04 tok/s | -22.3 % | no class (one load) |
|  | w3r2 | w3r2: 12.15, 12.14 → **12.14** (82.34 ms a token) | — | -5.97 tok/s | -33.0 % | no class (one load) |
|  | w3b2 | w3b2: 14.98, 14.95 → **14.96** (66.83 ms a token) | — | -3.15 tok/s | -17.4 % | no class (one load) |
|  | w4r2 | w4r2: 13.57, 13.54 → **13.56** (73.76 ms a token) | — | -4.56 tok/s | -25.2 % | no class (one load) |
|  | w4b2 | w4b2: 14.88, 14.87 → **14.88** (67.21 ms a token) | — | -3.24 tok/s | -17.9 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | w1h1 | w1h1: **19.72** | — | control | | |
|  | w2b1 | w2b1: **14.87** | — | -4.85 tok/s | -24.6 % | no class (one load) |
|  | w2r1 | w2r1: **14.36** | — | -5.37 tok/s | -27.2 % | no class (one load) |
|  | w3b1 | w3b1: **15.85** | — | -3.88 tok/s | -19.7 % | no class (one load) |
|  | w3r1 | w3r1: **14.52** | — | -5.21 tok/s | -26.4 % | no class (one load) |
|  | w4b1 | w4b1: **15.31** | — | -4.42 tok/s | -22.4 % | no class (one load) |
|  | w4r1 | w4r1: **15.34** | — | -4.38 tok/s | -22.2 % | no class (one load) |
|  | w2r2 | w2r2: **14.33** | — | -5.39 tok/s | -27.3 % | no class (one load) |
|  | w2b2 | w2b2: **14.86** | — | -4.86 tok/s | -24.6 % | no class (one load) |
|  | w3r2 | w3r2: **14.53** | — | -5.19 tok/s | -26.3 % | no class (one load) |
|  | w3b2 | w3b2: **15.84** | — | -3.89 tok/s | -19.7 % | no class (one load) |
|  | w4r2 | w4r2: **15.34** | — | -4.38 tok/s | -22.2 % | no class (one load) |
|  | w4b2 | w4b2: **15.31** | — | -4.42 tok/s | -22.4 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w1h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 46.79 (256 steps) | — | control | | |
|  | w2b1 | w2b1: 121.60 (135 steps) | — | +74.82 ms | +159.9 % | no class (one load) |
|  | w2r1 | w2r1: 117.86 (135 steps) | — | +71.07 ms | +151.9 % | no class (one load) |
|  | w3b1 | w3b1: 153.59 (100 steps) | — | +106.80 ms | +228.3 % | no class (one load) |
|  | w3r1 | w3r1: 148.44 (96 steps) | — | +101.65 ms | +217.3 % | no class (one load) |
|  | w4b1 | w4b1: 179.69 (91 steps) | — | +132.91 ms | +284.1 % | no class (one load) |
|  | w4r1 | w4r1: 183.82 (79 steps) | — | +137.04 ms | +292.9 % | no class (one load) |
|  | w2r2 | w2r2: 117.81 (135 steps) | — | +71.02 ms | +151.8 % | no class (one load) |
|  | w2b2 | w2b2: 121.69 (135 steps) | — | +74.90 ms | +160.1 % | no class (one load) |
|  | w3r2 | w3r2: 148.18 (96 steps) | — | +101.39 ms | +216.7 % | no class (one load) |
|  | w3b2 | w3b2: 153.47 (100 steps) | — | +106.68 ms | +228.0 % | no class (one load) |
|  | w4r2 | w4r2: 184.11 (79 steps) | — | +137.32 ms | +293.5 % | no class (one load) |
|  | w4b2 | w4b2: 179.56 (91 steps) | — | +132.77 ms | +283.8 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 55.30 (256 steps) | — | control | | |
|  | w2b1 | w2b1: 134.45 (135 steps) | — | +79.15 ms | +143.1 % | no class (one load) |
|  | w2r1 | w2r1: 131.46 (151 steps) | — | +76.15 ms | +137.7 % | no class (one load) |
|  | w3b1 | w3b1: 174.17 (98 steps) | — | +118.87 ms | +214.9 % | no class (one load) |
|  | w3r1 | w3r1: 170.10 (124 steps) | — | +114.80 ms | +207.6 % | no class (one load) |
|  | w4b1 | w4b1: 204.72 (84 steps) | — | +149.42 ms | +270.2 % | no class (one load) |
|  | w4r1 | w4r1: 207.63 (91 steps) | — | +152.32 ms | +275.4 % | no class (one load) |
|  | w2r2 | w2r2: 131.96 (151 steps) | — | +76.66 ms | +138.6 % | no class (one load) |
|  | w2b2 | w2b2: 134.51 (135 steps) | — | +79.21 ms | +143.2 % | no class (one load) |
|  | w3r2 | w3r2: 169.98 (124 steps) | — | +114.67 ms | +207.4 % | no class (one load) |
|  | w3b2 | w3b2: 174.42 (98 steps) | — | +119.11 ms | +215.4 % | no class (one load) |
|  | w4r2 | w4r2: 207.28 (91 steps) | — | +151.98 ms | +274.8 % | no class (one load) |
|  | w4b2 | w4b2: 204.76 (84 steps) | — | +149.46 ms | +270.3 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w1h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 45.63 (256 steps) | — | control | | |
|  | w2b1 | w2b1: 119.35 (135 steps) | — | +73.72 ms | +161.6 % | no class (one load) |
|  | w2r1 | w2r1: 116.16 (135 steps) | — | +70.53 ms | +154.6 % | no class (one load) |
|  | w3b1 | w3b1: 150.78 (100 steps) | — | +105.15 ms | +230.4 % | no class (one load) |
|  | w3r1 | w3r1: 146.60 (96 steps) | — | +100.97 ms | +221.3 % | no class (one load) |
|  | w4b1 | w4b1: 177.43 (91 steps) | — | +131.80 ms | +288.8 % | no class (one load) |
|  | w4r1 | w4r1: 182.90 (79 steps) | — | +137.27 ms | +300.8 % | no class (one load) |
|  | w2r2 | w2r2: 116.35 (135 steps) | — | +70.72 ms | +155.0 % | no class (one load) |
|  | w2b2 | w2b2: 119.31 (135 steps) | — | +73.68 ms | +161.5 % | no class (one load) |
|  | w3r2 | w3r2: 146.66 (96 steps) | — | +101.04 ms | +221.4 % | no class (one load) |
|  | w3b2 | w3b2: 150.96 (100 steps) | — | +105.33 ms | +230.8 % | no class (one load) |
|  | w4r2 | w4r2: 182.78 (79 steps) | — | +137.15 ms | +300.6 % | no class (one load) |
|  | w4b2 | w4b2: 177.48 (91 steps) | — | +131.85 ms | +289.0 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 55.08 (256 steps) | — | control | | |
|  | w2b1 | w2b1: 134.66 (135 steps) | — | +79.58 ms | +144.5 % | no class (one load) |
|  | w2r1 | w2r1: 131.63 (151 steps) | — | +76.55 ms | +139.0 % | no class (one load) |
|  | w3b1 | w3b1: 174.65 (98 steps) | — | +119.57 ms | +217.1 % | no class (one load) |
|  | w3r1 | w3r1: 170.33 (124 steps) | — | +115.25 ms | +209.2 % | no class (one load) |
|  | w4b1 | w4b1: 204.75 (84 steps) | — | +149.67 ms | +271.7 % | no class (one load) |
|  | w4r1 | w4r1: 207.57 (91 steps) | — | +152.49 ms | +276.9 % | no class (one load) |
|  | w2r2 | w2r2: 131.84 (151 steps) | — | +76.76 ms | +139.4 % | no class (one load) |
|  | w2b2 | w2b2: 134.79 (135 steps) | — | +79.71 ms | +144.7 % | no class (one load) |
|  | w3r2 | w3r2: 170.03 (124 steps) | — | +114.95 ms | +208.7 % | no class (one load) |
|  | w3b2 | w3b2: 174.71 (98 steps) | — | +119.64 ms | +217.2 % | no class (one load) |
|  | w4r2 | w4r2: 207.73 (91 steps) | — | +152.65 ms | +277.1 % | no class (one load) |
|  | w4b2 | w4b2: 204.90 (84 steps) | — | +149.82 ms | +272.0 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 43.8 (64 read on 0 cached) | — | control | | |
|  | w2b1 | w2b1: 30.1 (64 read on 0 cached) | — | -13.8 tok/s | -31.4 % | no class (one load) |
|  | w2r1 | w2r1: 30.8 (64 read on 0 cached) | — | -13.1 tok/s | -29.8 % | no class (one load) |
|  | w3b1 | w3b1: 31.6 (64 read on 0 cached) | — | -12.2 tok/s | -27.9 % | no class (one load) |
|  | w3r1 | w3r1: 30.7 (64 read on 0 cached) | — | -13.1 tok/s | -30.0 % | no class (one load) |
|  | w4b1 | w4b1: 30.5 (64 read on 0 cached) | — | -13.3 tok/s | -30.4 % | no class (one load) |
|  | w4r1 | w4r1: 30.7 (64 read on 0 cached) | — | -13.1 tok/s | -29.9 % | no class (one load) |
|  | w2r2 | w2r2: 32.6 (64 read on 0 cached) | — | -11.2 tok/s | -25.6 % | no class (one load) |
|  | w2b2 | w2b2: 28.9 (64 read on 0 cached) | — | -14.9 tok/s | -34.0 % | no class (one load) |
|  | w3r2 | w3r2: 29.6 (64 read on 0 cached) | — | -14.2 tok/s | -32.4 % | no class (one load) |
|  | w3b2 | w3b2: 31.7 (64 read on 0 cached) | — | -12.1 tok/s | -27.6 % | no class (one load) |
|  | w4r2 | w4r2: 30.6 (64 read on 0 cached) | — | -13.2 tok/s | -30.2 % | no class (one load) |
|  | w4b2 | w4b2: 29.5 (64 read on 0 cached) | — | -14.4 tok/s | -32.8 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 124.9 (8128 read on 0 cached) | — | control | | |
|  | w2b1 | w2b1: 48.6 (8128 read on 0 cached) | — | -76.3 tok/s | -61.1 % | no class (one load) |
|  | w2r1 | w2r1: 48.7 (8128 read on 0 cached) | — | -76.2 tok/s | -61.0 % | no class (one load) |
|  | w3b1 | w3b1: 48.6 (8128 read on 0 cached) | — | -76.3 tok/s | -61.1 % | no class (one load) |
|  | w3r1 | w3r1: 48.8 (8128 read on 0 cached) | — | -76.1 tok/s | -61.0 % | no class (one load) |
|  | w4b1 | w4b1: 48.8 (8128 read on 0 cached) | — | -76.2 tok/s | -61.0 % | no class (one load) |
|  | w4r1 | w4r1: 48.8 (8128 read on 0 cached) | — | -76.2 tok/s | -61.0 % | no class (one load) |
|  | w2r2 | w2r2: 48.6 (8128 read on 0 cached) | — | -76.3 tok/s | -61.1 % | no class (one load) |
|  | w2b2 | w2b2: 48.6 (8128 read on 0 cached) | — | -76.3 tok/s | -61.1 % | no class (one load) |
|  | w3r2 | w3r2: 48.8 (8128 read on 0 cached) | — | -76.1 tok/s | -61.0 % | no class (one load) |
|  | w3b2 | w3b2: 48.8 (8128 read on 0 cached) | — | -76.1 tok/s | -60.9 % | no class (one load) |
|  | w4r2 | w4r2: 48.6 (8128 read on 0 cached) | — | -76.3 tok/s | -61.1 % | no class (one load) |
|  | w4b2 | w4b2: 48.9 (8128 read on 0 cached) | — | -76.0 tok/s | -60.8 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | w1h1 | w1h1: no draft, 1.00 a step, 46.8 ms a step |
|  | w2b1 | w2b1: 121 of 134 (0.903), 1.90 a step, 121.6 ms a step |
|  | w2r1 | w2r1: 121 of 134 (0.903), 1.90 a step, 117.9 ms a step |
|  | w3b1 | w3b1: 156 of 197 (0.792), 2.56 a step, 153.6 ms a step |
|  | w3r1 | w3r1: 160 of 189 (0.847), 2.67 a step, 148.4 ms a step |
|  | w4b1 | w4b1: 165 of 265 (0.623), 2.81 a step, 179.7 ms a step |
|  | w4r1 | w4r1: 177 of 232 (0.763), 3.24 a step, 183.8 ms a step |
|  | w2r2 | w2r2: 121 of 134 (0.903), 1.90 a step, 117.8 ms a step |
|  | w2b2 | w2b2: 121 of 134 (0.903), 1.90 a step, 121.7 ms a step |
|  | w3r2 | w3r2: 160 of 189 (0.847), 2.67 a step, 148.2 ms a step |
|  | w3b2 | w3b2: 156 of 197 (0.792), 2.56 a step, 153.5 ms a step |
|  | w4r2 | w4r2: 177 of 232 (0.763), 3.24 a step, 184.1 ms a step |
|  | w4b2 | w4b2: 165 of 265 (0.623), 2.81 a step, 179.6 ms a step |
| 8,192 | w1h1 | w1h1: no draft, 1.00 a step, 55.3 ms a step |
|  | w2b1 | w2b1: 121 of 133 (0.910), 1.90 a step, 134.4 ms a step |
|  | w2r1 | w2r1: 105 of 150 (0.700), 1.70 a step, 131.5 ms a step |
|  | w3b1 | w3b1: 158 of 193 (0.819), 2.61 a step, 174.2 ms a step |
|  | w3r1 | w3r1: 132 of 245 (0.539), 2.06 a step, 170.1 ms a step |
|  | w4b1 | w4b1: 172 of 247 (0.696), 3.05 a step, 204.7 ms a step |
|  | w4r1 | w4r1: 165 of 268 (0.616), 2.81 a step, 207.6 ms a step |
|  | w2r2 | w2r2: 105 of 150 (0.700), 1.70 a step, 132.0 ms a step |
|  | w2b2 | w2b2: 121 of 133 (0.910), 1.90 a step, 134.5 ms a step |
|  | w3r2 | w3r2: 132 of 245 (0.539), 2.06 a step, 170.0 ms a step |
|  | w3b2 | w3b2: 158 of 193 (0.819), 2.61 a step, 174.4 ms a step |
|  | w4r2 | w4r2: 165 of 268 (0.616), 2.81 a step, 207.3 ms a step |
|  | w4b2 | w4b2: 172 of 247 (0.696), 3.05 a step, 204.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `w1h1` in w2r1, w3r1, w4b1, w4r1, w2r2, w3r2, w4r2, w4b2.**
- Greedy reply at depth 8,192: **differs from `w1h1` in w2b1, w2r1, w3b1, w3r1, w4b1, w4r1, w2r2, w2b2, w3r2, w3b2, w4r2, w4b2.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: w1h1: 0 of 6 requests; w2b1: 0 of 6 requests; w2r1: 0 of 6 requests; w3b1: 0 of 6 requests; w3r1: 0 of 6 requests; w4b1: 0 of 6 requests; w4r1: 0 of 6 requests; w2r2: 0 of 6 requests; w2b2: 0 of 6 requests; w3r2: 0 of 6 requests; w3b2: 0 of 6 requests; w4r2: 0 of 6 requests; w4b2: 0 of 6 requests.
