## The deep lane: w4h1, w4g1, w5h1, w5g1, w1h1, w1g1, w4h2, w4g2, w5h2, w5g2, w1h2, w1g2, w4u

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: w4h1, w4g1, w5h1, w5g1, w1h1, w1g1, w4h2, w4g2, w5h2, w5g2, w1h2, w1g2, w4u1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| w4h1 (control) | w4h1 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w4h1.csv LLAMA_STEP_TRACE=<OUT>/st-w4h1.jsonl` |
| w4g1 | w4g1 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w4g1.csv GGML_CUDA_PROFILE_MODE=group` |
| w5h1 | w5h1 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w5h1.csv LLAMA_STEP_TRACE=<OUT>/st-w5h1.jsonl`, `--spec-draft-n-max 4` |
| w5g1 | w5g1 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w5g1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-draft-n-max 4` |
| w1h1 | w1h1 | off | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w1h1.csv LLAMA_STEP_TRACE=<OUT>/st-w1h1.jsonl` |
| w1g1 | w1g1 | off | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w1g1.csv GGML_CUDA_PROFILE_MODE=group` |
| w4h2 | w4h2 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w4h2.csv LLAMA_STEP_TRACE=<OUT>/st-w4h2.jsonl` |
| w4g2 | w4g2 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w4g2.csv GGML_CUDA_PROFILE_MODE=group` |
| w5h2 | w5h2 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w5h2.csv LLAMA_STEP_TRACE=<OUT>/st-w5h2.jsonl`, `--spec-draft-n-max 4` |
| w5g2 | w5g2 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w5g2.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-draft-n-max 4` |
| w1h2 | w1h2 | off | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w1h2.csv LLAMA_STEP_TRACE=<OUT>/st-w1h2.jsonl` |
| w1g2 | w1g2 | off | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-w1g2.csv GGML_CUDA_PROFILE_MODE=group` |
| w4u | w4u1 | on | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `w4h1`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `w4h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w4h1 | w4h1: 22.72, 23.87 → **23.28** (42.95 ms a token) | — | control | | |
|  | w4g1 | w4g1: 17.82, 18.17 → **17.99** (55.58 ms a token) | — | -5.29 tok/s | -22.7 % | no class (one load) |
|  | w5h1 | w5h1: 19.86, 21.34 → **20.57** (48.61 ms a token) | — | -2.71 tok/s | -11.6 % | no class (one load) |
|  | w5g1 | w5g1: 16.34, 17.28 → **16.80** (59.53 ms a token) | — | -6.48 tok/s | -27.8 % | no class (one load) |
|  | w1h1 | w1h1: 18.94, 19.53 → **19.23** (52.00 ms a token) | — | -4.05 tok/s | -17.4 % | no class (one load) |
|  | w1g1 | w1g1: 13.01, 13.01 → **13.01** (76.86 ms a token) | — | -10.27 tok/s | -44.1 % | no class (one load) |
|  | w4h2 | w4h2: 22.46, 23.88 → **23.15** (43.20 ms a token) | — | -0.13 tok/s | -0.6 % | no class (one load) |
|  | w4g2 | w4g2: 17.85, 18.35 → **18.09** (55.27 ms a token) | — | -5.19 tok/s | -22.3 % | no class (one load) |
|  | w5h2 | w5h2: 19.84, 21.33 → **20.55** (48.65 ms a token) | — | -2.73 tok/s | -11.7 % | no class (one load) |
|  | w5g2 | w5g2: 16.36, 17.33 → **16.83** (59.43 ms a token) | — | -6.45 tok/s | -27.7 % | no class (one load) |
|  | w1h2 | w1h2: 18.93, 19.53 → **19.23** (52.00 ms a token) | — | -4.05 tok/s | -17.4 % | no class (one load) |
|  | w1g2 | w1g2: 13.02, 13.01 → **13.02** (76.81 ms a token) | — | -10.26 tok/s | -44.1 % | no class (one load) |
|  | w4u | w4u1: 22.82, 24.04 → **23.41** (42.72 ms a token) | — | +0.13 tok/s | +0.6 % | no class (one load) |
| 4,096 | w4h1 | w4h1: 28.27, 29.26 → **28.76** (34.78 ms a token) | — | control | | |
|  | w4g1 | w4g1: 21.03, 22.99 → **21.97** (45.52 ms a token) | — | -6.79 tok/s | -23.6 % | no class (one load) |
|  | w5h1 | w5h1: 23.71, 24.21 → **23.96** (41.74 ms a token) | — | -4.80 tok/s | -16.7 % | no class (one load) |
|  | w5g1 | w5g1: 18.09, 19.65 → **18.84** (53.09 ms a token) | — | -9.92 tok/s | -34.5 % | no class (one load) |
|  | w1h1 | w1h1: 18.38, 18.55 → **18.47** (54.15 ms a token) | — | -10.29 tok/s | -35.8 % | no class (one load) |
|  | w1g1 | w1g1: 12.32, 12.49 → **12.40** (80.64 ms a token) | — | -16.35 tok/s | -56.9 % | no class (one load) |
|  | w4h2 | w4h2: 28.31, 29.37 → **28.83** (34.69 ms a token) | — | +0.07 tok/s | +0.2 % | no class (one load) |
|  | w4g2 | w4g2: 22.45, 23.04 → **22.74** (43.97 ms a token) | — | -6.01 tok/s | -20.9 % | no class (one load) |
|  | w5h2 | w5h2: 23.67, 24.23 → **23.95** (41.76 ms a token) | — | -4.81 tok/s | -16.7 % | no class (one load) |
|  | w5g2 | w5g2: 20.01, 19.69 → **19.85** (50.38 ms a token) | — | -8.91 tok/s | -31.0 % | no class (one load) |
|  | w1h2 | w1h2: 18.08, 18.56 → **18.31** (54.61 ms a token) | — | -10.44 tok/s | -36.3 % | no class (one load) |
|  | w1g2 | w1g2: 12.50, 12.15 → **12.32** (81.16 ms a token) | — | -16.44 tok/s | -57.2 % | no class (one load) |
|  | w4u | w4u1: 28.48, 29.50 → **28.98** (34.51 ms a token) | — | +0.22 tok/s | +0.8 % | no class (one load) |
| 8,192 | w4h1 | w4h1: 29.77, 31.43 → **30.58** (32.70 ms a token) | — | control | | |
|  | w4g1 | w4g1: 24.11, 24.68 → **24.39** (41.00 ms a token) | — | -6.19 tok/s | -20.2 % | no class (one load) |
|  | w5h1 | w5h1: 24.74, 25.67 → **25.20** (39.69 ms a token) | — | -5.38 tok/s | -17.6 % | no class (one load) |
|  | w5g1 | w5g1: 21.15, 21.69 → **21.42** (46.69 ms a token) | — | -9.16 tok/s | -30.0 % | no class (one load) |
|  | w1h1 | w1h1: 18.31, 18.47 → **18.39** (54.38 ms a token) | — | -12.19 tok/s | -39.9 % | no class (one load) |
|  | w1g1 | w1g1: 12.50, 12.53 → **12.51** (79.91 ms a token) | — | -18.06 tok/s | -59.1 % | no class (one load) |
|  | w4h2 | w4h2: 29.78, 31.38 → **30.56** (32.72 ms a token) | — | -0.02 tok/s | -0.1 % | no class (one load) |
|  | w4g2 | w4g2: 24.15, 24.52 → **24.33** (41.10 ms a token) | — | -6.24 tok/s | -20.4 % | no class (one load) |
|  | w5h2 | w5h2: 24.76, 26.02 → **25.37** (39.41 ms a token) | — | -5.21 tok/s | -17.0 % | no class (one load) |
|  | w5g2 | w5g2: 21.21, 21.63 → **21.42** (46.69 ms a token) | — | -9.16 tok/s | -30.0 % | no class (one load) |
|  | w1h2 | w1h2: 18.25, 18.46 → **18.35** (54.48 ms a token) | — | -12.22 tok/s | -40.0 % | no class (one load) |
|  | w1g2 | w1g2: 12.36, 12.36 → **12.36** (80.89 ms a token) | — | -18.22 tok/s | -59.6 % | no class (one load) |
|  | w4u | w4u1: 30.00, 31.40 → **30.68** (32.59 ms a token) | — | +0.11 tok/s | +0.3 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `w4h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | w4h1 | w4h1: **27.17** | — | control | | |
|  | w4g1 | w4g1: **21.11** | — | -6.05 tok/s | -22.3 % | no class (one load) |
|  | w5h1 | w5h1: **23.07** | — | -4.10 tok/s | -15.1 % | no class (one load) |
|  | w5g1 | w5g1: **18.83** | — | -8.33 tok/s | -30.7 % | no class (one load) |
|  | w1h1 | w1h1: **18.69** | — | -8.48 tok/s | -31.2 % | no class (one load) |
|  | w1g1 | w1g1: **12.64** | — | -14.53 tok/s | -53.5 % | no class (one load) |
|  | w4h2 | w4h2: **27.12** | — | -0.04 tok/s | -0.2 % | no class (one load) |
|  | w4g2 | w4g2: **21.38** | — | -5.79 tok/s | -21.3 % | no class (one load) |
|  | w5h2 | w5h2: **23.11** | — | -4.06 tok/s | -14.9 % | no class (one load) |
|  | w5g2 | w5g2: **19.17** | — | -8.00 tok/s | -29.4 % | no class (one load) |
|  | w1h2 | w1h2: **18.62** | — | -8.54 tok/s | -31.4 % | no class (one load) |
|  | w1g2 | w1g2: **12.56** | — | -14.61 tok/s | -53.8 % | no class (one load) |
|  | w4u | w4u1: **27.32** | — | +0.15 tok/s | +0.6 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w4h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w4h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w4h1 | w4h1: 103.36 (109 steps) | — | control | | |
|  | w4g1 | w4g1: 131.80 (109 steps) | — | +28.44 ms | +27.5 % | no class (one load) |
|  | w5h1 | w5h1: 143.26 (90 steps) | — | +39.90 ms | +38.6 % | no class (one load) |
|  | w5g1 | w5g1: 174.03 (90 steps) | — | +70.67 ms | +68.4 % | no class (one load) |
|  | w1h1 | w1h1: 52.80 (256 steps) | — | -50.56 ms | -48.9 % | no class (the draft differs) |
|  | w1g1 | w1g1: 76.85 (256 steps) | — | -26.52 ms | -25.7 % | no class (the draft differs) |
|  | w4h2 | w4h2: 104.56 (109 steps) | — | +1.20 ms | +1.2 % | no class (one load) |
|  | w4g2 | w4g2: 131.58 (109 steps) | — | +28.21 ms | +27.3 % | no class (one load) |
|  | w5h2 | w5h2: 143.39 (90 steps) | — | +40.03 ms | +38.7 % | no class (one load) |
|  | w5g2 | w5g2: 173.89 (90 steps) | — | +70.53 ms | +68.2 % | no class (one load) |
|  | w1h2 | w1h2: 52.81 (256 steps) | — | -50.55 ms | -48.9 % | no class (the draft differs) |
|  | w1g2 | w1g2: 76.79 (256 steps) | — | -26.57 ms | -25.7 % | no class (the draft differs) |
|  | w4u | w4u1: 102.93 (109 steps) | — | -0.43 ms | -0.4 % | no class (one load) |
| 4,096 | w4h1 | w4h1: 114.63 (79 steps) | — | control | | |
|  | w4g1 | w4g1: 154.07 (79 steps) | — | +39.44 ms | +34.4 % | no class (one load) |
|  | w5h1 | w5h1: 158.80 (68 steps) | — | +44.17 ms | +38.5 % | no class (one load) |
|  | w5g1 | w5g1: 208.16 (68 steps) | — | +93.53 ms | +81.6 % | no class (one load) |
|  | w1h1 | w1h1: 54.40 (256 steps) | — | -60.23 ms | -52.5 % | no class (the draft differs) |
|  | w1g1 | w1g1: 81.19 (256 steps) | — | -33.44 ms | -29.2 % | no class (the draft differs) |
|  | w4h2 | w4h2: 114.48 (79 steps) | — | -0.14 ms | -0.1 % | no class (one load) |
|  | w4g2 | w4g2: 144.33 (79 steps) | — | +29.71 ms | +25.9 % | no class (one load) |
|  | w5h2 | w5h2: 159.02 (68 steps) | — | +44.39 ms | +38.7 % | no class (one load) |
|  | w5g2 | w5g2: 188.13 (68 steps) | — | +73.50 ms | +64.1 % | no class (one load) |
|  | w1h2 | w1h2: 55.32 (256 steps) | — | -59.30 ms | -51.7 % | no class (the draft differs) |
|  | w1g2 | w1g2: 80.02 (256 steps) | — | -34.61 ms | -30.2 % | no class (the draft differs) |
|  | w4u | w4u1: 113.79 (79 steps) | — | -0.84 ms | -0.7 % | no class (one load) |
| 8,192 | w4h1 | w4h1: 119.44 (72 steps) | — | control | | |
|  | w4g1 | w4g1: 147.44 (72 steps) | — | +28.00 ms | +23.4 % | no class (one load) |
|  | w5h1 | w5h1: 172.43 (60 steps) | — | +52.99 ms | +44.4 % | no class (one load) |
|  | w5g1 | w5g1: 201.75 (60 steps) | — | +82.31 ms | +68.9 % | no class (one load) |
|  | w1h1 | w1h1: 54.62 (256 steps) | — | -64.82 ms | -54.3 % | no class (the draft differs) |
|  | w1g1 | w1g1: 80.00 (256 steps) | — | -39.44 ms | -33.0 % | no class (the draft differs) |
|  | w4h2 | w4h2: 119.39 (72 steps) | — | -0.05 ms | -0.0 % | no class (one load) |
|  | w4g2 | w4g2: 147.23 (72 steps) | — | +27.79 ms | +23.3 % | no class (one load) |
|  | w5h2 | w5h2: 172.34 (60 steps) | — | +52.90 ms | +44.3 % | no class (one load) |
|  | w5g2 | w5g2: 201.14 (60 steps) | — | +81.70 ms | +68.4 % | no class (one load) |
|  | w1h2 | w1h2: 54.78 (256 steps) | — | -64.66 ms | -54.1 % | no class (the draft differs) |
|  | w1g2 | w1g2: 80.89 (256 steps) | — | -38.55 ms | -32.3 % | no class (the draft differs) |
|  | w4u | w4u1: 118.52 (72 steps) | — | -0.92 ms | -0.8 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w4h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w4h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w4h1 | w4h1: 99.32 (108 steps) | — | control | | |
|  | w4g1 | w4g1: 130.48 (108 steps) | — | +31.17 ms | +31.4 % | no class (one load) |
|  | w5h1 | w5h1: 137.86 (87 steps) | — | +38.54 ms | +38.8 % | no class (one load) |
|  | w5g1 | w5g1: 170.31 (87 steps) | — | +70.99 ms | +71.5 % | no class (one load) |
|  | w1h1 | w1h1: 51.19 (256 steps) | — | -48.12 ms | -48.5 % | no class (the draft differs) |
|  | w1g1 | w1g1: 76.88 (256 steps) | — | -22.44 ms | -22.6 % | no class (the draft differs) |
|  | w4h2 | w4h2: 99.27 (108 steps) | — | -0.05 ms | -0.1 % | no class (one load) |
|  | w4g2 | w4g2: 129.21 (108 steps) | — | +29.89 ms | +30.1 % | no class (one load) |
|  | w5h2 | w5h2: 137.98 (87 steps) | — | +38.66 ms | +38.9 % | no class (one load) |
|  | w5g2 | w5g2: 169.84 (87 steps) | — | +70.52 ms | +71.0 % | no class (one load) |
|  | w1h2 | w1h2: 51.19 (256 steps) | — | -48.13 ms | -48.5 % | no class (the draft differs) |
|  | w1g2 | w1g2: 76.84 (256 steps) | — | -22.48 ms | -22.6 % | no class (the draft differs) |
|  | w4u | w4u1: 98.62 (108 steps) | — | -0.70 ms | -0.7 % | no class (one load) |
| 4,096 | w4h1 | w4h1: 110.75 (79 steps) | — | control | | |
|  | w4g1 | w4g1: 140.96 (79 steps) | — | +30.21 ms | +27.3 % | no class (one load) |
|  | w5h1 | w5h1: 155.50 (68 steps) | — | +44.75 ms | +40.4 % | no class (one load) |
|  | w5g1 | w5g1: 191.57 (68 steps) | — | +80.82 ms | +73.0 % | no class (one load) |
|  | w1h1 | w1h1: 53.90 (256 steps) | — | -56.85 ms | -51.3 % | no class (the draft differs) |
|  | w1g1 | w1g1: 80.08 (256 steps) | — | -30.67 ms | -27.7 % | no class (the draft differs) |
|  | w4h2 | w4h2: 110.34 (79 steps) | — | -0.41 ms | -0.4 % | no class (one load) |
|  | w4g2 | w4g2: 140.63 (79 steps) | — | +29.88 ms | +27.0 % | no class (one load) |
|  | w5h2 | w5h2: 155.38 (68 steps) | — | +44.63 ms | +40.3 % | no class (one load) |
|  | w5g2 | w5g2: 191.19 (68 steps) | — | +80.44 ms | +72.6 % | no class (one load) |
|  | w1h2 | w1h2: 53.89 (256 steps) | — | -56.87 ms | -51.3 % | no class (the draft differs) |
|  | w1g2 | w1g2: 82.31 (256 steps) | — | -28.44 ms | -25.7 % | no class (the draft differs) |
|  | w4u | w4u1: 109.85 (79 steps) | — | -0.90 ms | -0.8 % | no class (one load) |
| 8,192 | w4h1 | w4h1: 114.71 (71 steps) | — | control | | |
|  | w4g1 | w4g1: 146.12 (71 steps) | — | +31.41 ms | +27.4 % | no class (one load) |
|  | w5h1 | w5h1: 169.04 (59 steps) | — | +54.33 ms | +47.4 % | no class (one load) |
|  | w5g1 | w5g1: 200.03 (59 steps) | — | +85.32 ms | +74.4 % | no class (one load) |
|  | w1h1 | w1h1: 54.15 (256 steps) | — | -60.57 ms | -52.8 % | no class (the draft differs) |
|  | w1g1 | w1g1: 79.82 (256 steps) | — | -34.89 ms | -30.4 % | no class (the draft differs) |
|  | w4h2 | w4h2: 114.91 (71 steps) | — | +0.19 ms | +0.2 % | no class (one load) |
|  | w4g2 | w4g2: 147.05 (71 steps) | — | +32.33 ms | +28.2 % | no class (one load) |
|  | w5h2 | w5h2: 166.78 (59 steps) | — | +52.06 ms | +45.4 % | no class (one load) |
|  | w5g2 | w5g2: 200.63 (59 steps) | — | +85.92 ms | +74.9 % | no class (one load) |
|  | w1h2 | w1h2: 54.18 (256 steps) | — | -60.53 ms | -52.8 % | no class (the draft differs) |
|  | w1g2 | w1g2: 80.90 (256 steps) | — | -33.82 ms | -29.5 % | no class (the draft differs) |
|  | w4u | w4u1: 114.84 (71 steps) | — | +0.12 ms | +0.1 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `w4h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w4h1 | w4h1: 51.7 (64 read on 0 cached) | — | control | | |
|  | w4g1 | w4g1: 48.4 (64 read on 0 cached) | — | -3.3 tok/s | -6.4 % | no class (one load) |
|  | w5h1 | w5h1: 54.2 (64 read on 0 cached) | — | +2.5 tok/s | +4.7 % | no class (one load) |
|  | w5g1 | w5g1: 47.5 (64 read on 0 cached) | — | -4.3 tok/s | -8.2 % | no class (one load) |
|  | w1h1 | w1h1: 56.5 (64 read on 0 cached) | — | +4.7 tok/s | +9.2 % | no class (one load) |
|  | w1g1 | w1g1: 51.3 (64 read on 0 cached) | — | -0.4 tok/s | -0.8 % | no class (one load) |
|  | w4h2 | w4h2: 51.7 (64 read on 0 cached) | — | -0.0 tok/s | -0.0 % | no class (one load) |
|  | w4g2 | w4g2: 49.0 (64 read on 0 cached) | — | -2.8 tok/s | -5.3 % | no class (one load) |
|  | w5h2 | w5h2: 53.2 (64 read on 0 cached) | — | +1.5 tok/s | +2.9 % | no class (one load) |
|  | w5g2 | w5g2: 47.4 (64 read on 0 cached) | — | -4.3 tok/s | -8.4 % | no class (one load) |
|  | w1h2 | w1h2: 55.1 (64 read on 0 cached) | — | +3.4 tok/s | +6.5 % | no class (one load) |
|  | w1g2 | w1g2: 51.0 (64 read on 0 cached) | — | -0.7 tok/s | -1.4 % | no class (one load) |
|  | w4u | w4u1: 51.4 (64 read on 0 cached) | — | -0.3 tok/s | -0.6 % | no class (one load) |
| 4,096 | w4h1 | w4h1: 154.3 (4032 read on 0 cached) | — | control | | |
|  | w4g1 | w4g1: 83.3 (4032 read on 0 cached) | — | -71.0 tok/s | -46.0 % | no class (one load) |
|  | w5h1 | w5h1: 156.6 (4032 read on 0 cached) | — | +2.3 tok/s | +1.5 % | no class (one load) |
|  | w5g1 | w5g1: 83.1 (4032 read on 0 cached) | — | -71.2 tok/s | -46.1 % | no class (one load) |
|  | w1h1 | w1h1: 169.4 (4032 read on 0 cached) | — | +15.0 tok/s | +9.7 % | no class (one load) |
|  | w1g1 | w1g1: 90.7 (4032 read on 0 cached) | — | -63.6 tok/s | -41.2 % | no class (one load) |
|  | w4h2 | w4h2: 157.1 (4032 read on 0 cached) | — | +2.8 tok/s | +1.8 % | no class (one load) |
|  | w4g2 | w4g2: 84.8 (4032 read on 0 cached) | — | -69.6 tok/s | -45.1 % | no class (one load) |
|  | w5h2 | w5h2: 153.2 (4032 read on 0 cached) | — | -1.1 tok/s | -0.7 % | no class (one load) |
|  | w5g2 | w5g2: 84.5 (4032 read on 0 cached) | — | -69.8 tok/s | -45.2 % | no class (one load) |
|  | w1h2 | w1h2: 167.0 (4032 read on 0 cached) | — | +12.6 tok/s | +8.2 % | no class (one load) |
|  | w1g2 | w1g2: 90.9 (4032 read on 0 cached) | — | -63.4 tok/s | -41.1 % | no class (one load) |
|  | w4u | w4u1: 193.6 (4032 read on 0 cached) | — | +39.3 tok/s | +25.5 % | no class (one load) |
| 8,192 | w4h1 | w4h1: 128.7 (4612 read on 3516 cached) | — | control | | |
|  | w4g1 | w4g1: 69.6 (4612 read on 3516 cached) | — | -59.2 tok/s | -46.0 % | no class (one load) |
|  | w5h1 | w5h1: 129.5 (4612 read on 3516 cached) | — | +0.8 tok/s | +0.6 % | no class (one load) |
|  | w5g1 | w5g1: 70.8 (4612 read on 3516 cached) | — | -57.9 tok/s | -45.0 % | no class (one load) |
|  | w1h1 | w1h1: 138.3 (4612 read on 3516 cached) | — | +9.6 tok/s | +7.4 % | no class (one load) |
|  | w1g1 | w1g1: 72.8 (4612 read on 3516 cached) | — | -55.9 tok/s | -43.4 % | no class (one load) |
|  | w4h2 | w4h2: 129.8 (4612 read on 3516 cached) | — | +1.0 tok/s | +0.8 % | no class (one load) |
|  | w4g2 | w4g2: 70.2 (4612 read on 3516 cached) | — | -58.5 tok/s | -45.5 % | no class (one load) |
|  | w5h2 | w5h2: 129.1 (4612 read on 3516 cached) | — | +0.4 tok/s | +0.3 % | no class (one load) |
|  | w5g2 | w5g2: 68.9 (4612 read on 3516 cached) | — | -59.8 tok/s | -46.4 % | no class (one load) |
|  | w1h2 | w1h2: 137.2 (4612 read on 3516 cached) | — | +8.5 tok/s | +6.6 % | no class (one load) |
|  | w1g2 | w1g2: 73.4 (4612 read on 3516 cached) | — | -55.3 tok/s | -43.0 % | no class (one load) |
|  | w4u | w4u1: 226.9 (4612 read on 3516 cached) | — | +98.2 tok/s | +76.3 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | w4h1 | w4h1: 147 of 321 (0.458), 2.35 a step, 103.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 99.3 ms a step) |
|  | w4g1 | w4g1: 147 of 321 (0.458), 2.35 a step, 131.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 130.5 ms a step) |
|  | w5h1 | w5h1: 166 of 356 (0.466), 2.84 a step, 143.3 ms a step (pass 2: 169 of 344 (0.491), 2.94 a step, 137.9 ms a step) |
|  | w5g1 | w5g1: 166 of 356 (0.466), 2.84 a step, 174.0 ms a step (pass 2: 169 of 344 (0.491), 2.94 a step, 170.3 ms a step) |
|  | w1h1 | w1h1: no draft, 1.00 a step, 52.8 ms a step |
|  | w1g1 | w1g1: no draft, 1.00 a step, 76.8 ms a step |
|  | w4h2 | w4h2: 147 of 321 (0.458), 2.35 a step, 104.6 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 99.3 ms a step) |
|  | w4g2 | w4g2: 147 of 321 (0.458), 2.35 a step, 131.6 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 129.2 ms a step) |
|  | w5h2 | w5h2: 166 of 356 (0.466), 2.84 a step, 143.4 ms a step (pass 2: 169 of 344 (0.491), 2.94 a step, 138.0 ms a step) |
|  | w5g2 | w5g2: 166 of 356 (0.466), 2.84 a step, 173.9 ms a step (pass 2: 169 of 344 (0.491), 2.94 a step, 169.8 ms a step) |
|  | w1h2 | w1h2: no draft, 1.00 a step, 52.8 ms a step |
|  | w1g2 | w1g2: no draft, 1.00 a step, 76.8 ms a step |
|  | w4u | w4u1: 147 of 321 (0.458), 2.35 a step, 102.9 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.6 ms a step) |
| 4,096 | w4h1 | w4h1: 177 of 231 (0.766), 3.24 a step, 114.6 ms a step |
|  | w4g1 | w4g1: 177 of 231 (0.766), 3.24 a step, 154.1 ms a step |
|  | w5h1 | w5h1: 188 of 266 (0.707), 3.76 a step, 158.8 ms a step (pass 2: 188 of 264 (0.712), 3.76 a step, 155.5 ms a step) |
|  | w5g1 | w5g1: 188 of 266 (0.707), 3.76 a step, 208.2 ms a step (pass 2: 188 of 264 (0.712), 3.76 a step, 191.6 ms a step) |
|  | w1h1 | w1h1: no draft, 1.00 a step, 54.4 ms a step |
|  | w1g1 | w1g1: no draft, 1.00 a step, 81.2 ms a step |
|  | w4h2 | w4h2: 177 of 231 (0.766), 3.24 a step, 114.5 ms a step |
|  | w4g2 | w4g2: 177 of 231 (0.766), 3.24 a step, 144.3 ms a step |
|  | w5h2 | w5h2: 188 of 266 (0.707), 3.76 a step, 159.0 ms a step (pass 2: 188 of 264 (0.712), 3.76 a step, 155.4 ms a step) |
|  | w5g2 | w5g2: 188 of 266 (0.707), 3.76 a step, 188.1 ms a step (pass 2: 188 of 264 (0.712), 3.76 a step, 191.2 ms a step) |
|  | w1h2 | w1h2: no draft, 1.00 a step, 55.3 ms a step |
|  | w1g2 | w1g2: no draft, 1.00 a step, 80.0 ms a step |
|  | w4u | w4u1: 177 of 231 (0.766), 3.24 a step, 113.8 ms a step |
| 8,192 | w4h1 | w4h1: 184 of 212 (0.868), 3.56 a step, 119.4 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.7 ms a step) |
|  | w4g1 | w4g1: 184 of 212 (0.868), 3.56 a step, 147.4 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 146.1 ms a step) |
|  | w5h1 | w5h1: 196 of 233 (0.841), 4.27 a step, 172.4 ms a step (pass 2: 197 of 229 (0.860), 4.34 a step, 169.0 ms a step) |
|  | w5g1 | w5g1: 196 of 233 (0.841), 4.27 a step, 201.7 ms a step (pass 2: 197 of 229 (0.860), 4.34 a step, 200.0 ms a step) |
|  | w1h1 | w1h1: no draft, 1.00 a step, 54.6 ms a step |
|  | w1g1 | w1g1: no draft, 1.00 a step, 80.0 ms a step |
|  | w4h2 | w4h2: 184 of 212 (0.868), 3.56 a step, 119.4 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.9 ms a step) |
|  | w4g2 | w4g2: 184 of 212 (0.868), 3.56 a step, 147.2 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 147.0 ms a step) |
|  | w5h2 | w5h2: 196 of 233 (0.841), 4.27 a step, 172.3 ms a step (pass 2: 197 of 229 (0.860), 4.34 a step, 166.8 ms a step) |
|  | w5g2 | w5g2: 196 of 233 (0.841), 4.27 a step, 201.1 ms a step (pass 2: 197 of 229 (0.860), 4.34 a step, 200.6 ms a step) |
|  | w1h2 | w1h2: no draft, 1.00 a step, 54.8 ms a step |
|  | w1g2 | w1g2: no draft, 1.00 a step, 80.9 ms a step |
|  | w4u | w4u1: 184 of 212 (0.868), 3.56 a step, 118.5 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.8 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | w4h1 | w4h1: 24.12 tok/s, 154 of 302 (0.510), 2.51 a step, 104.1 ms a step |
|  | w4g1 | w4g1: 18.79 tok/s, 154 of 302 (0.510), 2.51 a step, 133.6 ms a step |
|  | w5h1 | w5h1: 21.96 tok/s, 175 of 318 (0.550), 3.16 a step, 143.9 ms a step |
|  | w5g1 | w5g1: 17.63 tok/s, 175 of 318 (0.550), 3.16 a step, 179.3 ms a step |
|  | w1h1 | w1h1: 19.03 tok/s, no draft, 1.00 a step, 52.5 ms a step |
|  | w1g1 | w1g1: 12.98 tok/s, no draft, 1.00 a step, 77.0 ms a step |
|  | w4h2 | w4h2: 23.85 tok/s, 154 of 302 (0.510), 2.51 a step, 105.2 ms a step |
|  | w4g2 | w4g2: 18.74 tok/s, 154 of 302 (0.510), 2.51 a step, 133.9 ms a step |
|  | w5h2 | w5h2: 21.71 tok/s, 175 of 318 (0.550), 3.16 a step, 145.6 ms a step |
|  | w5g2 | w5g2: 17.65 tok/s, 175 of 318 (0.550), 3.16 a step, 179.1 ms a step |
|  | w1h2 | w1h2: 19.04 tok/s, no draft, 1.00 a step, 52.5 ms a step |
|  | w1g2 | w1g2: 12.99 tok/s, no draft, 1.00 a step, 77.0 ms a step |
|  | w4u | w4u1: 24.02 tok/s, 154 of 302 (0.510), 2.51 a step, 104.5 ms a step |
| 4,096 | w4h1 | w4h1: 29.66 tok/s, 179 of 228 (0.785), 3.32 a step, 112.1 ms a step |
|  | w4g1 | w4g1: 19.58 tok/s, 179 of 228 (0.785), 3.32 a step, 169.8 ms a step |
|  | w5h1 | w5h1: 22.80 tok/s, 184 of 281 (0.655), 3.56 a step, 155.9 ms a step |
|  | w5g1 | w5g1: 18.96 tok/s, 184 of 281 (0.655), 3.56 a step, 187.5 ms a step |
|  | w1h1 | w1h1: 18.28 tok/s, no draft, 1.00 a step, 54.7 ms a step |
|  | w1g1 | w1g1: 12.32 tok/s, no draft, 1.00 a step, 81.2 ms a step |
|  | w4h2 | w4h2: 29.56 tok/s, 179 of 228 (0.785), 3.32 a step, 112.5 ms a step |
|  | w4g2 | w4g2: 23.13 tok/s, 179 of 228 (0.785), 3.32 a step, 143.7 ms a step |
|  | w5h2 | w5h2: 22.79 tok/s, 184 of 281 (0.655), 3.56 a step, 156.0 ms a step |
|  | w5g2 | w5g2: 18.71 tok/s, 184 of 281 (0.655), 3.56 a step, 190.1 ms a step |
|  | w1h2 | w1h2: 18.21 tok/s, no draft, 1.00 a step, 54.9 ms a step |
|  | w1g2 | w1g2: 12.41 tok/s, no draft, 1.00 a step, 80.6 ms a step |
|  | w4u | w4u1: 29.66 tok/s, 179 of 228 (0.785), 3.32 a step, 112.1 ms a step |
| 8,192 | w4h1 | w4h1: 29.40 tok/s, 182 of 216 (0.843), 3.46 a step, 117.7 ms a step |
|  | w4g1 | w4g1: 23.34 tok/s, 182 of 216 (0.843), 3.46 a step, 148.2 ms a step |
|  | w5h1 | w5h1: 22.09 tok/s, 189 of 262 (0.721), 3.82 a step, 173.0 ms a step |
|  | w5g1 | w5g1: 17.51 tok/s, 189 of 262 (0.721), 3.82 a step, 218.2 ms a step |
|  | w1h1 | w1h1: 18.08 tok/s, no draft, 1.00 a step, 55.3 ms a step |
|  | w1g1 | w1g1: 11.87 tok/s, no draft, 1.00 a step, 84.2 ms a step |
|  | w4h2 | w4h2: 29.33 tok/s, 182 of 216 (0.843), 3.46 a step, 117.9 ms a step |
|  | w4g2 | w4g2: 23.41 tok/s, 182 of 216 (0.843), 3.46 a step, 147.8 ms a step |
|  | w5h2 | w5h2: 21.99 tok/s, 189 of 262 (0.721), 3.82 a step, 173.7 ms a step |
|  | w5g2 | w5g2: 18.76 tok/s, 189 of 262 (0.721), 3.82 a step, 203.6 ms a step |
|  | w1h2 | w1h2: 18.08 tok/s, no draft, 1.00 a step, 55.3 ms a step |
|  | w1g2 | w1g2: 11.80 tok/s, no draft, 1.00 a step, 84.7 ms a step |
|  | w4u | w4u1: 29.64 tok/s, 182 of 216 (0.843), 3.46 a step, 116.7 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `w4h1` in w5h1, w5g1, w1h1, w1g1, w5h2, w5g2, w1h2, w1g2.**
- Greedy reply at depth 4,096: **differs from `w4h1` in w5h1, w5g1, w1h1, w1g1, w5h2, w5g2, w1h2, w1g2.**
- Greedy reply at depth 8,192: **differs from `w4h1` in w5h1, w5g1, w1h1, w1g1, w5h2, w5g2, w1h2, w1g2.**
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: w4h1: 0 of 12 requests; w4g1: 0 of 12 requests; w5h1: 0 of 12 requests; w5g1: 0 of 12 requests; w1h1: 0 of 12 requests; w1g1: 0 of 12 requests; w4h2: 0 of 12 requests; w4g2: 0 of 12 requests; w5h2: 0 of 12 requests; w5g2: 0 of 12 requests; w1h2: 0 of 12 requests; w1g2: 0 of 12 requests; w4u1: 0 of 12 requests.
