## Generation speed by context depth: v2.0 against v3

Tokens per second as llama-server reports them (`timings.predicted_per_second`), 256 greedy tokens per request, each request twice per load (pass 1, pass 2), loads in the order v2a, v3a, v2b, v3b. A load's speed is its two passes' tokens over their time. Spread: the larger of the two builds' differences between their two loads. Class: gain or loss when the difference of the means is beyond the spread, else negligible.

| Depth (tokens) | v2.0 v2a: pass 1, pass 2 → load | v2.0 v2b: pass 1, pass 2 → load | v3 v3a: pass 1, pass 2 → load | v3 v3b: pass 1, pass 2 → load | Spread | v3 − v2.0 (means) | | Class |
|---:|---|---|---|---|---:|---:|---:|---|
| 0 | 48.40, 54.77 → **51.59** | 47.82, 54.59 → **51.18** | 47.15, 54.56 → **50.79** | 47.36, 54.53 → **50.89** | 0.41 | -0.55 tok/s | -1.1 % | loss |
| 8,192 | 44.03, 46.41 → **45.37** | 44.20, 46.52 → **45.51** | 43.30, 46.52 → **45.03** | 44.17, 46.46 → **45.46** | 0.43 | -0.19 tok/s | -0.4 % | negligible |
| 16,384 | 43.15, 44.36 → **43.92** | 43.13, 44.32 → **43.89** | 43.01, 44.20 → **43.77** | 43.08, 44.21 → **43.81** | 0.04 | -0.11 tok/s | -0.3 % | loss |
| 24,576 | 35.05, 35.87 → **35.59** | 34.99, 35.79 → **35.53** | 35.07, 35.83 → **35.58** | 35.00, 35.76 → **35.52** | 0.07 | -0.01 tok/s | -0.0 % | negligible |
| 32,768 | 34.00, 34.92 → **34.59** | 34.06, 34.95 → **34.64** | 33.97, 34.97 → **34.60** | 33.97, 34.87 → **34.55** | 0.05 | -0.04 tok/s | -0.1 % | negligible |
| all five | **41.25** | **41.21** | **41.07** | **41.13** | 0.06 | -0.14 tok/s | -0.3 % | loss |

### The pipelined guesses (`--pipe-depth 4`), pass 1 and pass 2: right of all guesses

| Depth (tokens) | v2.0 v2a | v2.0 v2b | v3 v3a | v3 v3b |
|---:|---|---|---|---|
| 0 | 50 of 447, 50 of 447 | 50 of 447, 50 of 447 | 50 of 447, 50 of 447 | 50 of 447, 50 of 447 |
| 8,192 | 54 of 687, 54 of 687 | 54 of 687, 54 of 687 | 54 of 687, 54 of 687 | 54 of 687, 54 of 687 |
| 16,384 | 77 of 627, 77 of 627 | 77 of 627, 77 of 627 | 77 of 627, 77 of 627 | 77 of 627, 77 of 627 |
| 24,576 | 53 of 692, 53 of 692 | 53 of 692, 53 of 692 | 53 of 692, 53 of 692 | 53 of 692, 53 of 692 |
| 32,768 | 75 of 650, 75 of 650 | 75 of 650, 75 of 650 | 75 of 650, 75 of 650 | 75 of 650, 75 of 650 |

### Prompt reading at each depth (the whole prompt fresh), pass 1, pass 2 → load, tok/s

| Depth (tokens) | Prompt tokens | v2.0 v2a | v2.0 v2b | v3 v3a | v3 v3b | Spread | v3 − v2.0 (means) | | Class |
|---:|---:|---|---|---|---|---:|---:|---:|---|
| 0 | 14 | 32.0, 44.3 → **37.2** | 31.5, 43.7 → **36.6** | 30.3, 41.6 → **35.0** | 30.0, 42.0 → **35.0** | 0.5 | -1.9 tok/s | -5.1 % | loss |
| 8,192 | 8,192 | 520.2, 551.1 → **535.2** | 527.2, 550.2 → **538.5** | 515.4, 541.7 → **528.2** | 514.8, 541.5 → **527.8** | 3.3 | -8.8 tok/s | -1.6 % | loss |
| 16,384 | 16,384 | 504.8, 507.0 → **505.9** | 506.6, 507.3 → **506.9** | 499.1, 500.5 → **499.8** | 499.2, 500.5 → **499.8** | 1.1 | -6.6 tok/s | -1.3 % | loss |
| 24,576 | 24,576 | 455.8, 458.6 → **457.2** | 457.4, 460.3 → **458.9** | 452.6, 454.3 → **453.4** | 452.2, 454.6 → **453.4** | 1.7 | -4.6 tok/s | -1.0 % | loss |
| 32,768 | 32,768 | 402.6, 406.0 → **404.3** | 403.1, 407.3 → **405.2** | 400.4, 404.6 → **402.5** | 399.9, 403.5 → **401.7** | 0.9 | -2.6 tok/s | -0.7 % | loss |

### Checks

- Prompts: the chat template's 1 head and 13 tail ids around the code of the first 11 files; prompt ids per depth 14, 8,192, 16,384, 24,576, 32,768; the same ids (md5 per depth) on every load.
- Tokens out: 256 on every timed request.
- Generated ids, v2.0: the same on both passes of both loads at every depth.
- Generated ids, v3: the same on both passes of both loads at every depth.
- Generated ids, v2.0 against v3: the same at every depth.
