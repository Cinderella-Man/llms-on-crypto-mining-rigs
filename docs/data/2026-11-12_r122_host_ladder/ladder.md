## Each rung against the rung before it

Step, ms: the mean of the greedy and the long request's median typical step (S37's `arms.py`). Reference: the two loads of the rung before; spread = their difference. The rung's line: the mean of its two loads. Paired: the greedy request's steps against the rung before's first load, step by step (mean difference and its standard error), where the replies are the same.

| rung | switch added | load | step, ms | greedy, ms | long, ms | reference loads, ms | minus their mean | spread | class | greedy paired, ms | greedy reply = base's | lowest MemAvailable, MiB |
|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---|---:|
| a | LLAMA_DRAFT_PREFETCH=1 (C615) | a | 114.76 | 110.57 | 118.95 | base 119.22, base-2 119.09 | -4.39 | 0.13 | gain | -4.50 ± 0.20 (81) | yes | 505 |
| a | LLAMA_DRAFT_PREFETCH=1 (C615) | a-2 | 114.66 | 110.43 | 118.89 | base 119.22, base-2 119.09 | -4.49 | 0.13 | gain | -4.80 ± 0.17 (81) | yes | 520 |
| **a** | | **both** | **114.71** | | | 119.15 (mean) | **-4.44** | 0.13 | **gain** | | | |
| b | LLAMA_VERIFY_PREPARE=2 (C738) | b | 113.85 | 109.94 | 117.75 | a 114.76, a-2 114.66 | -0.86 | 0.10 | gain | -0.58 ± 0.22 (81) | yes | 517 |
| b | LLAMA_VERIFY_PREPARE=2 (C738) | b-2 | 114.18 | 110.13 | 118.22 | a 114.76, a-2 114.66 | -0.53 | 0.10 | gain | -0.41 ± 0.15 (81) | yes | 522 |
| **b** | | **both** | **114.01** | | | 114.71 (mean) | **-0.70** | 0.10 | **gain** | | | |
| c | LLAMA_MTP_GRAPH_KEEP=1 (C619) | c | 112.89 | 108.72 | 117.05 | b 113.85, b-2 114.18 | -1.13 | 0.33 | gain | -1.32 ± 0.19 (81) | yes | 519 |
| c | LLAMA_MTP_GRAPH_KEEP=1 (C619) | c-2 | 113.08 | 108.97 | 117.19 | b 113.85, b-2 114.18 | -0.94 | 0.33 | gain | -1.34 ± 0.20 (81) | yes | 513 |
| **c** | | **both** | **112.98** | | | 114.01 (mean) | **-1.03** | 0.33 | **gain** | | | |
| d | LLAMA_KV_SEQ_RM_BOUND=1 (C740) | d | 112.65 | 108.48 | 116.83 | c 112.89, c-2 113.08 | -0.33 | 0.19 | gain | -0.48 ± 0.08 (81) | yes | 503 |
| d | LLAMA_KV_SEQ_RM_BOUND=1 (C740) | d-2 | 112.64 | 108.54 | 116.75 | c 112.89, c-2 113.08 | -0.34 | 0.19 | gain | -0.25 ± 0.15 (81) | yes | 478 |
| **d** | | **both** | **112.65** | | | 112.98 (mean) | **-0.33** | 0.19 | **gain** | | | |
| e | LLAMA_SERVER_YIELD=2 (C741) | e | 112.44 | 108.32 | 116.56 | d 112.65, d-2 112.64 | -0.21 | 0.01 | gain | +0.03 ± 0.15 (81) | yes | 505 |
| e | LLAMA_SERVER_YIELD=2 (C741) | e-2 | 112.64 | 108.51 | 116.76 | d 112.65, d-2 112.64 | -0.01 | 0.01 | gain | +0.11 ± 0.16 (81) | yes | 513 |
| **e** | | **both** | **112.54** | | | 112.65 (mean) | **-0.11** | 0.01 | **gain** | | | |
| f | LLAMA_MTP_EMBD_FIRST=1, LLAMA_MTP_DRAFT_CAP=1, LLAMA_MTP_SKIP_LAST_ROW=1 (C771, C744, C742) | f | 111.89 | 108.09 | 115.68 | e 112.44, e-2 112.64 | -0.65 | 0.20 | gain | — | yes | 519 |
| f | LLAMA_MTP_EMBD_FIRST=1, LLAMA_MTP_DRAFT_CAP=1, LLAMA_MTP_SKIP_LAST_ROW=1 (C771, C744, C742) | f-2 | 111.99 | 107.92 | 116.06 | e 112.44, e-2 112.64 | -0.55 | 0.20 | gain | — | yes | 510 |
| **f** | | **both** | **111.94** | | | 112.54 (mean) | **-0.60** | 0.20 | **gain** | | | |
| stack | the stack (a+b+c+d+e+f) against base | f | 111.89 | 108.09 | 115.68 | base 119.22, base-2 119.09 | -7.26 | 0.13 | gain | — | yes | 519 |
| stack | the stack (a+b+c+d+e+f) against base | f-2 | 111.99 | 107.92 | 116.06 | base 119.22, base-2 119.09 | -7.16 | 0.13 | gain | — | yes | 510 |
| **stack** | | **both** | **111.94** | | | 119.15 (mean) | **-7.21** | 0.13 | **gain** | | | |

## Every load: the server's counts and the host

| load | greedy: generated, drafted, accepted | greedy: tokens a step | greedy: ms a step (server) | long: generated | long: tokens a step | long: ms a step (server) | greedy reply = base's | lowest MemAvailable, MiB | swap at start / most, MiB | major faults | pswpin | pswpout |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| base | 256, 254, 170 | 2.977 | 156.136 | 256 | 2.753 | 145.281 | yes | 499 | 19 / 19 | 3347 | 6 | 0 |
| a | 256, 254, 170 | 2.977 | 151.794 | 256 | 2.753 | 130.466 | yes | 505 | 19 / 19 | 15 | 14 | 0 |
| b | 256, 254, 170 | 2.977 | 151.378 | 256 | 3.282 | 130.688 | yes | 517 | 19 / 19 | 0 | 0 | 0 |
| c | 256, 254, 170 | 2.977 | 150.102 | 256 | 2.943 | 129.007 | yes | 519 | 19 / 19 | 2 | 1 | 0 |
| d | 256, 254, 170 | 2.977 | 149.615 | 256 | 2.943 | 129.162 | yes | 503 | 19 / 19 | 1 | 1 | 0 |
| e | 256, 254, 170 | 2.977 | 148.976 | 256 | 3.012 | 141.688 | yes | 505 | 19 / 19 | 1 | 0 | 0 |
| f | 256, 254, 170 | 2.977 | 126.240 | 256 | 3.084 | 128.007 | yes | 519 | 19 / 19 | 1 | 0 | 0 |
| base-2 | 256, 254, 170 | 2.977 | 156.004 | 256 | 2.977 | 134.532 | yes | 515 | 19 / 19 | 3 | 1 | 0 |
| a-2 | 256, 254, 170 | 2.977 | 151.081 | 256 | 2.909 | 131.435 | yes | 520 | 19 / 19 | 0 | 0 | 0 |
| b-2 | 256, 254, 170 | 2.977 | 151.117 | 256 | 2.876 | 130.524 | yes | 522 | 19 / 19 | 0 | 0 | 0 |
| c-2 | 256, 254, 170 | 2.977 | 149.898 | 256 | 3.048 | 128.985 | yes | 513 | 19 / 19 | 2 | 1 | 0 |
| d-2 | 256, 254, 170 | 2.977 | 149.646 | 256 | 2.977 | 140.377 | yes | 478 | 19 / 19 | 1 | 0 | 0 |
| e-2 | 256, 254, 170 | 2.977 | 149.195 | 256 | 2.813 | 139.749 | yes | 513 | 19 / 19 | 3 | 1 | 0 |
| f-2 | 256, 254, 170 | 2.977 | 125.869 | 256 | 2.753 | 138.320 | yes | 510 | 19 / 19 | 3 | 3 | 0 |

## The server's time outside the profiled steps

Prompt and generation: the server's `prompt_ms` and `predicted_ms` for the reply. Step 0: the request's first generation step in the host profile, shown apart because the long request's step 0 is not an ordinary step (it lasts about 7 s, and the server books part of it as prompt time). The rest: the sum of the request's other generation steps (typical or not) and their count. Outside: generation minus the rest, the time the server counts as generation that the steps after the first do not cover (it holds step 0 too where that is an ordinary step, as in the greedy request). The greedy request is the first after the server starts.

| load | greedy: prompt, ms | greedy: generation, ms | greedy: step 0, ms | greedy: the rest, ms (steps) | greedy: outside, ms | long: prompt, ms | long: generation, ms | long: step 0, ms | long: the rest, ms (steps) | long: outside, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 8571.4 | 13427.7 | 119.8 | 9753.6 (83) | 3674.1 | 26388.4 | 13511.2 | 6936.0 | 11241.0 (90) | 2270.2 |
| a | 8530.0 | 13054.3 | 115.3 | 9386.4 (83) | 3667.8 | 26478.0 | 12133.3 | 6952.3 | 10876.1 (90) | 1257.2 |
| b | 8522.4 | 13018.5 | 115.1 | 9339.0 (83) | 3679.5 | 26482.8 | 10193.7 | 6957.4 | 8941.5 (75) | 1252.1 |
| c | 8522.4 | 12908.7 | 114.9 | 9229.1 (83) | 3679.6 | 26437.9 | 11223.6 | 6954.2 | 9970.8 (84) | 1252.8 |
| d | 8504.7 | 12866.9 | 114.2 | 9188.0 (83) | 3679.0 | 26445.5 | 11237.1 | 6976.7 | 9993.1 (84) | 1244.0 |
| e | 8516.7 | 12811.9 | 114.3 | 9189.8 (83) | 3622.2 | 26477.0 | 12043.5 | 6965.5 | 9777.4 (82) | 2266.1 |
| f | 7522.2 | 10856.6 | 113.8 | 9164.2 (83) | 1692.4 | 26436.8 | 10624.5 | 6960.1 | 9380.3 (80) | 1244.2 |
| base-2 | 8503.1 | 13416.3 | 119.3 | 9737.8 (83) | 3678.5 | 26504.9 | 11569.8 | 6977.3 | 10294.4 (83) | 1275.4 |
| a-2 | 8556.1 | 12993.0 | 115.3 | 9362.0 (83) | 3631.0 | 26494.5 | 11566.3 | 6981.3 | 10295.5 (85) | 1270.8 |
| b-2 | 8548.4 | 12996.1 | 114.7 | 9357.4 (83) | 3638.7 | 26467.4 | 11616.6 | 6943.5 | 10353.3 (86) | 1263.4 |
| c-2 | 8524.2 | 12891.2 | 114.7 | 9227.4 (83) | 3663.8 | 26386.4 | 10834.7 | 6888.2 | 9584.1 (81) | 1250.6 |
| d-2 | 8499.6 | 12869.6 | 114.3 | 9207.2 (83) | 3662.4 | 26484.9 | 12072.5 | 6946.3 | 9835.5 (83) | 2237.0 |
| e-2 | 8518.5 | 12830.8 | 114.3 | 9196.4 (83) | 3634.4 | 26499.0 | 12717.2 | 6947.5 | 10450.7 (88) | 2266.5 |
| f-2 | 7545.3 | 10824.7 | 113.6 | 9156.5 (83) | 1668.3 | 26304.4 | 12863.7 | 6833.4 | 10602.8 (90) | 2260.9 |

