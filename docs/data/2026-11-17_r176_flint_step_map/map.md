## 1. Every request (the server's own times)

| Load | Request | Depth | Prompt tokens read | Prompt ms | Tokens out | Generation ms | ms a token | Drafted | Accepted | Tokens a step | ms a step | Reply ids md5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| h1 | warmup 0 | 0 | 64 | 2761.527 | 32 | 1514.774 | 47.34 | 0 | 0 |  | 47.34 | `80a49ba0` |
| h1 | decode 1 | 0 | 64 | 751.117 | 256 | 9421.819 | 36.80 | 0 | 0 | 1.000 | 36.80 | `576dc466` |
| h1 | decode 2 | 0 | 4 | 144.753 | 256 | 8955.479 | 34.98 | 0 | 0 | 1.000 | 34.98 | `576dc466` |
| h1 | prime 0 | 8192 | 8128 | 33250.48 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| h1 | decode 1 | 8192 | 64 | 850.783 | 256 | 12630.803 | 49.34 | 0 | 0 | 1.000 | 49.34 | `e20d7b7e` |
| h1 | decode 2 | 8192 | 1 | 52.429 | 256 | 12274.53 | 47.95 | 0 | 0 | 1.000 | 47.95 | `e20d7b7e` |
| g1 | warmup 0 | 0 | 64 | 2807.082 | 32 | 2641.11 | 82.53 | 0 | 0 |  | 82.53 | `80a49ba0` |
| g1 | decode 1 | 0 | 64 | 874.948 | 256 | 18808.015 | 73.47 | 0 | 0 | 1.000 | 73.47 | `576dc466` |
| g1 | decode 2 | 0 | 4 | 162.744 | 256 | 18357.854 | 71.71 | 0 | 0 | 1.000 | 71.71 | `576dc466` |
| g1 | prime 0 | 8192 | 8128 | 66304.737 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| g1 | decode 1 | 8192 | 64 | 932.289 | 256 | 28774.004 | 112.40 | 0 | 0 | 1.000 | 112.40 | `e20d7b7e` |
| g1 | decode 2 | 8192 | 1 | 58.779 | 256 | 29453.318 | 115.05 | 0 | 0 | 1.000 | 115.05 | `e20d7b7e` |
| s8 | warmup 0 | 0 | 64 | 2614.585 | 32 | 1470.317 | 45.95 | 23 | 23 |  | 45.95 | `35a04f3c` |
| s8 | decode 1 | 0 | 64 | 840.042 | 256 | 8923.932 | 34.86 | 207 | 207 | 5.224 | 182.10 | `ec89d2b3` |
| s8 | decode 2 | 0 | 4 | 164.13 | 256 | 8857.284 | 34.60 | 207 | 207 | 5.224 | 180.74 | `ec89d2b3` |
| s8 | prime 0 | 8192 | 8128 | 37307.716 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| s8 | decode 1 | 8192 | 64 | 851.083 | 256 | 9262.816 | 36.18 | 221 | 221 | 7.314 | 264.64 | `de9d3fee` |
| s8 | decode 2 | 8192 | 1 | 56.11 | 256 | 9278.362 | 36.24 | 221 | 221 | 7.314 | 265.09 | `de9d3fee` |
| s4 | warmup 0 | 0 | 64 | 2761.923 | 32 | 1321.18 | 41.29 | 14 | 14 |  | 41.29 | `78ee33be` |
| s4 | decode 1 | 0 | 64 | 850.487 | 256 | 9637.848 | 37.65 | 132 | 132 | 2.065 | 77.74 | `cdb33df4` |
| s4 | decode 2 | 0 | 4 | 143.086 | 256 | 9139.038 | 35.70 | 132 | 132 | 2.065 | 73.72 | `cdb33df4` |
| s4 | prime 0 | 8192 | 8128 | 41708.65 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| s4 | decode 1 | 8192 | 64 | 878.562 | 256 | 9556.811 | 37.33 | 191 | 191 | 3.938 | 147.01 | `f7353ffa` |
| s4 | decode 2 | 8192 | 1 | 49.478 | 256 | 9545.14 | 37.29 | 191 | 191 | 3.938 | 146.83 | `f7353ffa` |
| s2 | warmup 0 | 0 | 64 | 3538.55 | 32 | 1350.751 | 42.21 | 11 | 11 |  | 42.21 | `ddef4b21` |
| s2 | decode 1 | 0 | 64 | 1047.047 | 256 | 10002.331 | 39.07 | 105 | 105 | 1.695 | 66.23 | `77b3f181` |
| s2 | decode 2 | 0 | 4 | 160.572 | 256 | 9478.538 | 37.03 | 105 | 105 | 1.695 | 62.76 | `77b3f181` |
| s2 | prime 0 | 8192 | 8128 | 39126.784 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| s2 | decode 1 | 8192 | 64 | 880.02 | 256 | 9956.725 | 38.89 | 123 | 123 | 1.925 | 74.87 | `e3490b33` |
| s2 | decode 2 | 8192 | 1 | 45.948 | 256 | 9942.772 | 38.84 | 123 | 123 | 1.925 | 74.76 | `e3490b33` |
| h2 | warmup 0 | 0 | 64 | 3016.216 | 32 | 1868.527 | 58.39 | 0 | 0 |  | 58.39 | `80a49ba0` |
| h2 | decode 1 | 0 | 64 | 751.618 | 256 | 9871.838 | 38.56 | 0 | 0 | 1.000 | 38.56 | `576dc466` |
| h2 | decode 2 | 0 | 4 | 154.26 | 256 | 8934.957 | 34.90 | 0 | 0 | 1.000 | 34.90 | `576dc466` |
| h2 | prime 0 | 8192 | 8128 | 34705.038 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| h2 | decode 1 | 8192 | 64 | 854.648 | 256 | 12324.165 | 48.14 | 0 | 0 | 1.000 | 48.14 | `e20d7b7e` |
| h2 | decode 2 | 8192 | 1 | 51.848 | 256 | 12038.989 | 47.03 | 0 | 0 | 1.000 | 47.03 | `e20d7b7e` |
| g2 | warmup 0 | 0 | 64 | 3073.672 | 32 | 2826.144 | 88.32 | 0 | 0 |  | 88.32 | `80a49ba0` |
| g2 | decode 1 | 0 | 64 | 810.034 | 256 | 19546.299 | 76.35 | 0 | 0 | 1.000 | 76.35 | `576dc466` |
| g2 | decode 2 | 0 | 4 | 161.234 | 256 | 18494.156 | 72.24 | 0 | 0 | 1.000 | 72.24 | `576dc466` |
| g2 | prime 0 | 8192 | 8128 | 69009.693 | 1 | 0.001 | 0.00 | 0 | 0 |  | 0.00 | `1207e034` |
| g2 | decode 1 | 8192 | 64 | 956.639 | 256 | 28757.654 | 112.33 | 0 | 0 | 1.000 | 112.33 | `e20d7b7e` |
| g2 | decode 2 | 8192 | 1 | 76.271 | 256 | 29475.866 | 115.14 | 0 | 0 | 1.000 | 115.14 | `e20d7b7e` |

## 2. The one-token step of the preset (pipelined guesses), host profile

Per request: the server's ms a token under the host profile, which waits at each card's next evaluation for its previous one and so keeps the host from queuing guesses ahead (not the preset's speed); the server log's guesses; each card's evaluations a generated token (a dropped guess is an evaluation that gave no token); the pass's GPU time (each card's median over its typical evaluations, summed over the nine cards) and the cards' host time a pass (the same, between entering and leaving each card's graph evaluation). The step trace has no line per pipelined token.

| Load | Depth | Pass | ms a token (host profile) | Guesses, right, wrong (server log) | Card 0 evaluations a token | Card 8 evaluations a token | Pass GPU ms | Cards' host ms a pass | Steps (trace) | Step interval ms: p10, median, p90 |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| h1 | 0 | 1 | 36.80 | 352, 80, 68 | 2.07 | 2.07 | 25.90 | 0.62 | 2 | 115.86, 115.86, 115.86 |
| h1 | 0 | 2 | 34.98 | 352, 80, 68 | 2.06 | 2.06 | 25.95 | 0.61 | 1 | , ,  |
| h1 | 8192 | 1 | 49.34 | 567, 93, 120 | 2.86 | 2.86 | 30.59 | 0.62 | 2 | 78.95, 78.95, 78.95 |
| h1 | 8192 | 2 | 47.95 | 567, 93, 120 | 2.85 | 2.85 | 30.57 | 0.62 | 1 | , ,  |
| h2 | 0 | 1 | 38.56 | 352, 80, 68 | 2.07 | 2.07 | 25.87 | 0.61 | 2 | 101.88, 101.88, 101.88 |
| h2 | 0 | 2 | 34.90 | 352, 80, 68 | 2.06 | 2.06 | 25.94 | 0.61 | 1 | , ,  |
| h2 | 8192 | 1 | 48.14 | 567, 93, 120 | 2.86 | 2.86 | 29.95 | 0.62 | 2 | 86.52, 86.52, 86.52 |
| h2 | 8192 | 2 | 47.03 | 567, 93, 120 | 2.85 | 2.85 | 29.94 | 0.62 | 1 | , ,  |

## 3. A one-token pass by part (group profile, the repeated request), GPU ms summed over the nine cards

| Part | g1, 0 | g2, 0 | g1, 8192 | g2, 8192 |
|---|---:|---:|---:|---:|
| routed experts | 7.82 | 7.92 | 7.73 | 7.71 |
| shared expert | 2.62 | 2.67 | 2.73 | 2.73 |
| router | 0.91 | 0.94 | 0.80 | 0.79 |
| attention products (q, k, v, out) | 8.04 | 8.13 | 8.02 | 7.98 |
| dense feed-forward | 0.55 | 0.55 | 0.54 | 0.54 |
| attention (FLASH_ATTN_EXT) | 1.97 | 1.99 | 7.31 | 7.30 |
| output layer | 2.03 | 2.01 | 2.03 | 2.02 |
| other weights | 0.00 | 0.00 | 0.00 | 0.00 |
| other ops without weights | 1.40 | 1.41 | 1.13 | 1.14 |
| **the pass** | 25.35 | 25.61 | 30.30 | 30.22 |

### 3a. The same, the first request at the depth

| Part | g1, 0 | g2, 0 | g1, 8192 | g2, 8192 |
|---|---:|---:|---:|---:|
| routed experts | 7.67 | 7.85 | 7.43 | 7.37 |
| shared expert | 2.57 | 2.64 | 2.62 | 2.61 |
| router | 0.89 | 0.92 | 0.75 | 0.74 |
| attention products (q, k, v, out) | 7.93 | 8.09 | 7.72 | 7.64 |
| dense feed-forward | 0.55 | 0.55 | 0.52 | 0.52 |
| attention (FLASH_ATTN_EXT) | 1.93 | 1.97 | 6.94 | 6.89 |
| output layer | 2.03 | 1.93 | 2.01 | 2.02 |
| other weights | 0.00 | 0.00 | 0.00 | 0.00 |
| other ops without weights | 1.34 | 1.36 | 1.10 | 1.11 |
| **the pass** | 24.92 | 25.32 | 29.10 | 28.90 |

### 3b. What grows from depth 0 to 8192, by launch-group label (the repeated request, ms a pass)

| Label | Part | g1: 0, 8192, growth | g2: 0, 8192, growth |
|---|---|---|---|
| `FLASH_ATTN_EXT` | attention (FLASH_ATTN_EXT) | 1.97, 7.31, +5.34 | 1.99, 7.30, +5.31 |
| `RMS_NORM+MUL norm` | other ops without weights | 0.53, 0.33, -0.20 | 0.54, 0.33, -0.20 |
| `MUL_MAT ffn_gate_inp:f16:2048x256` | router | 0.91, 0.80, -0.11 | 0.94, 0.79, -0.15 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q8_0:2048x512 ffn_up_exps:q8_0:2048x512` | routed experts | 4.17, 4.13, -0.04 | 4.24, 4.12, -0.12 |
| `MUL_MAT+ADD attn_output:q8_0:4096x2048` | attention products (q, k, v, out) | 2.44, 2.38, -0.06 | 2.48, 2.39, -0.09 |
| `ADD+RMS_NORM+MUL l_out` | other ops without weights | 0.06, 0.12, +0.06 | 0.06, 0.12, +0.06 |
| `MUL_MAT+ADD ffn_down_shexp:q8_0:1024x2048` | shared expert | 0.83, 0.90, +0.07 | 0.85, 0.90, +0.05 |
| `MUL_MAT_ID ffn_down_exps:q8_0:512x2048` | routed experts | 2.53, 2.49, -0.03 | 2.56, 2.49, -0.08 |
| `SIGMOID+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | other ops without weights | 0.32, 0.28, -0.04 | 0.33, 0.28, -0.06 |
| `SET_ROWS cache_k_l33_(view)` | other ops without weights | 0.04, 0.00, -0.04 | 0.04, 0.00, -0.04 |
| `MUL_MAT attn_k:q8_0:2048x512` | attention products (q, k, v, out) | 0.10, 0.14, +0.04 | 0.10, 0.14, +0.04 |
| `MUL_MAT+RMS_NORM+MUL+ROPE+MUL_MAT+MUL_MAT+RMS_NORM+MUL+ROPE+SET_ROWS+SET_ROWS attn_q:q8_0:2048x4096 attn_v:q8_0:2048x512 attn_k:q8_0:2048x512` | attention products (q, k, v, out) | 2.39, 2.40, +0.01 | 2.45, 2.39, -0.05 |

### 3c. Each card's GPU ms a one-token pass (group profile, the repeated request)

| Card | g1, 0 | g2, 0 | g1, 8192 | g2, 8192 |
|---:|---:|---:|---:|---:|
| 0 | 3.02 | 3.01 | 3.97 | 3.98 |
| 1 | 2.81 | 2.91 | 3.26 | 3.27 |
| 2 | 2.45 | 2.45 | 2.79 | 2.87 |
| 3 | 2.51 | 2.51 | 3.12 | 3.12 |
| 4 | 2.61 | 2.82 | 3.16 | 3.06 |
| 5 | 2.45 | 2.45 | 2.74 | 2.74 |
| 6 | 2.34 | 2.33 | 2.96 | 2.96 |
| 7 | 2.74 | 2.74 | 3.15 | 3.08 |
| 8 | 4.42 | 4.39 | 5.14 | 5.14 |

### 3d. The biggest launch-group labels of a one-token pass (the repeated request, depth 0)

| Label | Part | g1 | g2 |
|---|---|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q8_0:2048x512 ffn_up_exps:q8_0:2048x512` | routed experts | 4.17 | 4.24 |
| `MUL_MAT_ID ffn_down_exps:q8_0:512x2048` | routed experts | 2.53 | 2.56 |
| `MUL_MAT+ADD attn_output:q8_0:4096x2048` | attention products (q, k, v, out) | 2.44 | 2.48 |
| `MUL_MAT+RMS_NORM+MUL+ROPE+MUL_MAT+MUL_MAT+RMS_NORM+MUL+ROPE+SET_ROWS+SET_ROWS attn_q:q8_0:2048x4096 attn_v:q8_0:2048x512 attn_k:q8_0:2048x512` | attention products (q, k, v, out) | 2.39 | 2.45 |
| `MUL_MAT output:q5_0:2048x150144` | output layer | 2.03 | 2.01 |
| `FLASH_ATTN_EXT` | attention (FLASH_ATTN_EXT) | 1.97 | 1.99 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q8_0:2048x1024 ffn_up_shexp:q8_0:2048x1024` | shared expert | 1.05 | 1.07 |
| `MUL_MAT+RMS_NORM+MUL+ROPE+MUL_MAT+MUL_MAT+RMS_NORM+MUL+ROPE+SET_ROWS+SET_ROWS attn_q:q5_0:2048x4096 attn_v:q5_0:2048x512 attn_k:q5_0:2048x512` | attention products (q, k, v, out) | 1.01 | 1.01 |
| `MUL_MAT+ADD attn_output:q5_0:4096x2048` | attention products (q, k, v, out) | 0.94 | 0.93 |
| `MUL_MAT ffn_gate_inp:f16:2048x256` | router | 0.91 | 0.94 |
| `MUL_MAT+ADD ffn_down_shexp:q8_0:1024x2048` | shared expert | 0.83 | 0.85 |
| `MUL_MAT attn_q:q8_0:2048x4096` | attention products (q, k, v, out) | 0.63 | 0.63 |
| `RMS_NORM+MUL+MUL_MAT+SIGMOID+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV+MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT_ID+MUL+ADD+ADD+ADD+ADD+ADD+MUL_MAT+MUL_MAT+SWIGLU+MUL_MAT+ADD ffn_gate_inp:f16:2048x256 ffn_gate_exps:q5_0:2048x512 ffn_up_exps:q5_0:2048x512 ffn_down_exps:q8_0:512x2048 ffn_gate_shexp:q8_0:2048x1024 ffn_up_shexp:q8_0:2048x1024 ffn_down_shexp:q8_0:1024x2048` | routed experts | 0.61 | 0.61 |
| `RMS_NORM+MUL norm` | other ops without weights | 0.53 | 0.54 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q5_0:2048x512 ffn_up_exps:q5_0:2048x512` | routed experts | 0.51 | 0.50 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q5_0:2048x1024 ffn_up_shexp:q5_0:2048x1024` | shared expert | 0.42 | 0.42 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate:q8_0:2048x6144 ffn_up:q8_0:2048x6144` | dense feed-forward | 0.37 | 0.37 |
| `MUL_MAT+ADD ffn_down_shexp:q5_0:1024x2048` | shared expert | 0.33 | 0.32 |
| `SIGMOID+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | other ops without weights | 0.32 | 0.33 |
| `MUL_MAT attn_q:q5_0:2048x4096` | attention products (q, k, v, out) | 0.31 | 0.30 |
| `MUL_MAT+ADD ffn_down:q8_0:6144x2048` | dense feed-forward | 0.18 | 0.18 |
| `MUL+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | other ops without weights | 0.15 | 0.14 |
| `MUL_MAT attn_k:q8_0:2048x512` | attention products (q, k, v, out) | 0.10 | 0.10 |
| `TOP_K top_k` | other ops without weights | 0.10 | 0.09 |
| `MUL_MAT attn_v:q8_0:2048x512` | attention products (q, k, v, out) | 0.09 | 0.09 |

## 5. Step-trace marks per request (counts), for reading the step

- **h1 decode:0:1**: 2 steps over 115.9 ms; n: tok 64, yld 2; marks: v.sync 8, v.sync.end 8, s.upd 2, s.post 2, s.shift 2, s.sel 2, s.ckpt 2, s.pre.end 2, s.slots.end 2, s.render 2, s.dec 2, s.dec.y 2, s.dec.in 2, v.dec 2, v.dec.bal 2, v.dec.mem 2, v.dec.out 2, v.ub.apply 2, v.ub.build 2, v.ub.in 2, v.ub.comp 2, v.ub.cp 2, v.dec.end 2, s.dec.ret 2, s.dec.out 2, s.dec.end 2, s.mpd 2, s.pd 2, s.upd.end 2
- **h1 decode:0:2**: 1 steps over 0.0 ms; n: tok 4, yld 1; marks: v.sync 9, v.sync.end 9, s.upd 1, s.post 1, s.shift 1, s.sel 1, s.ckpt 1, s.pre.end 1, s.slots.end 1, s.render 1, s.dec 1, s.dec.y 1, s.dec.in 1, v.dec 1, v.dec.bal 1, v.dec.mem 1, v.dec.out 1, v.ub.apply 1, v.ub.build 1, v.ub.in 1, v.ub.comp 1, v.ub.cp 1, v.dec.end 1, s.dec.ret 1, s.dec.out 1, s.dec.end 1, s.mpd 1, s.pd 1, s.upd.end 1
- **h1 decode:8192:1**: 2 steps over 78.9 ms; n: tok 64, yld 2; marks: v.sync 8, v.sync.end 8, s.upd 2, s.post 2, s.shift 2, s.sel 2, s.ckpt 2, s.pre.end 2, s.slots.end 2, s.render 2, s.dec 2, s.dec.y 2, s.dec.in 2, v.dec 2, v.dec.bal 2, v.dec.mem 2, v.dec.out 2, v.ub.apply 2, v.ub.build 2, v.ub.in 2, v.ub.comp 2, v.ub.cp 2, v.dec.end 2, s.dec.ret 2, s.dec.out 2, s.dec.end 2, s.mpd 2, s.pd 2, s.upd.end 2
- **h1 decode:8192:2**: 1 steps over 0.0 ms; n: tok 1, yld 1; marks: v.sync 8, v.sync.end 8, s.upd 1, s.post 1, s.shift 1, s.sel 1, s.ckpt 1, s.pre.end 1, s.slots.end 1, s.render 1, s.dec 1, s.dec.y 1, s.dec.in 1, v.dec 1, v.dec.bal 1, v.dec.mem 1, v.dec.out 1, v.ub.apply 1, v.ub.build 1, v.ub.in 1, v.ub.comp 1, v.ub.cp 1, v.dec.end 1, s.dec.ret 1, s.dec.out 1, s.dec.end 1, s.mpd 1, s.pd 1, s.upd.end 1
- **h2 decode:0:1**: 2 steps over 101.9 ms; n: tok 64, yld 2; marks: v.sync 8, v.sync.end 8, s.upd 2, s.post 2, s.shift 2, s.sel 2, s.ckpt 2, s.pre.end 2, s.slots.end 2, s.render 2, s.dec 2, s.dec.y 2, s.dec.in 2, v.dec 2, v.dec.bal 2, v.dec.mem 2, v.dec.out 2, v.ub.apply 2, v.ub.build 2, v.ub.in 2, v.ub.comp 2, v.ub.cp 2, v.dec.end 2, s.dec.ret 2, s.dec.out 2, s.dec.end 2, s.mpd 2, s.pd 2, s.upd.end 2
- **h2 decode:0:2**: 1 steps over 0.0 ms; n: tok 4, yld 1; marks: v.sync 9, v.sync.end 9, s.upd 1, s.post 1, s.shift 1, s.sel 1, s.ckpt 1, s.pre.end 1, s.slots.end 1, s.render 1, s.dec 1, s.dec.y 1, s.dec.in 1, v.dec 1, v.dec.bal 1, v.dec.mem 1, v.dec.out 1, v.ub.apply 1, v.ub.build 1, v.ub.in 1, v.ub.comp 1, v.ub.cp 1, v.dec.end 1, s.dec.ret 1, s.dec.out 1, s.dec.end 1, s.mpd 1, s.pd 1, s.upd.end 1
- **h2 decode:8192:1**: 2 steps over 86.5 ms; n: tok 64, yld 2; marks: v.sync 8, v.sync.end 8, s.upd 2, s.post 2, s.shift 2, s.sel 2, s.ckpt 2, s.pre.end 2, s.slots.end 2, s.render 2, s.dec 2, s.dec.y 2, s.dec.in 2, v.dec 2, v.dec.bal 2, v.dec.mem 2, v.dec.out 2, v.ub.apply 2, v.ub.build 2, v.ub.in 2, v.ub.comp 2, v.ub.cp 2, v.dec.end 2, s.dec.ret 2, s.dec.out 2, s.dec.end 2, s.mpd 2, s.pd 2, s.upd.end 2
- **h2 decode:8192:2**: 1 steps over 0.0 ms; n: tok 1, yld 1; marks: v.sync 8, v.sync.end 8, s.upd 1, s.post 1, s.shift 1, s.sel 1, s.ckpt 1, s.pre.end 1, s.slots.end 1, s.render 1, s.dec 1, s.dec.y 1, s.dec.in 1, v.dec 1, v.dec.bal 1, v.dec.mem 1, v.dec.out 1, v.ub.apply 1, v.ub.build 1, v.ub.in 1, v.ub.comp 1, v.ub.cp 1, v.dec.end 1, s.dec.ret 1, s.dec.out 1, s.dec.end 1, s.mpd 1, s.pd 1, s.upd.end 1
- **s2 decode:0:1**: 152 steps over 10974.1 ms; n: acc 105, drf 105, tok 319, yld 154; marks: v.sync 1688, v.sync.end 1688, s.tok 209, s.upd 152, s.post 152, s.shift 152, s.sel 152, s.ckpt 152, s.pre.end 152, s.slots.end 152, s.render 152, s.dec 152, s.dec.y 152, s.dec.in 152, v.dec 152, v.dec.bal 152, v.dec.mem 152, v.dec.out 152, v.ub.apply 152, v.ub.in 152, v.ub.comp 152, v.ub.cp 152, v.dec.end 152, s.dec.ret 152, s.dec.out 152, s.dec.end 152, s.mpd 152, s.cu.in 152, s.cu.out 152, s.cu.end 152, s.pd 152, s.upd.end 152, s.drf 150, s.drf.in 150, s.drf.out 150, s.drf.end 150, s.ckgen 150, s.bat 150, v.ub.reuse 115, s.acc 105
- **s2 decode:0:2**: 151 steps over 9563.8 ms; n: acc 105, drf 105, tok 259, yld 152; marks: v.sync 1689, v.sync.end 1689, s.tok 209, s.upd 151, s.post 151, s.shift 151, s.sel 151, s.ckpt 151, s.pre.end 151, s.slots.end 151, s.render 151, s.dec 151, s.dec.y 151, s.dec.in 151, v.dec 151, v.dec.bal 151, v.dec.mem 151, v.dec.out 151, v.ub.apply 151, v.ub.in 151, v.ub.comp 151, v.ub.cp 151, v.dec.end 151, s.dec.ret 151, s.dec.out 151, s.dec.end 151, s.mpd 151, s.cu.in 151, s.cu.out 151, s.cu.end 151, s.pd 151, s.upd.end 151, s.drf 150, s.drf.in 150, s.drf.out 150, s.drf.end 150, s.ckgen 150, s.bat 150, v.ub.reuse 115, s.acc 105
- **s2 decode:8192:1**: 134 steps over 10795.9 ms; n: acc 123, drf 123, tok 319, yld 136; marks: v.sync 1670, v.sync.end 1670, s.tok 246, s.upd 134, s.post 134, s.shift 134, s.sel 134, s.ckpt 134, s.pre.end 134, s.slots.end 134, s.render 134, s.dec 134, s.dec.y 134, s.dec.in 134, v.dec 134, v.dec.bal 134, v.dec.mem 134, v.dec.out 134, v.ub.apply 134, v.ub.in 134, v.ub.comp 134, v.ub.cp 134, v.dec.end 134, s.dec.ret 134, s.dec.out 134, s.dec.end 134, s.mpd 134, s.cu.in 134, s.cu.out 134, s.cu.end 134, s.pd 134, s.upd.end 134, s.ckgen 132, s.bat 132, s.drf 131, s.drf.in 131, s.drf.out 131, s.drf.end 131, s.acc 123, s.acc.cl 123
- **s2 decode:8192:2**: 133 steps over 9948.3 ms; n: acc 123, drf 123, tok 256, yld 134; marks: v.sync 1670, v.sync.end 1670, s.tok 246, s.upd 133, s.post 133, s.shift 133, s.sel 133, s.ckpt 133, s.pre.end 133, s.slots.end 133, s.render 133, s.dec 133, s.dec.y 133, s.dec.in 133, v.dec 133, v.dec.bal 133, v.dec.mem 133, v.dec.out 133, v.ub.apply 133, v.ub.in 133, v.ub.comp 133, v.ub.cp 133, v.dec.end 133, s.dec.ret 133, s.dec.out 133, s.dec.end 133, s.mpd 133, s.cu.in 133, s.cu.out 133, s.cu.end 133, s.pd 133, s.upd.end 133, s.ckgen 132, s.bat 132, s.drf 131, s.drf.in 131, s.drf.out 131, s.drf.end 131, s.acc 123, s.acc.cl 123
- **s4 decode:0:1**: 125 steps over 10451.9 ms; n: acc 132, drf 132, tok 319, yld 127; marks: v.sync 1661, v.sync.end 1661, s.tok 218, s.upd 125, s.post 125, s.shift 125, s.sel 125, s.ckpt 125, s.pre.end 125, s.slots.end 125, s.render 125, s.dec 125, s.dec.y 125, s.dec.in 125, v.dec 125, v.dec.bal 125, v.dec.mem 125, v.dec.out 125, v.ub.apply 125, v.ub.in 125, v.ub.comp 125, v.ub.cp 125, v.dec.end 125, s.dec.ret 125, s.dec.out 125, s.dec.end 125, s.mpd 125, s.cu.in 125, s.cu.out 125, s.cu.end 125, s.pd 125, s.upd.end 125, s.ckgen 123, s.bat 123, s.drf 122, s.drf.in 122, s.drf.out 122, s.drf.end 122, v.ub.reuse 106, s.acc 86
- **s4 decode:0:2**: 124 steps over 9245.3 ms; n: acc 132, drf 132, tok 259, yld 125; marks: v.sync 1662, v.sync.end 1662, s.tok 218, s.upd 124, s.post 124, s.shift 124, s.sel 124, s.ckpt 124, s.pre.end 124, s.slots.end 124, s.render 124, s.dec 124, s.dec.y 124, s.dec.in 124, v.dec 124, v.dec.bal 124, v.dec.mem 124, v.dec.out 124, v.ub.apply 124, v.ub.in 124, v.ub.comp 124, v.ub.cp 124, v.dec.end 124, s.dec.ret 124, s.dec.out 124, s.dec.end 124, s.mpd 124, s.cu.in 124, s.cu.out 124, s.cu.end 124, s.pd 124, s.upd.end 124, s.ckgen 123, s.bat 123, s.drf 122, s.drf.in 122, s.drf.out 122, s.drf.end 122, v.ub.reuse 106, s.acc 86
- **s4 decode:8192:1**: 66 steps over 10310.8 ms; n: acc 191, drf 191, tok 319, yld 68; marks: v.sync 1602, v.sync.end 1602, s.tok 254, s.upd 66, s.post 66, s.shift 66, s.sel 66, s.ckpt 66, s.pre.end 66, s.slots.end 66, s.render 66, s.dec 66, s.dec.y 66, s.dec.in 66, v.dec 66, v.dec.bal 66, v.dec.mem 66, v.dec.out 66, v.ub.apply 66, v.ub.in 66, v.ub.comp 66, v.ub.cp 66, v.dec.end 66, s.dec.ret 66, s.dec.out 66, s.dec.end 66, s.mpd 66, s.cu.in 66, s.cu.out 66, s.cu.end 66, s.pd 66, s.upd.end 66, s.drf 64, s.drf.in 64, s.drf.out 64, s.drf.end 64, s.ckgen 64, s.bat 64, s.acc 64, s.acc.cl 64
- **s4 decode:8192:2**: 65 steps over 9470.0 ms; n: acc 191, drf 191, tok 256, yld 66; marks: v.sync 1602, v.sync.end 1602, s.tok 254, s.upd 65, s.post 65, s.shift 65, s.sel 65, s.ckpt 65, s.pre.end 65, s.slots.end 65, s.render 65, s.dec 65, s.dec.y 65, s.dec.in 65, v.dec 65, v.dec.bal 65, v.dec.mem 65, v.dec.out 65, v.ub.apply 65, v.ub.in 65, v.ub.comp 65, v.ub.cp 65, v.dec.end 65, s.dec.ret 65, s.dec.out 65, s.dec.end 65, s.mpd 65, s.cu.in 65, s.cu.out 65, s.cu.end 65, s.pd 65, s.upd.end 65, s.drf 64, s.drf.in 64, s.drf.out 64, s.drf.end 64, s.ckgen 64, s.bat 64, s.acc 64, s.acc.cl 64
- **s8 decode:0:1**: 50 steps over 9481.0 ms; n: acc 207, drf 207, tok 319, yld 52; marks: v.sync 1586, v.sync.end 1586, s.tok 236, s.upd 50, s.post 50, s.shift 50, s.sel 50, s.ckpt 50, s.pre.end 50, s.slots.end 50, s.render 50, s.dec 50, s.dec.y 50, s.dec.in 50, v.dec 50, v.dec.bal 50, v.dec.mem 50, v.dec.out 50, v.ub.apply 50, v.ub.in 50, v.ub.comp 50, v.ub.cp 50, v.dec.end 50, s.dec.ret 50, s.dec.out 50, s.dec.end 50, s.mpd 50, s.cu.in 50, s.cu.out 50, s.cu.end 50, s.pd 50, s.upd.end 50, s.drf 48, s.drf.in 48, s.drf.out 48, s.drf.end 48, s.ckgen 48, s.bat 48, v.ub.reuse 43, s.acc 30
- **s8 decode:0:2**: 49 steps over 8738.4 ms; n: acc 207, drf 207, tok 259, yld 50; marks: v.sync 1587, v.sync.end 1587, s.tok 236, s.upd 49, s.post 49, s.shift 49, s.sel 49, s.ckpt 49, s.pre.end 49, s.slots.end 49, s.render 49, s.dec 49, s.dec.y 49, s.dec.in 49, v.dec 49, v.dec.bal 49, v.dec.mem 49, v.dec.out 49, v.ub.apply 49, v.ub.in 49, v.ub.comp 49, v.ub.cp 49, v.dec.end 49, s.dec.ret 49, s.dec.out 49, s.dec.end 49, s.mpd 49, s.cu.in 49, s.cu.out 49, s.cu.end 49, s.pd 49, s.upd.end 49, s.drf 48, s.drf.in 48, s.drf.out 48, s.drf.end 48, s.ckgen 48, s.bat 48, v.ub.reuse 43, s.acc 30
- **s8 decode:8192:1**: 36 steps over 9901.0 ms; n: acc 221, drf 221, tok 319, yld 38; marks: v.sync 1572, v.sync.end 1572, s.tok 254, s.upd 36, s.post 36, s.shift 36, s.sel 36, s.ckpt 36, s.pre.end 36, s.slots.end 36, s.render 36, s.dec 36, s.dec.y 36, s.dec.in 36, v.dec 36, v.dec.bal 36, v.dec.mem 36, v.dec.out 36, v.ub.apply 36, v.ub.in 36, v.ub.comp 36, v.ub.cp 36, v.dec.end 36, s.dec.ret 36, s.dec.out 36, s.dec.end 36, s.mpd 36, s.cu.in 36, s.cu.out 36, s.cu.end 36, s.pd 36, s.upd.end 36, s.drf 34, s.drf.in 34, s.drf.out 34, s.drf.end 34, s.ckgen 34, s.bat 34, s.acc 34, s.acc.cl 34
- **s8 decode:8192:2**: 35 steps over 9121.2 ms; n: acc 221, drf 221, tok 256, yld 36; marks: v.sync 1572, v.sync.end 1572, s.tok 254, s.upd 35, s.post 35, s.shift 35, s.sel 35, s.ckpt 35, s.pre.end 35, s.slots.end 35, s.render 35, s.dec 35, s.dec.y 35, s.dec.in 35, v.dec 35, v.dec.bal 35, v.dec.mem 35, v.dec.out 35, v.ub.apply 35, v.ub.in 35, v.ub.comp 35, v.ub.cp 35, v.dec.end 35, s.dec.ret 35, s.dec.out 35, s.dec.end 35, s.mpd 35, s.cu.in 35, s.cu.out 35, s.cu.end 35, s.pd 35, s.upd.end 35, s.drf 34, s.drf.in 34, s.drf.out 34, s.drf.end 34, s.ckgen 34, s.bat 34, s.acc 34, s.acc.cl 34

## Notes

- hp-h1.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 58 529 528 65 731 729
- server-h1.log: 6 pipelined-guess lines for 6 requests that generate
- st-h1.jsonl: 6 pieces for 6 requests; steps per piece: 2 2 1 17 2 1
- gp-g1.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 58 529 528 65 731 730
- server-g1.log: 6 pipelined-guess lines for 6 requests that generate
- hp-s8.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 10 50 49 65 36 34
- server-s8.log: 6 pipelined-guess lines for 6 requests that generate
- st-s8.jsonl: 6 pieces for 6 requests; steps per piece: 10 50 49 17 36 35
- hp-s4.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 19 125 124 65 66 64
- server-s4.log: 6 pipelined-guess lines for 6 requests that generate
- st-s4.jsonl: 6 pieces for 6 requests; steps per piece: 19 125 124 17 66 65
- hp-s2.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 22 152 151 65 134 132
- server-s2.log: 6 pipelined-guess lines for 6 requests that generate
- st-s2.jsonl: 6 pieces for 6 requests; steps per piece: 22 152 151 17 134 133
- hp-h2.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 58 529 528 65 731 729
- server-h2.log: 6 pipelined-guess lines for 6 requests that generate
- st-h2.jsonl: 6 pieces for 6 requests; steps per piece: 2 2 1 17 2 1
- gp-g2.csv: 7 pieces for 6 requests; card 0's evaluations per piece: 2 58 529 528 65 731 730
- server-g2.log: 6 pipelined-guess lines for 6 requests that generate
