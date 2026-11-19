# R197: arm base (Ornith's v5 preset) against arm ro (+ the read-once switches), request p1 (the first request at the depth); each load by value, in run order

## 1. The verify pass, GPU ms a pass, nine cards (the group profile's launch groups summed)

| width | depth | base | ro | ro − base, paired | change of the means | spread base, ro | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | 85.92, 85.94 | 82.86, 82.95 | -3.06, -2.99 | -3.52 % | 0.02, 0.09 | **gain** |
| 2 | 8,192 | 97.59, 97.60 | 94.22, 94.35 | -3.36, -3.25 | -3.39 % | 0.01, 0.12 | **gain** |
| 3 | 0 | 111.37, 111.31 | 107.21, 107.22 | -4.16, -4.09 | -3.71 % | 0.06, 0.00 | **gain** |
| 3 | 8,192 | 129.20, 129.23 | 124.54, 124.65 | -4.66, -4.58 | -3.58 % | 0.03, 0.11 | **gain** |
| 4 | 0 | 132.81, 132.57 | 137.60, 137.44 | 4.79, 4.87 | +3.64 % | 0.24, 0.16 | **loss** |
| 4 | 8,192 | 151.94, 152.04 | 154.47, 154.57 | 2.54, 2.53 | +1.67 % | 0.11, 0.10 | **loss** |

## 2. The pass by family of launch groups, ms a pass, nine cards (families that moved 0.05 ms or more)

### 2 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.92, 25.86 | 23.52, 23.49 | -2.40, -2.38 |
| attention layers: q, k, v, output products | 7.86, 7.86 | 6.95, 6.95 | -0.90, -0.91 |
| shared expert: gate, up and GLU | 2.95, 2.94 | 2.87, 2.86 | -0.08, -0.08 |
| routed experts: down | 9.65, 9.66 | 9.72, 9.73 | 0.07, 0.07 |
| output head | 4.96, 4.95 | 5.04, 5.04 | 0.08, 0.09 |
| everything else (norms, adds, copies, the sampling chain) | 7.88, 7.95 | 7.96, 8.05 | 0.08, 0.09 |
| routed experts: gate, up and GLU | 16.79, 16.80 | 16.89, 16.89 | 0.10, 0.10 |

### 2 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.83, 25.81 | 23.50, 23.48 | -2.33, -2.33 |
| attention layers: q, k, v, output products | 7.90, 7.86 | 6.98, 6.99 | -0.91, -0.87 |
| shared expert: gate, up and GLU | 2.92, 2.93 | 2.89, 2.88 | -0.03, -0.05 |
| output head | 4.98, 4.97 | 5.06, 5.06 | 0.09, 0.09 |

### 3 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 32.03, 32.00 | 29.35, 29.37 | -2.68, -2.64 |
| attention layers: q, k, v, output products | 9.75, 9.74 | 8.78, 8.77 | -0.97, -0.96 |
| output head | 6.29, 6.29 | 5.96, 5.95 | -0.34, -0.34 |
| shared expert: gate, up and GLU | 3.59, 3.57 | 3.34, 3.34 | -0.25, -0.23 |
| routed experts: gate, up and GLU | 23.64, 23.64 | 23.73, 23.73 | 0.08, 0.09 |

### 3 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 32.06, 32.10 | 29.46, 29.49 | -2.60, -2.61 |
| attention layers: q, k, v, output products | 9.73, 9.74 | 8.80, 8.81 | -0.92, -0.93 |
| output head | 6.33, 6.32 | 5.98, 5.98 | -0.35, -0.34 |
| shared expert: gate, up and GLU | 3.57, 3.58 | 3.32, 3.32 | -0.26, -0.26 |
| routed experts: gate, up and GLU | 24.17, 24.18 | 24.01, 24.03 | -0.16, -0.15 |
| attention: FLASH_ATTN_EXT | 17.97, 17.99 | 17.82, 17.84 | -0.15, -0.15 |
| routed experts: down | 14.03, 14.04 | 13.92, 13.94 | -0.11, -0.11 |
| everything else (norms, adds, copies, the sampling chain) | 8.60, 8.56 | 8.54, 8.55 | -0.06, -0.01 |

### 4 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| everything else (norms, adds, copies, the sampling chain) | 8.73, 8.67 | 8.72, 8.76 | -0.02, 0.09 |
| shared expert: gate, up and GLU | 4.33, 4.38 | 4.57, 4.58 | 0.24, 0.20 |
| attention layers: q, k, v, output products | 11.22, 11.21 | 11.59, 11.57 | 0.36, 0.36 |
| routed experts: down | 17.44, 17.42 | 17.98, 17.96 | 0.55, 0.54 |
| output head | 7.61, 7.60 | 8.36, 8.37 | 0.75, 0.77 |
| routed experts: gate, up and GLU | 29.81, 29.75 | 30.74, 30.69 | 0.93, 0.94 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 37.05, 37.00 | 39.02, 38.93 | 1.96, 1.94 |

### 4 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| routed experts: gate, up and GLU | 31.05, 31.10 | 30.67, 30.69 | -0.38, -0.40 |
| routed experts: down | 18.15, 18.19 | 17.91, 17.93 | -0.24, -0.26 |
| attention: FLASH_ATTN_EXT | 18.03, 18.05 | 17.87, 17.89 | -0.17, -0.16 |
| everything else (norms, adds, copies, the sampling chain) | 8.79, 8.78 | 8.74, 8.74 | -0.05, -0.04 |
| shared expert: gate, up and GLU | 4.47, 4.44 | 4.75, 4.73 | 0.29, 0.29 |
| attention layers: q, k, v, output products | 11.23, 11.23 | 11.61, 11.62 | 0.38, 0.39 |
| output head | 7.64, 7.62 | 8.40, 8.39 | 0.75, 0.77 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 37.15, 37.21 | 39.13, 39.15 | 1.99, 1.94 |

## 3. The dense Q5_0 and Q8_0 products (launch groups with a plain MUL_MAT of a q5_0 or q8_0 tensor), ms a pass

| width | depth | base | ro | ro − base, paired | change of the means | groups |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 43.24, 43.17 | 39.99, 39.94 | -3.25, -3.23 | -7.51 % | 13 |
| 2 | 8,192 | 43.19, 43.15 | 40.04, 40.02 | -3.15, -3.13 | -7.27 % | 13 |
| 3 | 0 | 53.64, 53.59 | 49.38, 49.38 | -4.26, -4.21 | -7.90 % | 13 |
| 3 | 8,192 | 53.67, 53.73 | 49.51, 49.55 | -4.17, -4.17 | -7.76 % | 13 |
| 4 | 0 | 62.71, 62.64 | 66.02, 65.93 | 3.31, 3.29 | +5.26 % | 13 |
| 4 | 8,192 | 62.86, 62.87 | 66.27, 66.29 | 3.40, 3.42 | +5.43 % | 13 |

### 2 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.26, 9.26 | 6.70, 6.71 | -2.56, -2.56 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.12, 3.12 | 2.33, 2.33 | -0.78, -0.79 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.41, 9.40 | 9.53, 9.53 | 0.12, 0.13 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.32, 0.31 | 0.22, 0.21 | -0.10, -0.10 |
| `RMS_NORM+SCALE node` | 0.45, 0.37 | 0.47, 0.52 | 0.02, 0.15 |
| `MUL_MAT output:q5_0:3072x248320` | 4.96, 4.95 | 5.04, 5.04 | 0.08, 0.09 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 6.38, 6.39 | 0.08, 0.08 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.35, 0.35 | 0.27, 0.27 | -0.08, -0.08 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 4.14, 4.14 | 0.06, 0.06 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.78, 6.78 | 6.82, 6.82 | 0.05, 0.04 |

### 2 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.27 | 6.72, 6.72 | -2.55, -2.55 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.10, 3.10 | 2.32, 2.33 | -0.78, -0.77 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.44, 9.44 | 9.57, 9.57 | 0.13, 0.13 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.36, 0.33 | 0.24, 0.24 | -0.12, -0.09 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 6.41, 6.42 | 0.09, 0.10 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 5.06, 5.06 | 0.09, 0.09 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.35, 0.35 | 0.27, 0.27 | -0.08, -0.08 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.09 | 4.15, 4.15 | 0.06, 0.07 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.68, 3.69 | 3.63, 3.63 | -0.06, -0.06 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.47 | 3.41, 3.41 | -0.05, -0.06 |

### 3 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.43 | 9.92, 9.92 | -1.51, -1.51 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.78, 11.78 | 11.13, 11.13 | -0.65, -0.65 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.84, 3.83 | 3.37, 3.37 | -0.47, -0.46 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.93, 7.92 | 7.50, 7.50 | -0.44, -0.42 |
| `MUL_MAT output:q5_0:3072x248320` | 6.29, 6.29 | 5.96, 5.95 | -0.34, -0.34 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.15, 5.15 | 4.86, 4.86 | -0.29, -0.29 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 1.97, 1.96 | 1.84, 1.84 | -0.13, -0.12 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.36, 0.36 | 0.24, 0.24 | -0.12, -0.12 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 1.62, 1.61 | 1.50, 1.50 | -0.12, -0.11 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.40, 0.40 | 0.31, 0.31 | -0.10, -0.09 |

### 3 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.45, 11.48 | 9.97, 10.00 | -1.48, -1.47 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.81, 11.82 | 11.17, 11.18 | -0.63, -0.63 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.81, 3.81 | 3.37, 3.38 | -0.43, -0.44 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.97, 7.96 | 7.53, 7.53 | -0.44, -0.43 |
| `MUL_MAT output:q5_0:3072x248320` | 6.33, 6.32 | 5.98, 5.98 | -0.35, -0.34 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.16, 5.17 | 4.88, 4.89 | -0.28, -0.28 |
| `FLASH_ATTN_EXT node` | 17.97, 17.99 | 17.82, 17.84 | -0.15, -0.15 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 1.96, 1.96 | 1.82, 1.82 | -0.14, -0.14 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.36, 0.36 | 0.24, 0.24 | -0.12, -0.12 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 1.61, 1.61 | 1.50, 1.50 | -0.11, -0.11 |

### 4 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 14.14, 14.13 | 15.55, 15.53 | 1.41, 1.40 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 9.48, 9.46 | 10.45, 10.43 | 0.97, 0.97 |
| `MUL_MAT output:q5_0:3072x248320` | 7.61, 7.60 | 8.36, 8.37 | 0.75, 0.77 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 6.18, 6.17 | 6.81, 6.80 | 0.63, 0.63 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.48, 12.46 | 11.96, 11.94 | -0.52, -0.52 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.21, 6.20 | 6.54, 6.53 | 0.33, 0.33 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.06, 6.06 | 6.36, 6.36 | 0.30, 0.30 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.25, 12.24 | 12.54, 12.52 | 0.29, 0.28 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.19, 5.18 | 5.44, 5.44 | 0.26, 0.26 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 8.84, 8.82 | 9.00, 8.99 | 0.16, 0.17 |

### 4 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 14.19, 14.20 | 15.60, 15.61 | 1.41, 1.41 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 9.50, 9.52 | 10.48, 10.49 | 0.98, 0.97 |
| `MUL_MAT output:q5_0:3072x248320` | 7.64, 7.62 | 8.40, 8.39 | 0.75, 0.77 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 6.20, 6.20 | 6.84, 6.84 | 0.64, 0.64 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.51, 12.52 | 12.00, 12.01 | -0.51, -0.51 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.75, 12.79 | 12.59, 12.61 | -0.16, -0.18 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 2.03, 1.98 | 2.16, 2.18 | 0.14, 0.20 |
| `FLASH_ATTN_EXT node` | 18.03, 18.05 | 17.87, 17.89 | -0.17, -0.16 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.49, 6.49 | 6.36, 6.36 | -0.12, -0.13 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.44, 2.46 | 2.59, 2.55 | 0.15, 0.10 |

## 4. The one-token step, ms (the host-profile load; nothing synchronized)

| depth | this boot (w1h1): step | its GPU time, nine cards | the rest | R176 on v4.0 at stock (w1h1, w1h2): step | GPU time | change of the step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45.61 | 44.02 | 1.56 | 53.73, 53.74 | 52.15, 52.16 | -15.11 % |
| 8,192 | 55.16 | 53.45 | 1.71 | 64.42, 64.44 | 62.71, 62.73 | -14.38 % |

## 5. The break-even a draft needs, tokens a step (est.)

The step of k tokens (est.) = this run's verify pass × R176's ratio of the host profile's verify GPU time to the group profile's pass at the same width + R176's verify rest + R176's draft span less its GPU time + R176's draft GPU time × (this boot's one-token GPU time / R176's). Break-even = that step / this boot's one-token step. "Verify alone" leaves the draft's span out: the price of a verify for any draft. Tokens a step and ms a token are this greedy reply's.

| width | depth | arm | the step (est.) | break-even | verify alone: break-even | tokens a step (this reply) | ms a token (est.) | one-token step | R176 measured: step, break-even |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | base | 97.41 | 2.14 | 1.98 | 1.90, 1.90 | 51.38 | 45.61 | 108.77, 2.02 |
| 2 | 0 | ro | 94.36 | 2.07 | 1.92 | 1.90, 1.90 | 49.77 | 45.61 | 108.77, 2.02 |
| 2 | 8,192 | base | 110.96 | 2.01 | 1.85 | 1.90, 1.90 | 58.52 | 55.16 | 123.58, 1.92 |
| 2 | 8,192 | ro | 107.63 | 1.95 | 1.79 | 1.70, 1.70 | 63.50 | 55.16 | 123.58, 1.92 |
| 3 | 0 | base | 128.34 | 2.81 | 2.56 | 2.56, 2.56 | 50.13 | 45.61 | 145.15, 2.70 |
| 3 | 0 | ro | 124.19 | 2.72 | 2.47 | 2.67, 2.67 | 46.56 | 45.61 | 145.15, 2.70 |
| 3 | 8,192 | base | 149.94 | 2.72 | 2.45 | 2.61, 2.61 | 57.41 | 55.16 | 167.93, 2.61 |
| 3 | 8,192 | ro | 145.30 | 2.63 | 2.36 | 2.06, 2.06 | 70.36 | 55.16 | 167.93, 2.61 |
| 4 | 0 | base | 155.57 | 3.41 | 3.06 | 2.81, 2.81 | 55.30 | 45.61 | 176.40, 3.28 |
| 4 | 0 | ro | 160.44 | 3.52 | 3.16 | 3.24, 3.24 | 49.50 | 45.61 | 176.40, 3.28 |
| 4 | 8,192 | base | 179.51 | 3.25 | 2.88 | 3.05, 3.05 | 58.89 | 55.16 | 202.17, 3.14 |
| 4 | 8,192 | ro | 182.06 | 3.30 | 2.92 | 2.81, 2.81 | 64.72 | 55.16 | 202.17, 3.14 |

## 6. The replies (greedy, 256 tokens): md5 of the token ids, first 8 characters

| width | depth | base | ro | the arms' replies | first token where ro's first load differs from base's | one-token reply (w1h1) | R176's reply at this width (v4.0) |
|---:|---:|---|---|---|---:|---|---|
| 1 | 0 | b8279751 | | | | | b8279751 |
| 1 | 8,192 | eea8bef3 | | | | | eea8bef3 |
| 2 | 0 | b8279751, b8279751 | f1051d98, f1051d98 | each arm its own | 1 | base | base |
| 2 | 8,192 | dacfc86f, dacfc86f | bb0720f0, bb0720f0 | each arm its own | 1 | neither | base |
| 3 | 0 | b8279751, b8279751 | f1051d98, f1051d98 | each arm its own | 1 | base | base |
| 3 | 8,192 | dacfc86f, dacfc86f | bb0720f0, bb0720f0 | each arm its own | 1 | neither | base |
| 4 | 0 | 69f664b1, 69f664b1 | 4b74571a, 4b74571a | each arm its own | 8 | neither | base |
| 4 | 8,192 | b5bc3400, b5bc3400 | 0aa44b4a, 0aa44b4a | each arm its own | 1 | neither | base |

