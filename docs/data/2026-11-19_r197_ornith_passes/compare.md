# R197: arm base (Ornith's v5 preset) against arm ro (+ the read-once switches), request p2 (the repeat); each load by value, in run order

## 1. The verify pass, GPU ms a pass, nine cards (the group profile's launch groups summed)

| width | depth | base | ro | ro − base, paired | change of the means | spread base, ro | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | 85.99, 86.02 | 82.98, 83.08 | -3.01, -2.95 | -3.47 % | 0.03, 0.10 | **gain** |
| 2 | 8,192 | 97.67, 97.68 | 94.30, 94.41 | -3.37, -3.27 | -3.40 % | 0.01, 0.11 | **gain** |
| 3 | 0 | 111.50, 111.49 | 107.36, 107.42 | -4.14, -4.06 | -3.68 % | 0.01, 0.06 | **gain** |
| 3 | 8,192 | 129.32, 129.38 | 124.68, 124.72 | -4.63, -4.66 | -3.59 % | 0.06, 0.03 | **gain** |
| 4 | 0 | 132.89, 132.82 | 137.82, 137.67 | 4.94, 4.85 | +3.68 % | 0.06, 0.15 | **loss** |
| 4 | 8,192 | 151.95, 152.07 | 154.53, 154.55 | 2.58, 2.48 | +1.66 % | 0.13, 0.02 | **loss** |

## 2. The pass by family of launch groups, ms a pass, nine cards (families that moved 0.05 ms or more)

### 2 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.94, 25.89 | 23.54, 23.53 | -2.40, -2.36 |
| attention layers: q, k, v, output products | 7.87, 7.86 | 6.97, 6.97 | -0.90, -0.89 |
| shared expert: gate, up and GLU | 2.95, 2.94 | 2.87, 2.85 | -0.07, -0.09 |
| shared expert: down | 1.57, 1.56 | 1.60, 1.62 | 0.04, 0.05 |
| routed experts: down | 9.66, 9.67 | 9.74, 9.75 | 0.08, 0.07 |
| output head | 4.97, 4.95 | 5.06, 5.04 | 0.09, 0.09 |
| everything else (norms, adds, copies, the sampling chain) | 7.88, 7.93 | 7.97, 8.02 | 0.10, 0.09 |
| routed experts: gate, up and GLU | 16.80, 16.82 | 16.90, 16.92 | 0.10, 0.10 |

### 2 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.88, 25.86 | 23.54, 23.52 | -2.34, -2.34 |
| attention layers: q, k, v, output products | 7.91, 7.86 | 6.97, 6.99 | -0.93, -0.87 |
| shared expert: gate, up and GLU | 2.93, 2.94 | 2.88, 2.87 | -0.05, -0.07 |
| router (ffn_gate_inp) and its top-k | 1.74, 1.73 | 1.70, 1.68 | -0.04, -0.05 |
| output head | 4.98, 4.97 | 5.06, 5.06 | 0.09, 0.09 |

### 3 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 32.02, 32.05 | 29.42, 29.41 | -2.60, -2.64 |
| attention layers: q, k, v, output products | 9.75, 9.76 | 8.79, 8.79 | -0.96, -0.96 |
| output head | 6.31, 6.30 | 5.98, 5.96 | -0.34, -0.34 |
| shared expert: gate, up and GLU | 3.59, 3.58 | 3.35, 3.35 | -0.24, -0.24 |
| everything else (norms, adds, copies, the sampling chain) | 8.66, 8.57 | 8.53, 8.59 | -0.12, 0.02 |
| routed experts: down | 13.75, 13.77 | 13.81, 13.82 | 0.06, 0.05 |
| routed experts: gate, up and GLU | 23.65, 23.68 | 23.75, 23.77 | 0.10, 0.09 |

### 3 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 32.13, 32.15 | 29.51, 29.51 | -2.62, -2.64 |
| attention layers: q, k, v, output products | 9.73, 9.75 | 8.81, 8.82 | -0.92, -0.93 |
| output head | 6.32, 6.33 | 5.99, 6.00 | -0.32, -0.34 |
| shared expert: gate, up and GLU | 3.58, 3.58 | 3.32, 3.32 | -0.26, -0.26 |
| routed experts: gate, up and GLU | 24.18, 24.21 | 24.04, 24.04 | -0.15, -0.16 |
| attention: FLASH_ATTN_EXT | 17.99, 18.01 | 17.84, 17.86 | -0.15, -0.15 |
| routed experts: down | 14.04, 14.05 | 13.94, 13.94 | -0.10, -0.11 |
| everything else (norms, adds, copies, the sampling chain) | 8.61, 8.56 | 8.53, 8.55 | -0.08, -0.01 |

### 4 tokens, depth 0

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| everything else (norms, adds, copies, the sampling chain) | 8.73, 8.69 | 8.75, 8.75 | 0.03, 0.07 |
| shared expert: gate, up and GLU | 4.35, 4.39 | 4.58, 4.59 | 0.23, 0.20 |
| attention layers: q, k, v, output products | 11.23, 11.23 | 11.61, 11.60 | 0.38, 0.37 |
| routed experts: down | 17.44, 17.45 | 17.99, 17.99 | 0.55, 0.54 |
| output head | 7.62, 7.60 | 8.38, 8.37 | 0.76, 0.76 |
| routed experts: gate, up and GLU | 29.81, 29.80 | 30.76, 30.73 | 0.95, 0.93 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 37.08, 37.09 | 39.09, 39.03 | 2.01, 1.94 |

### 4 tokens, depth 8,192

| family | base | ro | ro − base, paired |
|---|---:|---:|---:|
| routed experts: gate, up and GLU | 31.06, 31.10 | 30.68, 30.68 | -0.38, -0.42 |
| routed experts: down | 18.16, 18.21 | 17.91, 17.92 | -0.25, -0.29 |
| attention: FLASH_ATTN_EXT | 18.04, 18.06 | 17.88, 17.88 | -0.16, -0.18 |
| shared expert: gate, up and GLU | 4.45, 4.45 | 4.73, 4.72 | 0.28, 0.27 |
| attention layers: q, k, v, output products | 11.23, 11.23 | 11.61, 11.62 | 0.39, 0.39 |
| output head | 7.64, 7.63 | 8.40, 8.40 | 0.77, 0.78 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 37.14, 37.23 | 39.13, 39.13 | 1.99, 1.90 |

## 3. The dense Q5_0 and Q8_0 products (launch groups with a plain MUL_MAT of a q5_0 or q8_0 tensor), ms a pass

| width | depth | base | ro | ro − base, paired | change of the means | groups |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 43.29, 43.21 | 40.04, 40.01 | -3.25, -3.19 | -7.44 % | 13 |
| 2 | 8,192 | 43.26, 43.21 | 40.07, 40.06 | -3.19, -3.15 | -7.33 % | 13 |
| 3 | 0 | 53.66, 53.68 | 49.49, 49.46 | -4.17, -4.23 | -7.82 % | 13 |
| 3 | 8,192 | 53.75, 53.80 | 49.58, 49.59 | -4.17, -4.22 | -7.80 % | 13 |
| 4 | 0 | 62.77, 62.78 | 66.15, 66.05 | 3.38, 3.27 | +5.30 % | 13 |
| 4 | 8,192 | 62.85, 62.90 | 66.25, 66.26 | 3.41, 3.35 | +5.37 % | 13 |

### 2 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.26 | 6.71, 6.72 | -2.56, -2.55 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.12, 3.12 | 2.34, 2.34 | -0.78, -0.78 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.42, 9.41 | 9.54, 9.55 | 0.12, 0.14 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.32, 0.31 | 0.21, 0.21 | -0.11, -0.10 |
| `MUL_MAT output:q5_0:3072x248320` | 4.97, 4.95 | 5.06, 5.04 | 0.09, 0.09 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 6.39, 6.40 | 0.08, 0.08 |
| `RMS_NORM+SCALE node` | 0.47, 0.36 | 0.47, 0.51 | 0.01, 0.15 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.35, 0.35 | 0.27, 0.27 | -0.08, -0.08 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 4.14, 4.15 | 0.06, 0.07 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 1.57, 1.56 | 1.60, 1.62 | 0.04, 0.05 |

### 2 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.28, 9.28 | 6.72, 6.73 | -2.55, -2.56 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.10, 3.10 | 2.33, 2.33 | -0.78, -0.77 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.45, 9.45 | 9.57, 9.58 | 0.13, 0.13 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.37, 0.33 | 0.23, 0.24 | -0.14, -0.09 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 6.42, 6.42 | 0.09, 0.10 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 5.06, 5.06 | 0.09, 0.09 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.35, 0.35 | 0.27, 0.27 | -0.08, -0.08 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.09, 4.09 | 4.15, 4.15 | 0.06, 0.07 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.69, 3.69 | 3.63, 3.63 | -0.06, -0.06 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.48 | 3.42, 3.42 | -0.06, -0.06 |

### 3 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.44 | 9.94, 9.94 | -1.49, -1.51 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.78, 11.80 | 11.15, 11.16 | -0.63, -0.64 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.84, 3.84 | 3.37, 3.37 | -0.46, -0.47 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.94, 7.94 | 7.51, 7.52 | -0.43, -0.42 |
| `MUL_MAT output:q5_0:3072x248320` | 6.31, 6.30 | 5.98, 5.96 | -0.34, -0.34 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.15, 5.16 | 4.87, 4.87 | -0.29, -0.29 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 1.97, 1.98 | 1.85, 1.85 | -0.13, -0.13 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.36, 0.36 | 0.25, 0.24 | -0.11, -0.12 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 1.61, 1.61 | 1.51, 1.50 | -0.11, -0.11 |
| `MUL_MAT attn_v:q8_0:3072x512` | 0.40, 0.40 | 0.31, 0.31 | -0.09, -0.09 |

### 3 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.49, 11.50 | 9.98, 9.99 | -1.51, -1.50 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.82, 11.83 | 11.20, 11.19 | -0.63, -0.64 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.81, 3.82 | 3.38, 3.38 | -0.44, -0.44 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.98, 7.97 | 7.54, 7.54 | -0.44, -0.43 |
| `MUL_MAT output:q5_0:3072x248320` | 6.32, 6.33 | 5.99, 6.00 | -0.32, -0.34 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.17, 5.18 | 4.89, 4.89 | -0.28, -0.29 |
| `FLASH_ATTN_EXT node` | 17.99, 18.01 | 17.84, 17.86 | -0.15, -0.15 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 1.96, 1.96 | 1.82, 1.82 | -0.14, -0.14 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.36, 0.36 | 0.24, 0.24 | -0.12, -0.12 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 1.61, 1.62 | 1.50, 1.50 | -0.11, -0.12 |

### 4 tokens, depth 0: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 14.16, 14.17 | 15.58, 15.56 | 1.42, 1.39 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 9.48, 9.48 | 10.47, 10.46 | 0.99, 0.98 |
| `MUL_MAT output:q5_0:3072x248320` | 7.62, 7.60 | 8.38, 8.37 | 0.76, 0.76 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 6.18, 6.18 | 6.82, 6.81 | 0.64, 0.64 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.49, 12.48 | 11.98, 11.97 | -0.51, -0.51 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.21, 6.21 | 6.53, 6.54 | 0.32, 0.32 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.06, 6.07 | 6.37, 6.36 | 0.30, 0.30 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.25, 12.25 | 12.54, 12.54 | 0.29, 0.29 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.19, 5.20 | 5.45, 5.45 | 0.26, 0.25 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 8.84, 8.84 | 9.01, 9.00 | 0.17, 0.16 |

### 4 tokens, depth 8,192: the 10 launch groups that moved most

| launch group (label) | base | ro | ro − base, paired |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 14.19, 14.21 | 15.60, 15.60 | 1.42, 1.40 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 9.51, 9.52 | 10.48, 10.48 | 0.98, 0.96 |
| `MUL_MAT output:q5_0:3072x248320` | 7.64, 7.63 | 8.40, 8.40 | 0.77, 0.78 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 6.20, 6.20 | 6.84, 6.84 | 0.64, 0.64 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.51, 12.52 | 12.00, 12.00 | -0.51, -0.52 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.76, 12.80 | 12.59, 12.60 | -0.17, -0.20 |
| `FLASH_ATTN_EXT node` | 18.04, 18.06 | 17.88, 17.88 | -0.16, -0.18 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 2.00, 2.01 | 2.16, 2.16 | 0.16, 0.15 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.48, 6.49 | 6.37, 6.36 | -0.12, -0.13 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.45, 2.44 | 2.58, 2.56 | 0.13, 0.12 |

## 4. The one-token step, ms (the host-profile load; nothing synchronized)

| depth | this boot (w1h1): step | its GPU time, nine cards | the rest | R176 on v4.0 at stock (w1h1, w1h2): step | GPU time | change of the step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45.60 | 44.05 | 1.55 | 53.72, 53.73 | 52.17, 52.18 | -15.13 % |
| 8,192 | 55.17 | 53.46 | 1.71 | 64.41, 64.42 | 62.72, 62.71 | -14.35 % |

## 5. The break-even a draft needs, tokens a step (est.)

The step of k tokens (est.) = this run's verify pass × R176's ratio of the host profile's verify GPU time to the group profile's pass at the same width + R176's verify rest + R176's draft span less its GPU time + R176's draft GPU time × (this boot's one-token GPU time / R176's). Break-even = that step / this boot's one-token step. "Verify alone" leaves the draft's span out: the price of a verify for any draft. Tokens a step and ms a token are this greedy reply's.

| width | depth | arm | the step (est.) | break-even | verify alone: break-even | tokens a step (this reply) | ms a token (est.) | one-token step | R176 measured: step, break-even |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | base | 97.39 | 2.14 | 1.98 | 1.90, 1.90 | 51.37 | 45.60 | 108.38, 2.02 |
| 2 | 0 | ro | 94.39 | 2.07 | 1.92 | 1.90, 1.90 | 49.78 | 45.60 | 108.38, 2.02 |
| 2 | 8,192 | base | 111.06 | 2.01 | 1.86 | 1.90, 1.90 | 58.57 | 55.17 | 123.60, 1.92 |
| 2 | 8,192 | ro | 107.72 | 1.95 | 1.79 | 1.70, 1.70 | 63.55 | 55.17 | 123.60, 1.92 |
| 3 | 0 | base | 128.42 | 2.82 | 2.56 | 2.56, 2.56 | 50.17 | 45.60 | 143.96, 2.68 |
| 3 | 0 | ro | 124.30 | 2.73 | 2.47 | 2.67, 2.67 | 46.60 | 45.60 | 143.96, 2.68 |
| 3 | 8,192 | base | 150.02 | 2.72 | 2.45 | 2.61, 2.61 | 57.44 | 55.17 | 167.91, 2.61 |
| 3 | 8,192 | ro | 145.35 | 2.63 | 2.36 | 2.06, 2.06 | 70.39 | 55.17 | 167.91, 2.61 |
| 4 | 0 | base | 155.68 | 3.41 | 3.06 | 2.81, 2.81 | 55.34 | 45.60 | 176.00, 3.28 |
| 4 | 0 | ro | 160.62 | 3.52 | 3.17 | 3.24, 3.24 | 49.56 | 45.60 | 176.00, 3.28 |
| 4 | 8,192 | base | 179.62 | 3.26 | 2.88 | 3.05, 3.05 | 58.93 | 55.17 | 202.27, 3.14 |
| 4 | 8,192 | ro | 182.16 | 3.30 | 2.93 | 2.81, 2.81 | 64.76 | 55.17 | 202.27, 3.14 |

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

