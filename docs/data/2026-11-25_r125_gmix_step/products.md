# The verify pass by part and product: `pmix` against `gmix`

ms a verify pass (four tokens), the kept passes of the request's typical steps (verify_parts.py), the profiler's cost subtracted per span; one load per file.

## Request d0p2

Kept verify passes: `pmix` 82, `gmix` 100.

### 1. By part (R41's parts), all cards

| Part | `pmix` | `gmix` | `gmix` − `pmix` |
|---|---:|---:|---:|
| routed experts | 20.388 | 36.210 | +15.822 |
| shared expert | 3.905 | 4.269 | +0.364 |
| hyper-connections (BF16) | 14.250 | 14.190 | -0.060 |
| router | 1.415 | 1.421 | +0.006 |
| attention and recurrent projections | 18.345 | 20.300 | +1.955 |
| Gated DeltaNet state and conv | 8.887 | 8.908 | +0.021 |
| attention (FLASH_ATTN_EXT) | 1.028 | 1.038 | +0.010 |
| output layer | 3.361 | 3.367 | +0.006 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 7.457 | 7.458 | +0.001 |
| other weights | 0.957 | 0.977 | +0.020 |
| sum of the parts | 79.992 | 98.140 | +18.148 |

### 2. By product, all cards (the label without types and shapes)

| Product | `pmix` types | `gmix` types | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 42.00 | 48.00 | 11.381 | 28.924 | +17.543 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps ffn_up_exps ffn_gate_shexp ffn_up_shexp` | q2_0 | - | 6.00 | 0.00 | 1.799 | 0.000 | -1.799 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.957 | 5.909 | +0.952 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.631 | 5.305 | +0.674 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 12.00 | 12.00 | 2.169 | 2.365 | +0.196 |
| `MUL_MAT ffn_gate_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.124 | 0.292 | +0.168 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 36.00 | 36.00 | 3.384 | 3.524 | +0.140 |
| `MUL_MAT ffn_up_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.100 | 0.231 | +0.131 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 48.00 | 48.00 | 7.208 | 7.287 | +0.078 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 12.00 | 12.00 | 2.019 | 1.948 | -0.071 |
| `MUL_MAT ffn_down_shexp` | q2_0, q4_0, q5_0, q8_0 | q2_0, q4_0, q5_0, q8_0 | 48.00 | 48.00 | 1.347 | 1.397 | +0.049 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.752 | 0.799 | +0.048 |
| `DSV4_HC_PRE hc_mixed` | - | - | 96.00 | 96.00 | 0.506 | 0.471 | -0.035 |
| `RMS_NORM+MUL norm` | - | - | 48.00 | 48.00 | 0.156 | 0.122 | -0.035 |
| `MUL_MAT hc_attn_up` | bf16 | bf16 | 48.00 | 48.00 | 2.902 | 2.873 | -0.029 |
| `MUL_MAT ssm_beta` | bf16 | bf16 | 36.00 | 36.00 | 0.313 | 0.342 | +0.029 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.189 | 0.213 | +0.025 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.225 | 0.200 | -0.025 |
| `MUL_MAT attn_k` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 12.00 | 12.00 | 0.267 | 0.291 | +0.024 |
| `MUL_MAT indexer.k_proj` | bf16 | bf16 | 12.00 | 12.00 | 0.186 | 0.209 | +0.023 |
| `ADD node` | - | - | 84.00 | 84.00 | 0.358 | 0.337 | -0.021 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.887 | 8.908 | +0.021 |
| `SCALE node` | - | - | 111.00 | 111.00 | 0.229 | 0.209 | -0.020 |
| the other 129 products (each under 0.02 ms apart) | | | | | 25.903 | 25.985 | +0.083 |
| **the pass (sum of the labels)** | | | | | **79.992** | **98.140** | **+18.148** |

### 3. The labels of the products above

| Label | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 0.00 | 2.00 | 0.000 | 1.094 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 0.00 | 7.00 | 0.000 | 6.848 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 0.00 | 4.00 | 0.000 | 4.167 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 0.00 | 11.00 | 0.000 | 10.310 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 42.00 | 24.00 | 11.381 | 6.506 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_u` | 6.00 | 0.00 | 1.799 | 0.000 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 31.00 | 23.00 | 4.348 | 3.247 |
| `MUL_MAT attn_qkv:q2_0:2560x10240` | 1.00 | 0.00 | 0.166 | 0.000 |
| `MUL_MAT attn_qkv:q4_K:2560x10240` | 4.00 | 9.00 | 0.442 | 0.995 |
| `MUL_MAT attn_qkv:q5_K:2560x10240` | 0.00 | 1.00 | 0.000 | 0.155 |
| `MUL_MAT attn_qkv:q6_K:2560x10240` | 0.00 | 3.00 | 0.000 | 1.513 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 20.00 | 17.00 | 2.118 | 1.779 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 11.00 | 6.00 | 1.533 | 0.841 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 3.00 | 5.00 | 0.530 | 0.884 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 2.00 | 8.00 | 0.450 | 1.800 |
| `MUL_MAT attn_output:q4_K:6144x2560` | 4.00 | 2.00 | 0.563 | 0.283 |
| `MUL_MAT attn_output:q5_K:6144x2560` | 4.00 | 3.00 | 0.709 | 0.532 |
| `MUL_MAT attn_output:q6_K:6144x2560` | 4.00 | 7.00 | 0.898 | 1.549 |
| `MUL_MAT ffn_gate_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.105 |
| `MUL_MAT ffn_gate_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.124 | 0.187 |
| `MUL_MAT attn_gate:iq4_xs:2560x6144` | 29.00 | 30.00 | 2.716 | 2.824 |
| `MUL_MAT attn_gate:q2_0:2560x6144` | 4.00 | 0.00 | 0.434 | 0.000 |
| `MUL_MAT attn_gate:q4_K:2560x6144` | 3.00 | 5.00 | 0.234 | 0.383 |
| `MUL_MAT attn_gate:q6_K:2560x6144` | 0.00 | 1.00 | 0.000 | 0.317 |
| `MUL_MAT ffn_up_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.079 |
| `MUL_MAT ffn_up_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.100 | 0.153 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 48.00 | 48.00 | 7.208 | 7.287 |
| `MUL_MAT attn_q:iq4_xs:2560x12288` | 10.00 | 8.00 | 1.628 | 1.299 |
| `MUL_MAT attn_q:q2_0:2560x12288` | 2.00 | 2.00 | 0.391 | 0.391 |
| `MUL_MAT attn_q:q4_K:2560x12288` | 0.00 | 2.00 | 0.000 | 0.259 |
| `MUL_MAT ffn_down_shexp:q2_0:640x2560` | 16.00 | 15.00 | 0.350 | 0.317 |
| `MUL_MAT ffn_down_shexp:q4_0:640x2560` | 16.00 | 11.00 | 0.380 | 0.264 |
| `MUL_MAT ffn_down_shexp:q5_0:640x2560` | 14.00 | 21.00 | 0.523 | 0.769 |
| `MUL_MAT ffn_down_shexp:q8_0:640x2560` | 2.00 | 1.00 | 0.094 | 0.047 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.752 | 0.799 |
| `DSV4_HC_PRE hc_mixed` | 96.00 | 96.00 | 0.506 | 0.471 |
| `RMS_NORM+MUL norm` | 48.00 | 48.00 | 0.156 | 0.122 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 48.00 | 48.00 | 2.902 | 2.873 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 36.00 | 36.00 | 0.313 | 0.342 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.189 | 0.213 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.225 | 0.200 |
| `MUL_MAT attn_k:iq4_xs:2560x512` | 7.00 | 5.00 | 0.139 | 0.097 |
| `MUL_MAT attn_k:q4_K:2560x512` | 2.00 | 3.00 | 0.037 | 0.048 |
| `MUL_MAT attn_k:q5_K:2560x512` | 2.00 | 1.00 | 0.046 | 0.024 |
| `MUL_MAT attn_k:q6_K:2560x512` | 1.00 | 3.00 | 0.045 | 0.123 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 12.00 | 12.00 | 0.186 | 0.209 |
| `ADD node` | 84.00 | 84.00 | 0.358 | 0.337 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.887 | 8.908 |
| `SCALE node` | 111.00 | 111.00 | 0.229 | 0.209 |

## Request d8192p2

Kept verify passes: `pmix` 79, `gmix` 92.

### 1. By part (R41's parts), all cards

| Part | `pmix` | `gmix` | `gmix` − `pmix` |
|---|---:|---:|---:|
| routed experts | 20.965 | 36.462 | +15.497 |
| shared expert | 3.875 | 4.305 | +0.430 |
| hyper-connections (BF16) | 14.230 | 14.248 | +0.018 |
| router | 1.413 | 1.412 | -0.001 |
| attention and recurrent projections | 18.471 | 20.367 | +1.896 |
| Gated DeltaNet state and conv | 8.934 | 8.972 | +0.038 |
| attention (FLASH_ATTN_EXT) | 12.559 | 12.603 | +0.044 |
| output layer | 3.384 | 3.383 | -0.001 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 8.819 | 8.773 | -0.046 |
| other weights | 0.976 | 0.993 | +0.017 |
| sum of the parts | 93.625 | 111.517 | +17.892 |

### 2. By product, all cards (the label without types and shapes)

| Product | `pmix` types | `gmix` types | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 42.00 | 48.00 | 11.697 | 29.027 | +17.330 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps ffn_up_exps ffn_gate_shexp ffn_up_shexp` | q2_0 | - | 6.00 | 0.00 | 1.846 | 0.000 | -1.846 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.994 | 5.897 | +0.903 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.632 | 5.319 | +0.687 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 12.00 | 12.00 | 2.170 | 2.371 | +0.201 |
| `MUL_MAT ffn_gate_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.124 | 0.294 | +0.170 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 36.00 | 36.00 | 3.394 | 3.527 | +0.133 |
| `MUL_MAT ffn_up_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.102 | 0.234 | +0.133 |
| `MUL_MAT ffn_down_shexp` | q2_0, q4_0, q5_0, q8_0 | q2_0, q4_0, q5_0, q8_0 | 48.00 | 48.00 | 1.328 | 1.418 | +0.090 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 12.00 | 12.00 | 2.034 | 1.963 | -0.071 |
| `FLASH_ATTN_EXT node` | - | - | 12.00 | 12.00 | 12.559 | 12.603 | +0.044 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.158 | 0.114 | -0.043 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.934 | 8.972 | +0.038 |
| `MUL_MAT attn_k` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 12.00 | 12.00 | 0.270 | 0.297 | +0.027 |
| `CPY kpool_k_idxs` | - | - | 9.00 | 9.00 | 0.031 | 0.058 | +0.027 |
| `ADD node` | - | - | 84.00 | 84.00 | 0.341 | 0.367 | +0.026 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.756 | 0.731 | -0.025 |
| `MUL_MAT ffn_gate_inp_shexp` | f32 | f32 | 48.00 | 48.00 | 0.257 | 0.279 | +0.022 |
| the other 134 products (each under 0.02 ms apart) | | | | | 37.997 | 38.044 | +0.047 |
| **the pass (sum of the labels)** | | | | | **93.626** | **111.518** | **+17.892** |

### 3. The labels of the products above

| Label | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 0.00 | 2.00 | 0.000 | 1.102 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 0.00 | 7.00 | 0.000 | 6.773 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 0.00 | 4.00 | 0.000 | 4.202 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 0.00 | 11.00 | 0.000 | 10.300 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 42.00 | 24.00 | 11.697 | 6.651 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_u` | 6.00 | 0.00 | 1.846 | 0.000 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 31.00 | 23.00 | 4.385 | 3.225 |
| `MUL_MAT attn_qkv:q2_0:2560x10240` | 1.00 | 0.00 | 0.167 | 0.000 |
| `MUL_MAT attn_qkv:q4_K:2560x10240` | 4.00 | 9.00 | 0.443 | 0.998 |
| `MUL_MAT attn_qkv:q5_K:2560x10240` | 0.00 | 1.00 | 0.000 | 0.156 |
| `MUL_MAT attn_qkv:q6_K:2560x10240` | 0.00 | 3.00 | 0.000 | 1.517 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 20.00 | 17.00 | 2.113 | 1.776 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 11.00 | 6.00 | 1.539 | 0.846 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 3.00 | 5.00 | 0.530 | 0.886 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 2.00 | 8.00 | 0.450 | 1.811 |
| `MUL_MAT attn_output:q4_K:6144x2560` | 4.00 | 2.00 | 0.565 | 0.285 |
| `MUL_MAT attn_output:q5_K:6144x2560` | 4.00 | 3.00 | 0.710 | 0.530 |
| `MUL_MAT attn_output:q6_K:6144x2560` | 4.00 | 7.00 | 0.895 | 1.556 |
| `MUL_MAT ffn_gate_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.105 |
| `MUL_MAT ffn_gate_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.124 | 0.189 |
| `MUL_MAT attn_gate:iq4_xs:2560x6144` | 29.00 | 30.00 | 2.727 | 2.828 |
| `MUL_MAT attn_gate:q2_0:2560x6144` | 4.00 | 0.00 | 0.434 | 0.000 |
| `MUL_MAT attn_gate:q4_K:2560x6144` | 3.00 | 5.00 | 0.233 | 0.383 |
| `MUL_MAT attn_gate:q6_K:2560x6144` | 0.00 | 1.00 | 0.000 | 0.317 |
| `MUL_MAT ffn_up_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.081 |
| `MUL_MAT ffn_up_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.102 | 0.153 |
| `MUL_MAT ffn_down_shexp:q2_0:640x2560` | 16.00 | 15.00 | 0.346 | 0.312 |
| `MUL_MAT ffn_down_shexp:q4_0:640x2560` | 16.00 | 11.00 | 0.378 | 0.260 |
| `MUL_MAT ffn_down_shexp:q5_0:640x2560` | 14.00 | 21.00 | 0.510 | 0.799 |
| `MUL_MAT ffn_down_shexp:q8_0:640x2560` | 2.00 | 1.00 | 0.094 | 0.047 |
| `MUL_MAT attn_q:iq4_xs:2560x12288` | 10.00 | 8.00 | 1.645 | 1.310 |
| `MUL_MAT attn_q:q2_0:2560x12288` | 2.00 | 2.00 | 0.389 | 0.391 |
| `MUL_MAT attn_q:q4_K:2560x12288` | 0.00 | 2.00 | 0.000 | 0.261 |
| `FLASH_ATTN_EXT node` | 12.00 | 12.00 | 12.559 | 12.603 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.158 | 0.114 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.934 | 8.972 |
| `MUL_MAT attn_k:iq4_xs:2560x512` | 7.00 | 5.00 | 0.142 | 0.101 |
| `MUL_MAT attn_k:q4_K:2560x512` | 2.00 | 3.00 | 0.039 | 0.049 |
| `MUL_MAT attn_k:q5_K:2560x512` | 2.00 | 1.00 | 0.044 | 0.024 |
| `MUL_MAT attn_k:q6_K:2560x512` | 1.00 | 3.00 | 0.046 | 0.124 |
| `CPY kpool_k_idxs` | 9.00 | 9.00 | 0.031 | 0.058 |
| `ADD node` | 84.00 | 84.00 | 0.341 | 0.367 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.756 | 0.731 |
| `MUL_MAT ffn_gate_inp_shexp:f32:2560x1` | 48.00 | 48.00 | 0.257 | 0.279 |

## Request d0p1

Kept verify passes: `pmix` 83, `gmix` 101.

### 1. By part (R41's parts), all cards

| Part | `pmix` | `gmix` | `gmix` − `pmix` |
|---|---:|---:|---:|
| routed experts | 20.355 | 36.144 | +15.789 |
| shared expert | 3.896 | 4.287 | +0.391 |
| hyper-connections (BF16) | 14.188 | 14.229 | +0.041 |
| router | 1.413 | 1.420 | +0.007 |
| attention and recurrent projections | 18.364 | 20.275 | +1.911 |
| Gated DeltaNet state and conv | 8.878 | 8.906 | +0.028 |
| attention (FLASH_ATTN_EXT) | 1.030 | 1.034 | +0.004 |
| output layer | 3.360 | 3.359 | -0.001 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 7.445 | 7.368 | -0.077 |
| other weights | 0.960 | 0.980 | +0.020 |
| sum of the parts | 79.889 | 98.001 | +18.112 |

### 2. By product, all cards (the label without types and shapes)

| Product | `pmix` types | `gmix` types | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 42.00 | 48.00 | 11.364 | 28.876 | +17.512 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps ffn_up_exps ffn_gate_shexp ffn_up_shexp` | q2_0 | - | 6.00 | 0.00 | 1.796 | 0.000 | -1.796 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.948 | 5.889 | +0.941 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.630 | 5.306 | +0.675 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 12.00 | 12.00 | 2.169 | 2.364 | +0.195 |
| `MUL_MAT ffn_gate_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.122 | 0.294 | +0.172 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 36.00 | 36.00 | 3.379 | 3.522 | +0.143 |
| `MUL_MAT ffn_up_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.099 | 0.232 | +0.133 |
| `MUL_MAT ffn_down_shexp` | q2_0, q4_0, q5_0, q8_0 | q2_0, q4_0, q5_0, q8_0 | 48.00 | 48.00 | 1.329 | 1.410 | +0.082 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 48.00 | 48.00 | 7.195 | 7.267 | +0.072 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 12.00 | 12.00 | 2.016 | 1.944 | -0.072 |
| `MUL_MAT attn_k` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 12.00 | 12.00 | 0.261 | 0.295 | +0.033 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.878 | 8.906 | +0.028 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp ffn_up_shexp` | iq4_xs, q4_K, q5_K | iq4_xs, q2_0, q4_K, q5_K | 40.00 | 43.00 | 2.064 | 2.090 | +0.026 |
| `MUL_MAT ffn_gate_inp_shexp` | f32 | f32 | 48.00 | 48.00 | 0.281 | 0.260 | -0.021 |
| `SOFT_MAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | - | - | 48.00 | 48.00 | 0.527 | 0.506 | -0.021 |
| the other 136 products (each under 0.02 ms apart) | | | | | 28.828 | 28.840 | +0.012 |
| **the pass (sum of the labels)** | | | | | **79.888** | **98.002** | **+18.114** |

### 3. The labels of the products above

| Label | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 0.00 | 2.00 | 0.000 | 1.093 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 0.00 | 7.00 | 0.000 | 6.839 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 0.00 | 4.00 | 0.000 | 4.156 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 0.00 | 11.00 | 0.000 | 10.299 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 42.00 | 24.00 | 11.364 | 6.489 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_u` | 6.00 | 0.00 | 1.796 | 0.000 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 31.00 | 23.00 | 4.340 | 3.228 |
| `MUL_MAT attn_qkv:q2_0:2560x10240` | 1.00 | 0.00 | 0.166 | 0.000 |
| `MUL_MAT attn_qkv:q4_K:2560x10240` | 4.00 | 9.00 | 0.442 | 0.994 |
| `MUL_MAT attn_qkv:q5_K:2560x10240` | 0.00 | 1.00 | 0.000 | 0.155 |
| `MUL_MAT attn_qkv:q6_K:2560x10240` | 0.00 | 3.00 | 0.000 | 1.512 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 20.00 | 17.00 | 2.110 | 1.776 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 11.00 | 6.00 | 1.541 | 0.841 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 3.00 | 5.00 | 0.529 | 0.885 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 2.00 | 8.00 | 0.450 | 1.804 |
| `MUL_MAT attn_output:q4_K:6144x2560` | 4.00 | 2.00 | 0.562 | 0.283 |
| `MUL_MAT attn_output:q5_K:6144x2560` | 4.00 | 3.00 | 0.709 | 0.530 |
| `MUL_MAT attn_output:q6_K:6144x2560` | 4.00 | 7.00 | 0.897 | 1.551 |
| `MUL_MAT ffn_gate_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.108 |
| `MUL_MAT ffn_gate_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.122 | 0.187 |
| `MUL_MAT attn_gate:iq4_xs:2560x6144` | 29.00 | 30.00 | 2.712 | 2.824 |
| `MUL_MAT attn_gate:q2_0:2560x6144` | 4.00 | 0.00 | 0.433 | 0.000 |
| `MUL_MAT attn_gate:q4_K:2560x6144` | 3.00 | 5.00 | 0.233 | 0.382 |
| `MUL_MAT attn_gate:q6_K:2560x6144` | 0.00 | 1.00 | 0.000 | 0.316 |
| `MUL_MAT ffn_up_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.080 |
| `MUL_MAT ffn_up_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.099 | 0.152 |
| `MUL_MAT ffn_down_shexp:q2_0:640x2560` | 16.00 | 15.00 | 0.346 | 0.312 |
| `MUL_MAT ffn_down_shexp:q4_0:640x2560` | 16.00 | 11.00 | 0.375 | 0.268 |
| `MUL_MAT ffn_down_shexp:q5_0:640x2560` | 14.00 | 21.00 | 0.514 | 0.783 |
| `MUL_MAT ffn_down_shexp:q8_0:640x2560` | 2.00 | 1.00 | 0.093 | 0.047 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 48.00 | 48.00 | 7.195 | 7.267 |
| `MUL_MAT attn_q:iq4_xs:2560x12288` | 10.00 | 8.00 | 1.626 | 1.296 |
| `MUL_MAT attn_q:q2_0:2560x12288` | 2.00 | 2.00 | 0.391 | 0.390 |
| `MUL_MAT attn_q:q4_K:2560x12288` | 0.00 | 2.00 | 0.000 | 0.259 |
| `MUL_MAT attn_k:iq4_xs:2560x512` | 7.00 | 5.00 | 0.136 | 0.100 |
| `MUL_MAT attn_k:q4_K:2560x512` | 2.00 | 3.00 | 0.037 | 0.048 |
| `MUL_MAT attn_k:q5_K:2560x512` | 2.00 | 1.00 | 0.043 | 0.024 |
| `MUL_MAT attn_k:q6_K:2560x512` | 1.00 | 3.00 | 0.045 | 0.123 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.878 | 8.906 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:iq4_xs:2560x640 ffn_up_shexp:iq4_xs:2560x640` | 27.00 | 22.00 | 1.535 | 1.248 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.00 | 4.00 | 0.000 | 0.142 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q4_K:2560x640 ffn_up_shexp:q4_K:2560x640` | 10.00 | 13.00 | 0.392 | 0.518 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q5_K:2560x640 ffn_up_shexp:q5_K:2560x640` | 3.00 | 4.00 | 0.138 | 0.183 |
| `MUL_MAT ffn_gate_inp_shexp:f32:2560x1` | 48.00 | 48.00 | 0.281 | 0.260 |
| `SOFT_MAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | 48.00 | 48.00 | 0.527 | 0.506 |

## Request d8192p1

Kept verify passes: `pmix` 82, `gmix` 92.

### 1. By part (R41's parts), all cards

| Part | `pmix` | `gmix` | `gmix` − `pmix` |
|---|---:|---:|---:|
| routed experts | 21.007 | 36.441 | +15.434 |
| shared expert | 3.895 | 4.299 | +0.404 |
| hyper-connections (BF16) | 14.249 | 14.233 | -0.016 |
| router | 1.413 | 1.412 | -0.001 |
| attention and recurrent projections | 18.457 | 20.354 | +1.897 |
| Gated DeltaNet state and conv | 8.935 | 8.976 | +0.041 |
| attention (FLASH_ATTN_EXT) | 12.559 | 12.596 | +0.037 |
| output layer | 3.383 | 3.378 | -0.005 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 8.777 | 8.787 | +0.010 |
| other weights | 0.996 | 0.982 | -0.014 |
| sum of the parts | 93.670 | 111.460 | +17.790 |

### 2. By product, all cards (the label without types and shapes)

| Product | `pmix` types | `gmix` types | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 42.00 | 48.00 | 11.723 | 29.017 | +17.294 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps ffn_up_exps ffn_gate_shexp ffn_up_shexp` | q2_0 | - | 6.00 | 0.00 | 1.850 | 0.000 | -1.850 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.970 | 5.903 | +0.932 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 4.634 | 5.306 | +0.672 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 12.00 | 12.00 | 2.171 | 2.374 | +0.203 |
| `MUL_MAT ffn_gate_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.125 | 0.295 | +0.170 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 36.00 | 36.00 | 3.386 | 3.522 | +0.136 |
| `MUL_MAT ffn_up_shexp` | q6_K | iq3_s, q6_K | 2.00 | 5.00 | 0.101 | 0.231 | +0.130 |
| `MUL_MAT ffn_down_shexp` | q2_0, q4_0, q5_0, q8_0 | q2_0, q4_0, q5_0, q8_0 | 48.00 | 48.00 | 1.332 | 1.411 | +0.079 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 12.00 | 12.00 | 2.034 | 1.958 | -0.075 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.935 | 8.976 | +0.041 |
| `FLASH_ATTN_EXT node` | - | - | 12.00 | 12.00 | 12.558 | 12.596 | +0.038 |
| `CPY kpool_k_idxs` | - | - | 9.00 | 9.00 | 0.029 | 0.067 | +0.038 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.382 | 0.344 | -0.038 |
| `FILL node` | - | - | 40.00 | 40.00 | 0.151 | 0.123 | -0.029 |
| `ADD node` | - | - | 84.00 | 84.00 | 0.334 | 0.361 | +0.027 |
| `MUL_MAT attn_k` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 12.00 | 12.00 | 0.274 | 0.295 | +0.021 |
| `DSV4_HC_PRE hc_mixed` | - | - | 96.00 | 96.00 | 0.511 | 0.491 | -0.021 |
| `CONCAT node` | - | - | 13.00 | 13.00 | 0.156 | 0.176 | +0.021 |
| the other 133 products (each under 0.02 ms apart) | | | | | 38.013 | 38.015 | +0.001 |
| **the pass (sum of the labels)** | | | | | **93.669** | **111.460** | **+17.791** |

### 3. The labels of the products above

| Label | `pmix` groups | `gmix` groups | `pmix` ms | `gmix` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 0.00 | 2.00 | 0.000 | 1.099 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 0.00 | 7.00 | 0.000 | 6.764 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 0.00 | 4.00 | 0.000 | 4.205 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 0.00 | 11.00 | 0.000 | 10.302 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 42.00 | 24.00 | 11.723 | 6.647 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_u` | 6.00 | 0.00 | 1.850 | 0.000 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 31.00 | 23.00 | 4.360 | 3.232 |
| `MUL_MAT attn_qkv:q2_0:2560x10240` | 1.00 | 0.00 | 0.167 | 0.000 |
| `MUL_MAT attn_qkv:q4_K:2560x10240` | 4.00 | 9.00 | 0.443 | 0.998 |
| `MUL_MAT attn_qkv:q5_K:2560x10240` | 0.00 | 1.00 | 0.000 | 0.156 |
| `MUL_MAT attn_qkv:q6_K:2560x10240` | 0.00 | 3.00 | 0.000 | 1.518 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 20.00 | 17.00 | 2.115 | 1.775 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 11.00 | 6.00 | 1.539 | 0.838 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 3.00 | 5.00 | 0.530 | 0.887 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 2.00 | 8.00 | 0.450 | 1.807 |
| `MUL_MAT attn_output:q4_K:6144x2560` | 4.00 | 2.00 | 0.566 | 0.287 |
| `MUL_MAT attn_output:q5_K:6144x2560` | 4.00 | 3.00 | 0.710 | 0.532 |
| `MUL_MAT attn_output:q6_K:6144x2560` | 4.00 | 7.00 | 0.896 | 1.554 |
| `MUL_MAT ffn_gate_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.106 |
| `MUL_MAT ffn_gate_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.125 | 0.189 |
| `MUL_MAT attn_gate:iq4_xs:2560x6144` | 29.00 | 30.00 | 2.720 | 2.821 |
| `MUL_MAT attn_gate:q2_0:2560x6144` | 4.00 | 0.00 | 0.434 | 0.000 |
| `MUL_MAT attn_gate:q4_K:2560x6144` | 3.00 | 5.00 | 0.232 | 0.383 |
| `MUL_MAT attn_gate:q6_K:2560x6144` | 0.00 | 1.00 | 0.000 | 0.317 |
| `MUL_MAT ffn_up_shexp:iq3_s:2560x640` | 0.00 | 2.00 | 0.000 | 0.078 |
| `MUL_MAT ffn_up_shexp:q6_K:2560x640` | 2.00 | 3.00 | 0.101 | 0.153 |
| `MUL_MAT ffn_down_shexp:q2_0:640x2560` | 16.00 | 15.00 | 0.346 | 0.314 |
| `MUL_MAT ffn_down_shexp:q4_0:640x2560` | 16.00 | 11.00 | 0.377 | 0.259 |
| `MUL_MAT ffn_down_shexp:q5_0:640x2560` | 14.00 | 21.00 | 0.515 | 0.792 |
| `MUL_MAT ffn_down_shexp:q8_0:640x2560` | 2.00 | 1.00 | 0.094 | 0.047 |
| `MUL_MAT attn_q:iq4_xs:2560x12288` | 10.00 | 8.00 | 1.644 | 1.306 |
| `MUL_MAT attn_q:q2_0:2560x12288` | 2.00 | 2.00 | 0.390 | 0.391 |
| `MUL_MAT attn_q:q4_K:2560x12288` | 0.00 | 2.00 | 0.000 | 0.261 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.935 | 8.976 |
| `FLASH_ATTN_EXT node` | 12.00 | 12.00 | 12.558 | 12.596 |
| `CPY kpool_k_idxs` | 9.00 | 9.00 | 0.029 | 0.067 |
| `MUL gate` | 36.00 | 36.00 | 0.382 | 0.344 |
| `FILL node` | 40.00 | 40.00 | 0.151 | 0.123 |
| `ADD node` | 84.00 | 84.00 | 0.334 | 0.361 |
| `MUL_MAT attn_k:iq4_xs:2560x512` | 7.00 | 5.00 | 0.142 | 0.099 |
| `MUL_MAT attn_k:q4_K:2560x512` | 2.00 | 3.00 | 0.039 | 0.049 |
| `MUL_MAT attn_k:q5_K:2560x512` | 2.00 | 1.00 | 0.048 | 0.024 |
| `MUL_MAT attn_k:q6_K:2560x512` | 1.00 | 3.00 | 0.046 | 0.124 |
| `DSV4_HC_PRE hc_mixed` | 96.00 | 96.00 | 0.511 | 0.491 |
| `CONCAT node` | 13.00 | 13.00 | 0.156 | 0.176 |

