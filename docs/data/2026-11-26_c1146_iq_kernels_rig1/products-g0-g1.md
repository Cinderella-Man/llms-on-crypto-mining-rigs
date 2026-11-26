# The verify pass by part and product: `unset` against `c`

ms a verify pass (four tokens), the kept passes of the request's typical steps (verify_parts.py), the profiler's cost subtracted per span; one load per file.

## Request d0p2

Kept verify passes: `unset` 100, `c` 100.

### 1. By part (R41's parts), all cards

| Part | `unset` | `c` | `c` − `unset` |
|---|---:|---:|---:|
| routed experts | 35.902 | 26.364 | -9.538 |
| shared expert | 4.192 | 4.182 | -0.010 |
| hyper-connections (BF16) | 12.825 | 12.830 | +0.005 |
| router | 1.299 | 1.287 | -0.012 |
| attention and recurrent projections | 20.274 | 20.212 | -0.062 |
| Gated DeltaNet state and conv | 8.942 | 8.930 | -0.012 |
| attention (FLASH_ATTN_EXT) | 1.040 | 1.040 | +0.000 |
| output layer | 3.154 | 3.154 | +0.000 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 7.397 | 7.432 | +0.035 |
| other weights | 0.913 | 0.917 | +0.004 |
| sum of the parts | 95.938 | 86.349 | -9.589 |

### 2. By product, all cards (the label without types and shapes)

| Product | `unset` types | `c` types | `unset` groups | `c` groups | `unset` ms | `c` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 28.739 | 19.203 | -9.536 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.365 | 0.443 | +0.078 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.266 | 0.215 | -0.051 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 5.308 | 5.274 | -0.034 |
| the other 147 products (each under 0.02 ms apart) | | | | | 61.261 | 61.213 | -0.048 |
| **the pass (sum of the labels)** | | | | | **95.939** | **86.349** | **-9.590** |

### 3. The labels of the products above

| Label | `unset` groups | `c` groups | `unset` ms | `c` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 1.098 | 0.662 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 6.889 | 3.744 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 4.151 | 2.391 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 10.317 | 6.123 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.283 | 6.283 |
| `MUL gate` | 36.00 | 36.00 | 0.365 | 0.443 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.266 | 0.215 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 17.00 | 17.00 | 1.776 | 1.778 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 6.00 | 6.00 | 0.840 | 0.842 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 5.00 | 5.00 | 0.879 | 0.879 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 8.00 | 8.00 | 1.814 | 1.776 |

## Request d8192p2

Kept verify passes: `unset` 92, `c` 92.

### 1. By part (R41's parts), all cards

| Part | `unset` | `c` | `c` − `unset` |
|---|---:|---:|---:|
| routed experts | 36.085 | 26.604 | -9.481 |
| shared expert | 4.201 | 4.221 | +0.020 |
| hyper-connections (BF16) | 12.810 | 12.823 | +0.013 |
| router | 1.296 | 1.272 | -0.024 |
| attention and recurrent projections | 20.398 | 20.368 | -0.030 |
| Gated DeltaNet state and conv | 9.002 | 8.977 | -0.025 |
| attention (FLASH_ATTN_EXT) | 12.619 | 12.614 | -0.005 |
| output layer | 3.155 | 3.156 | +0.001 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 8.722 | 8.744 | +0.022 |
| other weights | 0.909 | 0.971 | +0.062 |
| sum of the parts | 109.196 | 99.750 | -9.446 |

### 2. By product, all cards (the label without types and shapes)

| Product | `unset` types | `c` types | `unset` groups | `c` groups | `unset` ms | `c` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 28.783 | 19.304 | -9.479 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.236 | 0.352 | +0.116 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.283 | 0.215 | -0.068 |
| `MUL_MAT hc_attn_up` | bf16 | bf16 | 48.00 | 48.00 | 2.572 | 2.622 | +0.049 |
| `MUL_MAT indexer.k_proj` | bf16 | bf16 | 12.00 | 12.00 | 0.195 | 0.236 | +0.041 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 5.322 | 5.285 | -0.037 |
| `REPEAT node` | - | - | 1.00 | 1.00 | 0.005 | 0.036 | +0.031 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.764 | 0.735 | -0.028 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 9.002 | 8.977 | -0.026 |
| `MUL_MAT ffn_gate_inp` | bf16 | bf16 | 48.00 | 48.00 | 1.296 | 1.272 | -0.023 |
| `RMS_NORM+MUL node` | - | - | 100.00 | 100.00 | 0.538 | 0.561 | +0.023 |
| `MUL_MAT hc_attn_inject` | bf16 | bf16 | 48.00 | 48.00 | 0.383 | 0.361 | -0.022 |
| `MUL_MAT attn_qkv` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 5.917 | 5.896 | -0.021 |
| the other 138 products (each under 0.02 ms apart) | | | | | 53.900 | 53.898 | -0.003 |
| **the pass (sum of the labels)** | | | | | **109.196** | **99.750** | **-9.447** |

### 3. The labels of the products above

| Label | `unset` groups | `c` groups | `unset` ms | `c` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 1.104 | 0.662 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 6.769 | 3.685 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 4.200 | 2.412 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 10.297 | 6.139 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.414 | 6.406 |
| `MUL gate` | 36.00 | 36.00 | 0.236 | 0.352 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.283 | 0.215 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 48.00 | 48.00 | 2.572 | 2.622 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 12.00 | 12.00 | 0.195 | 0.236 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 17.00 | 17.00 | 1.778 | 1.782 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 6.00 | 6.00 | 0.852 | 0.844 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 5.00 | 5.00 | 0.879 | 0.880 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 8.00 | 8.00 | 1.813 | 1.780 |
| `REPEAT node` | 1.00 | 1.00 | 0.005 | 0.036 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.764 | 0.735 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 9.002 | 8.977 |
| `MUL_MAT ffn_gate_inp:bf16:2560x512` | 48.00 | 48.00 | 1.296 | 1.272 |
| `RMS_NORM+MUL node` | 100.00 | 100.00 | 0.538 | 0.561 |
| `MUL_MAT hc_attn_inject:bf16:10240x4` | 48.00 | 48.00 | 0.383 | 0.361 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 23.00 | 23.00 | 3.262 | 3.228 |
| `MUL_MAT attn_qkv:q4_K:2560x10240` | 9.00 | 9.00 | 0.992 | 0.989 |
| `MUL_MAT attn_qkv:q5_K:2560x10240` | 1.00 | 1.00 | 0.147 | 0.148 |
| `MUL_MAT attn_qkv:q6_K:2560x10240` | 3.00 | 3.00 | 1.516 | 1.530 |

