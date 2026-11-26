# The verify pass by part and product: `c` against `ptx`

ms a verify pass (four tokens), the kept passes of the request's typical steps (verify_parts.py), the profiler's cost subtracted per span; one load per file.

## Request d0p2

Kept verify passes: `c` 100, `ptx` 100.

### 1. By part (R41's parts), all cards

| Part | `c` | `ptx` | `ptx` − `c` |
|---|---:|---:|---:|
| routed experts | 26.364 | 24.094 | -2.270 |
| shared expert | 4.182 | 4.189 | +0.007 |
| hyper-connections (BF16) | 12.830 | 12.810 | -0.020 |
| router | 1.287 | 1.297 | +0.010 |
| attention and recurrent projections | 20.212 | 20.209 | -0.003 |
| Gated DeltaNet state and conv | 8.930 | 8.906 | -0.024 |
| attention (FLASH_ATTN_EXT) | 1.040 | 1.040 | +0.000 |
| output layer | 3.154 | 3.151 | -0.003 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 7.432 | 7.441 | +0.009 |
| other weights | 0.917 | 0.921 | +0.004 |
| sum of the parts | 86.349 | 84.059 | -2.290 |

### 2. By product, all cards (the label without types and shapes)

| Product | `c` types | `ptx` types | `c` groups | `ptx` groups | `c` ms | `ptx` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 19.203 | 16.936 | -2.267 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.165 | 0.213 | +0.048 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.764 | 0.806 | +0.042 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.215 | 0.178 | -0.037 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.930 | 8.906 | -0.024 |
| `MUL_MAT ssm_beta` | bf16 | bf16 | 36.00 | 36.00 | 0.325 | 0.348 | +0.022 |
| the other 145 products (each under 0.02 ms apart) | | | | | 56.746 | 56.671 | -0.075 |
| **the pass (sum of the labels)** | | | | | **86.349** | **84.058** | **-2.291** |

### 3. The labels of the products above

| Label | `c` groups | `ptx` groups | `c` ms | `ptx` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 0.662 | 0.537 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 3.744 | 3.036 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 2.391 | 1.982 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 6.123 | 5.105 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.283 | 6.276 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.165 | 0.213 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.764 | 0.806 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.215 | 0.178 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.930 | 8.906 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 36.00 | 36.00 | 0.325 | 0.348 |

## Request d8192p2

Kept verify passes: `c` 92, `ptx` 92.

### 1. By part (R41's parts), all cards

| Part | `c` | `ptx` | `ptx` − `c` |
|---|---:|---:|---:|
| routed experts | 26.604 | 24.349 | -2.255 |
| shared expert | 4.221 | 4.227 | +0.006 |
| hyper-connections (BF16) | 12.823 | 12.805 | -0.018 |
| router | 1.272 | 1.301 | +0.029 |
| attention and recurrent projections | 20.368 | 20.278 | -0.090 |
| Gated DeltaNet state and conv | 8.977 | 9.022 | +0.045 |
| attention (FLASH_ATTN_EXT) | 12.614 | 12.618 | +0.004 |
| output layer | 3.156 | 3.173 | +0.017 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 8.744 | 8.729 | -0.015 |
| other weights | 0.971 | 0.946 | -0.025 |
| sum of the parts | 99.750 | 97.449 | -2.301 |

### 2. By product, all cards (the label without types and shapes)

| Product | `c` types | `ptx` types | `c` groups | `ptx` groups | `c` ms | `ptx` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 19.304 | 17.041 | -2.262 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.352 | 0.260 | -0.092 |
| `MUL_MAT ssm_beta` | bf16 | bf16 | 36.00 | 36.00 | 0.423 | 0.358 | -0.066 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.215 | 0.276 | +0.061 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.735 | 0.787 | +0.052 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.977 | 9.022 | +0.045 |
| `MUL_MAT hc_attn_up` | bf16 | bf16 | 48.00 | 48.00 | 2.622 | 2.593 | -0.029 |
| `MUL_MAT ffn_gate_inp` | bf16 | bf16 | 48.00 | 48.00 | 1.272 | 1.301 | +0.028 |
| `REPEAT node` | - | - | 1.00 | 1.00 | 0.036 | 0.008 | -0.028 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.086 | 0.113 | +0.027 |
| `MUL_MAT ssm_alpha` | bf16 | bf16 | 36.00 | 36.00 | 0.295 | 0.272 | -0.022 |
| the other 140 products (each under 0.02 ms apart) | | | | | 65.433 | 65.417 | -0.016 |
| **the pass (sum of the labels)** | | | | | **99.750** | **97.449** | **-2.301** |

### 3. The labels of the products above

| Label | `c` groups | `ptx` groups | `c` ms | `ptx` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 0.662 | 0.543 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 3.685 | 2.997 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 2.412 | 2.000 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 6.139 | 5.093 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.406 | 6.408 |
| `MUL gate` | 36.00 | 36.00 | 0.352 | 0.260 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 36.00 | 36.00 | 0.423 | 0.358 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.215 | 0.276 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.735 | 0.787 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.977 | 9.022 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 48.00 | 48.00 | 2.622 | 2.593 |
| `MUL_MAT ffn_gate_inp:bf16:2560x512` | 48.00 | 48.00 | 1.272 | 1.301 |
| `REPEAT node` | 1.00 | 1.00 | 0.036 | 0.008 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.086 | 0.113 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 36.00 | 36.00 | 0.295 | 0.272 |

