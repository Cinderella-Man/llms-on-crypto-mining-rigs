# The verify pass by part and product: `unset` against `ptx`

ms a verify pass (four tokens), the kept passes of the request's typical steps (verify_parts.py), the profiler's cost subtracted per span; one load per file.

## Request d0p2

Kept verify passes: `unset` 100, `ptx` 100.

### 1. By part (R41's parts), all cards

| Part | `unset` | `ptx` | `ptx` − `unset` |
|---|---:|---:|---:|
| routed experts | 35.902 | 24.094 | -11.808 |
| shared expert | 4.192 | 4.189 | -0.003 |
| hyper-connections (BF16) | 12.825 | 12.810 | -0.015 |
| router | 1.299 | 1.297 | -0.002 |
| attention and recurrent projections | 20.274 | 20.209 | -0.065 |
| Gated DeltaNet state and conv | 8.942 | 8.906 | -0.036 |
| attention (FLASH_ATTN_EXT) | 1.040 | 1.040 | +0.000 |
| output layer | 3.154 | 3.151 | -0.003 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 7.397 | 7.441 | +0.044 |
| other weights | 0.913 | 0.921 | +0.008 |
| sum of the parts | 95.938 | 84.059 | -11.879 |

### 2. By product, all cards (the label without types and shapes)

| Product | `unset` types | `ptx` types | `unset` groups | `ptx` groups | `unset` ms | `ptx` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 28.739 | 16.936 | -11.803 |
| `SIGMOID beta_sigmoid` | - | - | 36.00 | 36.00 | 0.266 | 0.178 | -0.088 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.365 | 0.452 | +0.087 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.163 | 0.213 | +0.049 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 5.308 | 5.259 | -0.049 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.769 | 0.806 | +0.037 |
| `GATED_DELTA_NET+CPY node` | - | - | 36.00 | 36.00 | 8.942 | 8.906 | -0.036 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | - | - | 96.00 | 96.00 | 1.282 | 1.260 | -0.022 |
| `RMS_NORM+MUL node` | - | - | 100.00 | 100.00 | 0.561 | 0.541 | -0.020 |
| the other 142 products (each under 0.02 ms apart) | | | | | 49.543 | 49.506 | -0.036 |
| **the pass (sum of the labels)** | | | | | **95.939** | **84.058** | **-11.881** |

### 3. The labels of the products above

| Label | `unset` groups | `ptx` groups | `unset` ms | `ptx` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 1.098 | 0.537 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 6.889 | 3.036 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 4.151 | 1.982 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 10.317 | 5.105 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.283 | 6.276 |
| `SIGMOID beta_sigmoid` | 36.00 | 36.00 | 0.266 | 0.178 |
| `MUL gate` | 36.00 | 36.00 | 0.365 | 0.452 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.163 | 0.213 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 17.00 | 17.00 | 1.776 | 1.775 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 6.00 | 6.00 | 0.840 | 0.842 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 5.00 | 5.00 | 0.879 | 0.879 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 8.00 | 8.00 | 1.814 | 1.764 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.769 | 0.806 |
| `GATED_DELTA_NET+CPY node` | 36.00 | 36.00 | 8.942 | 8.906 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 96.00 | 96.00 | 1.282 | 1.260 |
| `RMS_NORM+MUL node` | 100.00 | 100.00 | 0.561 | 0.541 |

## Request d8192p2

Kept verify passes: `unset` 92, `ptx` 92.

### 1. By part (R41's parts), all cards

| Part | `unset` | `ptx` | `ptx` − `unset` |
|---|---:|---:|---:|
| routed experts | 36.085 | 24.349 | -11.736 |
| shared expert | 4.201 | 4.227 | +0.026 |
| hyper-connections (BF16) | 12.810 | 12.805 | -0.005 |
| router | 1.296 | 1.301 | +0.005 |
| attention and recurrent projections | 20.398 | 20.278 | -0.120 |
| Gated DeltaNet state and conv | 9.002 | 9.022 | +0.020 |
| attention (FLASH_ATTN_EXT) | 12.619 | 12.618 | -0.001 |
| output layer | 3.155 | 3.173 | +0.018 |
| token embeddings | 0.000 | 0.000 | +0.000 |
| other ops without weights | 8.722 | 8.729 | +0.007 |
| other weights | 0.909 | 0.946 | +0.037 |
| sum of the parts | 109.196 | 97.449 | -11.747 |

### 2. By product, all cards (the label without types and shapes)

| Product | `unset` types | `ptx` types | `unset` groups | `ptx` groups | `unset` ms | `ptx` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps` | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 48.00 | 48.00 | 28.783 | 17.041 | -11.742 |
| `MUL_MAT ssm_beta` | bf16 | bf16 | 36.00 | 36.00 | 0.405 | 0.358 | -0.047 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 36.00 | 36.00 | 5.322 | 5.276 | -0.046 |
| `MUL_MAT indexer.k_proj` | bf16 | bf16 | 12.00 | 12.00 | 0.195 | 0.234 | +0.039 |
| `RMS_NORM+MUL node` | - | - | 100.00 | 100.00 | 0.538 | 0.576 | +0.038 |
| `CPY kpool_k_idxs` | - | - | 9.00 | 9.00 | 0.060 | 0.026 | -0.034 |
| `CONCAT node` | - | - | 13.00 | 13.00 | 0.182 | 0.155 | -0.027 |
| `MUL gate` | - | - | 36.00 | 36.00 | 0.236 | 0.260 | +0.025 |
| `GDN_CONV+CPY gdn_conv` | - | - | 36.00 | 36.00 | 0.764 | 0.787 | +0.024 |
| `SOFTPLUS a_softplus` | - | - | 36.00 | 36.00 | 0.090 | 0.113 | +0.023 |
| `SET_ROWS kpool_dev_idxs` | - | - | 9.00 | 9.00 | 0.027 | 0.049 | +0.022 |
| `MUL_MAT hc_attn_up` | bf16 | bf16 | 48.00 | 48.00 | 2.572 | 2.593 | +0.020 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp ffn_up_shexp` | iq4_xs, q2_0, q4_K, q5_K | iq4_xs, q2_0, q4_K, q5_K | 43.00 | 43.00 | 2.038 | 2.058 | +0.020 |
| the other 138 products (each under 0.02 ms apart) | | | | | 67.983 | 67.921 | -0.062 |
| **the pass (sum of the labels)** | | | | | **109.196** | **97.449** | **-11.748** |

### 3. The labels of the products above

| Label | `unset` groups | `ptx` groups | `unset` ms | `ptx` ms |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq1_m:2560x640 ffn_up_exps:iq1_m:2560x640` | 2.00 | 2.00 | 1.104 | 0.543 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_s:2560x640 ffn_up_exps:iq2_s:2560x640` | 7.00 | 7.00 | 6.769 | 2.997 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xs:2560x640 ffn_up_exps:iq2_xs:2560x640` | 4.00 | 4.00 | 4.200 | 2.000 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:iq2_xxs:2560x640 ffn_up_exps:iq2_xxs:2560x640` | 11.00 | 11.00 | 10.297 | 5.093 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 24.00 | 24.00 | 6.414 | 6.408 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 36.00 | 36.00 | 0.405 | 0.358 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 17.00 | 17.00 | 1.778 | 1.780 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 6.00 | 6.00 | 0.852 | 0.842 |
| `MUL_MAT ssm_out:q5_K:6144x2560` | 5.00 | 5.00 | 0.879 | 0.881 |
| `MUL_MAT ssm_out:q6_K:6144x2560` | 8.00 | 8.00 | 1.813 | 1.773 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 12.00 | 12.00 | 0.195 | 0.234 |
| `RMS_NORM+MUL node` | 100.00 | 100.00 | 0.538 | 0.576 |
| `CPY kpool_k_idxs` | 9.00 | 9.00 | 0.060 | 0.026 |
| `CONCAT node` | 13.00 | 13.00 | 0.182 | 0.155 |
| `MUL gate` | 36.00 | 36.00 | 0.236 | 0.260 |
| `GDN_CONV+CPY gdn_conv` | 36.00 | 36.00 | 0.764 | 0.787 |
| `SOFTPLUS a_softplus` | 36.00 | 36.00 | 0.090 | 0.113 |
| `SET_ROWS kpool_dev_idxs` | 9.00 | 9.00 | 0.027 | 0.049 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 48.00 | 48.00 | 2.572 | 2.593 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:iq4_xs:2560x640 ffn_up_shexp:iq4_xs:2560x640` | 22.00 | 22.00 | 1.224 | 1.226 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 4.00 | 4.00 | 0.141 | 0.148 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q4_K:2560x640 ffn_up_shexp:q4_K:2560x640` | 13.00 | 13.00 | 0.500 | 0.508 |
| `MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp:q5_K:2560x640 ffn_up_shexp:q5_K:2560x640` | 4.00 | 4.00 | 0.173 | 0.176 |

