## 1. The step, ms (request p2: the repeat), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 53.69, 53.71 | 69.22, 69.21 | 53.72, 53.73 | 52.17, 52.18 | 1.53, 1.53 | — | — | — |
| 1 | 8,192 | — | 64.33, 64.34 | 80.32, 80.43 | 64.41, 64.42 | 62.72, 62.71 | 1.69, 1.70 | — | — | — |
| 2 | 0 | — | 107.80 | 128.67 | 108.38 | 96.59 | 3.84 | 7.98 | 6.56 | 0.70 |
| 2 | 8,192 | — | 122.62 | 145.41 | 123.60 | 109.61 | 4.07 | 9.95 | 8.40 | 0.95 |
| 3 | 0 | — | 142.69 | 164.35 | 143.96 | 125.91 | 4.65 | 13.40 | 11.59 | 0.78 |
| 3 | 8,192 | — | 166.12 | 190.36 | 167.91 | 145.72 | 4.99 | 17.23 | 14.79 | 1.12 |
| 4 | 0 | — | 172.49 | 194.84 | 176.00 | 151.69 | 5.48 | 18.64 | 16.21 | 0.83 |
| 4 | 8,192 | — | 199.33 | 224.83 | 202.27 | 172.67 | 5.90 | 23.72 | 20.41 | 1.25 |

## 2. Width 1: the verify pass's GPU time by part, ms a pass, all nine cards (w1g1, w1g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 17.50, 17.46 | 17.52, 17.53 | 0.03, 0.06 |
| shared expert | 3.03, 3.04 | 3.03, 3.08 | 0.00, 0.04 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.18, 1.19 | 1.18, 1.18 | 0.00, -0.01 |
| attention and recurrent projections | 18.80, 18.80 | 18.64, 18.74 | -0.15, -0.06 |
| Gated DeltaNet state and conv | 3.71, 3.67 | 3.73, 3.69 | 0.02, 0.02 |
| attention (FLASH_ATTN_EXT) | 0.56, 0.56 | 10.79, 10.80 | 10.22, 10.23 |
| output layer | 3.01, 3.01 | 3.01, 3.02 | 0.00, 0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 4.06, 4.13 | 4.16, 4.11 | 0.09, -0.02 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 51.86, 51.87 | 62.07, 62.13 | 10.21, 10.26 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 3.15, 3.15 | 3.20, 3.18 |
| cards 0 to 7: routed experts | 16.11, 16.08 | 16.15, 16.14 |
| cards 0 to 7: shared expert, router, hyper-connections | 3.86, 3.87 | 3.87, 3.90 |
| cards 0 to 7: everything else | 21.49, 21.47 | 30.72, 30.77 |
| card 8: the head's product | 3.01, 3.01 | 3.01, 3.02 |
| card 8: the sampling chain after the product | 0.21, 0.21 | 0.21, 0.21 |
| card 8: GATED_DELTA_NET | 0.33, 0.31 | 0.29, 0.31 |
| card 8: routed experts | 1.39, 1.38 | 1.38, 1.39 |
| card 8: shared expert, router, hyper-connections | 0.35, 0.36 | 0.35, 0.36 |
| card 8: everything else | 1.96, 2.02 | 2.89, 2.86 |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 30.43 | 30.78 | 0.34 |
| shared expert | 5.16 | 5.16 | -0.00 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 1.48 | 1.50 | 0.01 |
| attention and recurrent projections | 37.38 | 37.38 | -0.01 |
| Gated DeltaNet state and conv | 6.42 | 6.43 | 0.00 |
| attention (FLASH_ATTN_EXT) | 0.76 | 13.29 | 12.53 |
| output layer | 5.55 | 5.56 | 0.01 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 8.74 | 8.85 | 0.11 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 95.94 | 108.93 | 12.99 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.89 | 5.90 |
| cards 0 to 7: routed experts | 28.68 | 28.99 |
| cards 0 to 7: shared expert, router, hyper-connections | 6.22 | 6.22 |
| cards 0 to 7: everything else | 43.70 | 55.29 |
| card 8: the head's product | 5.55 | 5.56 |
| card 8: the sampling chain after the product | 0.40 | 0.41 |
| card 8: GATED_DELTA_NET | 0.34 | 0.34 |
| card 8: routed experts | 1.75 | 1.79 |
| card 8: shared expert, router, hyper-connections | 0.43 | 0.43 |
| card 8: everything else | 2.98 | 4.00 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 43.62 | 44.23 | 0.61 |
| shared expert | 6.49 | 6.49 | -0.00 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 2.06 | 2.07 | 0.01 |
| attention and recurrent projections | 46.01 | 46.13 | 0.12 |
| Gated DeltaNet state and conv | 9.22 | 9.23 | 0.00 |
| attention (FLASH_ATTN_EXT) | 1.17 | 20.23 | 19.06 |
| output layer | 7.02 | 7.08 | 0.06 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 9.53 | 9.53 | -0.00 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 125.14 | 144.99 | 19.86 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 8.49 | 8.51 |
| cards 0 to 7: routed experts | 41.12 | 41.68 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.02 | 8.02 |
| cards 0 to 7: everything else | 52.73 | 70.25 |
| card 8: the head's product | 7.02 | 7.08 |
| card 8: the sampling chain after the product | 0.63 | 0.64 |
| card 8: GATED_DELTA_NET | 0.50 | 0.50 |
| card 8: routed experts | 2.50 | 2.56 |
| card 8: shared expert, router, hyper-connections | 0.54 | 0.54 |
| card 8: everything else | 3.59 | 5.21 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 55.62 | 57.41 | 1.79 |
| shared expert | 8.30 | 8.44 | 0.14 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 1.48 | 1.48 | 0.00 |
| attention and recurrent projections | 53.46 | 53.67 | 0.21 |
| Gated DeltaNet state and conv | 11.90 | 11.95 | 0.05 |
| attention (FLASH_ATTN_EXT) | 1.17 | 20.27 | 19.10 |
| output layer | 8.48 | 8.56 | 0.07 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 9.88 | 9.77 | -0.11 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 150.29 | 171.55 | 21.25 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 11.04 | 11.08 |
| cards 0 to 7: routed experts | 52.46 | 54.10 |
| cards 0 to 7: shared expert, router, hyper-connections | 9.17 | 9.30 |
| cards 0 to 7: everything else | 59.76 | 77.33 |
| card 8: the head's product | 8.48 | 8.56 |
| card 8: the sampling chain after the product | 0.86 | 0.87 |
| card 8: GATED_DELTA_NET | 0.65 | 0.65 |
| card 8: routed experts | 3.17 | 3.31 |
| card 8: shared expert, router, hyper-connections | 0.61 | 0.61 |
| card 8: everything else | 4.10 | 5.72 |

## 4a. Width 1: what grows from depth 0 to 8,192, ms a pass (each load: w1g1, w1g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.56, 0.56 | 10.79, 10.80 | 10.22, 10.23 |
| `MUL_MAT ssm_beta:q5_0:3072x64` | 0.54, 0.53 | 0.39, 0.47 | -0.15, -0.06 |
| `SIGMOID beta_sigmoid` | 0.10, 0.15 | 0.23, 0.21 | 0.13, 0.06 |
| `SOFTPLUS+MUL a_softplus` | 0.28, 0.31 | 0.25, 0.25 | -0.03, -0.06 |
| `RMS_NORM+MUL norm` | 0.52, 0.53 | 0.49, 0.48 | -0.03, -0.04 |
| `GATED_DELTA_NET+CPY node` | 3.47, 3.46 | 3.50, 3.49 | 0.02, 0.03 |
| `SILU+MUL node` | 0.09, 0.09 | 0.06, 0.06 | -0.02, -0.02 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82, 3.81 | 3.84, 3.83 | 0.02, 0.02 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81, 5.81 | 5.83, 5.83 | 0.01, 0.02 |
| `CPY cache_r_l6_(view)_(copy_of_conv_input-6_(view))` | 0.02, 0.02 | 0.04, 0.04 | 0.01, 0.02 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.72, 5.72 | 5.70, 5.72 | -0.02, -0.01 |
| `CPY cache_r_l18_(view)_(copy_of_conv_input-18_(view))` | 0.02, 0.02 | 0.04, 0.03 | 0.02, 0.01 |
| every label (the pass) | 51.86, 51.87 | 62.07, 62.13 | 10.21, 10.26 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.76 | 13.29 | 12.53 |
| `RMS_NORM+SCALE node` | 0.19 | 0.36 | 0.17 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.10 | 9.23 | 0.14 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.67 | 7.79 | 0.12 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.81 | 0.72 | -0.09 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.37 | 3.42 | 0.05 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.30 | 10.33 | 0.03 |
| `GET_ROWS node` | 0.45 | 0.42 | -0.03 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.07 | 10.10 | 0.03 |
| `SCALE node` | 0.33 | 0.31 | -0.02 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.56 | 4.59 | 0.02 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.50 | 10.53 | 0.02 |
| every label (the pass) | 95.94 | 108.93 | 12.99 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.16 | 20.23 | 19.06 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 10.76 | 11.06 | 0.29 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.11 | 13.33 | 0.22 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.44 | 12.53 | 0.08 |
| `MUL_MAT output:q5_0:3072x248320` | 7.02 | 7.08 | 0.06 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 15.00 | 15.06 | 0.06 |
| `SCALE node` | 0.19 | 0.14 | -0.05 |
| `ADD attn_residual` | 0.27 | 0.32 | 0.05 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.75 | 4.79 | 0.04 |
| `SWIGLU ffn_swiglu` | 0.12 | 0.08 | -0.04 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 13.09 | 13.13 | 0.04 |
| `RMS_NORM+SCALE node` | 0.25 | 0.21 | -0.04 |
| every label (the pass) | 125.13 | 144.99 | 19.86 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.17 | 20.27 | 19.10 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 16.69 | 17.31 | 0.63 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 13.60 | 14.19 | 0.58 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 19.43 | 19.77 | 0.34 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.90 | 6.14 | 0.24 |
| `SIGMOID shared_expert_gate_sigmoid` | 0.23 | 0.07 | -0.16 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.71 | 2.84 | 0.13 |
| `SWIGLU ffn_swiglu` | 0.17 | 0.29 | 0.13 |
| `MUL ffn_shexp_gated` | 0.22 | 0.11 | -0.11 |
| `MUL_MAT output:q5_0:3072x248320` | 8.48 | 8.56 | 0.07 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.62 | 0.68 | 0.07 |
| `RMS_NORM+MUL norm` | 0.58 | 0.64 | 0.06 |
| every label (the pass) | 150.29 | 171.55 | 21.25 |

## 4b. Depth 0: what grows from 1 token to 2, ms a pass (loads paired in order: w1g1 → w2g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 10.07 | 10.07 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.72 | 10.50 | 4.79 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 10.30 | 4.09 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82 | 7.67 | 3.85 |
| `CONCAT conv_input` | 0.48 | 4.01 | 3.53 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81 | 9.10 | 3.28 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 3.38 | 3.26 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.87 | 7.03 | 3.16 |
| `GATED_DELTA_NET+CPY node` | 3.47 | 6.23 | 2.76 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 5.55 | 2.54 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.44 | 4.56 | 2.12 |
| every label (the pass) | 51.86 | 95.94 | 44.08 |

## 4b. Depth 8,192: what grows from 1 token to 2, ms a pass (loads paired in order: w1g1 → w2g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 10.10 | 10.10 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.70 | 10.53 | 4.83 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 10.33 | 4.12 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.84 | 7.79 | 3.96 |
| `CONCAT conv_input` | 0.47 | 4.03 | 3.56 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.83 | 9.23 | 3.41 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 3.37 | 3.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.88 | 7.04 | 3.17 |
| `GATED_DELTA_NET+CPY node` | 3.50 | 6.24 | 2.75 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 5.56 | 2.54 |
| `FLASH_ATTN_EXT node` | 10.79 | 13.29 | 2.50 |
| every label (the pass) | 62.07 | 108.93 | 46.86 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.30 | 15.00 | 4.70 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.10 | 13.11 | 4.02 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.67 | 10.76 | 3.09 |
| `GATED_DELTA_NET+CPY node` | 6.23 | 8.99 | 2.76 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.50 | 13.09 | 2.59 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.07 | 12.44 | 2.37 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.03 | 8.83 | 1.80 |
| `MUL_MAT output:q5_0:3072x248320` | 5.55 | 7.02 | 1.47 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.37 | 4.75 | 1.38 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.56 | 5.74 | 1.17 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.38 | 4.16 | 0.79 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 1.48 | 2.06 | 0.58 |
| every label (the pass) | 95.94 | 125.13 | 29.20 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 13.29 | 20.23 | 6.94 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.33 | 15.06 | 4.73 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.23 | 13.33 | 4.10 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.79 | 11.06 | 3.26 |
| `GATED_DELTA_NET+CPY node` | 6.24 | 9.01 | 2.77 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.53 | 13.13 | 2.61 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.10 | 12.53 | 2.43 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.04 | 8.85 | 1.81 |
| `MUL_MAT output:q5_0:3072x248320` | 5.56 | 7.08 | 1.52 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.42 | 4.79 | 1.37 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.59 | 5.75 | 1.17 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.37 | 4.15 | 0.78 |
| every label (the pass) | 108.93 | 144.99 | 36.07 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 15.00 | 19.43 | 4.43 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.11 | 16.69 | 3.57 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 10.76 | 13.60 | 2.84 |
| `GATED_DELTA_NET+CPY node` | 8.99 | 11.69 | 2.70 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 13.09 | 15.77 | 2.68 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 8.83 | 10.58 | 1.75 |
| `MUL_MAT output:q5_0:3072x248320` | 7.02 | 8.48 | 1.46 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.44 | 13.63 | 1.18 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.74 | 6.91 | 1.18 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.75 | 5.90 | 1.16 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 2.06 | 1.48 | -0.58 |
| `MUL_MAT ffn_gate_inp_shexp:f32:3072x1` | 0.25 | 0.84 | 0.58 |
| every label (the pass) | 125.13 | 150.29 | 25.16 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 15.06 | 19.77 | 4.72 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.33 | 17.31 | 3.98 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 11.06 | 14.19 | 3.13 |
| `GATED_DELTA_NET+CPY node` | 9.01 | 11.73 | 2.72 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 13.13 | 15.83 | 2.70 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 8.85 | 10.61 | 1.76 |
| `MUL_MAT output:q5_0:3072x248320` | 7.08 | 8.56 | 1.48 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.79 | 6.14 | 1.35 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.75 | 6.93 | 1.18 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.53 | 13.68 | 1.15 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 2.07 | 1.48 | -0.59 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.25 | 2.84 | 0.59 |
| every label (the pass) | 144.99 | 171.55 | 26.55 |

## 4b. Depth 0: what grows from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 13.63 | 13.63 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 19.43 | 13.23 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81 | 16.69 | 10.87 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.72 | 15.77 | 10.06 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82 | 13.60 | 9.78 |
| `GATED_DELTA_NET+CPY node` | 3.47 | 11.69 | 8.22 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.87 | 10.58 | 6.70 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 8.48 | 5.47 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.44 | 6.91 | 4.47 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 4.58 | 4.46 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 1.65 | 5.90 | 4.25 |
| every label (the pass) | 51.86 | 150.29 | 98.43 |

## 4b. Depth 8,192: what grows from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 13.68 | 13.68 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 19.77 | 13.56 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.83 | 17.31 | 11.49 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.84 | 14.19 | 10.35 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.70 | 15.83 | 10.13 |
| `FLASH_ATTN_EXT node` | 10.79 | 20.27 | 9.48 |
| `GATED_DELTA_NET+CPY node` | 3.50 | 11.73 | 8.24 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.88 | 10.61 | 6.73 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 8.56 | 5.54 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 1.64 | 6.14 | 4.50 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.45 | 6.93 | 4.48 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 4.57 | 4.45 |
| every label (the pass) | 62.07 | 171.55 | 109.47 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p2; each load by value)

| family | 1 token, depth 0 | 1 token, depth 8,192 | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 12.02, 12.00 | 12.04, 12.04 | 19.40 | 19.56 | 28.11 | 28.39 | 36.12 | 37.09 |
| routed experts: down | 5.48, 5.46 | 5.48, 5.49 | 11.04 | 11.21 | 15.51 | 15.84 | 19.51 | 20.33 |
| shared expert: gate, up and GLU | 1.73, 1.74 | 1.73, 1.75 | 3.20 | 3.21 | 3.99 | 3.99 | 4.75 | 4.78 |
| shared expert: down | 1.15, 1.16 | 1.15, 1.17 | 1.76 | 1.77 | 2.25 | 2.25 | 2.71 | 2.84 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.15, 0.14 | 0.14, 0.16 | 0.20 | 0.18 | 0.25 | 0.26 | 0.84 | 0.82 |
| router (ffn_gate_inp) and its top-k | 1.61, 1.61 | 1.60, 1.58 | 1.86 | 1.86 | 2.45 | 2.45 | 1.83 | 1.83 |
| attention layers: q, k, v, output products | 4.22, 4.22 | 4.23, 4.23 | 8.68 | 8.70 | 10.73 | 10.73 | 12.42 | 12.43 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 14.58, 14.57 | 14.42, 14.51 | 28.71 | 28.68 | 35.29 | 35.40 | 41.04 | 41.24 |
| recurrent layers: GATED_DELTA_NET | 3.47, 3.46 | 3.50, 3.49 | 6.23 | 6.24 | 8.99 | 9.01 | 11.69 | 11.73 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.24, 0.21 | 0.24, 0.20 | 0.19 | 0.18 | 0.24 | 0.21 | 0.21 | 0.22 |
| attention: FLASH_ATTN_EXT | 0.56, 0.56 | 10.79, 10.80 | 0.76 | 13.29 | 1.16 | 20.23 | 1.17 | 20.27 |
| output head | 3.01, 3.01 | 3.01, 3.02 | 5.55 | 5.56 | 7.02 | 7.08 | 8.48 | 8.56 |
| everything else (norms, adds, copies, the sampling chain) | 3.64, 3.71 | 3.74, 3.71 | 8.37 | 8.49 | 9.15 | 9.15 | 9.52 | 9.41 |
| **the pass** | 51.86, 51.87 | 62.07, 62.13 | 95.94 | 108.93 | 125.13 | 144.99 | 150.29 | 171.55 |

## 6. The one-token step's host part, ms, median of the request's steps (host_split.py; w1h1, w1h2; request p2)

| part | depth 0 | depth 8,192 |
|---|---:|---:|
| the step (trace) | 53.72, 53.73 | 64.41, 64.42 |
| host, before the inputs are set | 0.13, 0.13 | 0.19, 0.19 |
| from the inputs set to the token on the host | 53.39, 53.41 | 64.03, 64.02 |
|   of it, the graphs queued | 2.30, 2.31 | 3.14, 3.13 |
|   the wait for the last card | 51.09, 51.09 | 60.86, 60.86 |
| the nine cards' GPU time (host profile) | 52.17, 52.18 | 62.72, 62.71 |
| the cards' part less their GPU time: hops, launch latency, gaps (est.) | 1.22, 1.23 | 1.31, 1.31 |
| host, after the token | 0.18, 0.18 | 0.19, 0.19 |

