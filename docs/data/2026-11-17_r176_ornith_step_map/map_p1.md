## 1. The step, ms (request p1: the first request at the depth), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 54.84, 55.35 | 70.28, 70.38 | 53.73, 53.74 | 52.15, 52.16 | 1.56, 1.56 | — | — | — |
| 1 | 8,192 | — | 64.49, 64.52 | 80.30, 80.35 | 64.42, 64.44 | 62.71, 62.73 | 1.70, 1.70 | — | — | — |
| 2 | 0 | — | 110.54 | 130.90 | 108.77 | 96.57 | 3.84 | 8.03 | 6.55 | 0.71 |
| 2 | 8,192 | — | 123.61 | 145.19 | 123.58 | 109.60 | 4.06 | 9.94 | 8.39 | 0.95 |
| 3 | 0 | — | 147.78 | 167.37 | 145.15 | 125.80 | 4.65 | 13.51 | 11.60 | 0.79 |
| 3 | 8,192 | — | 168.13 | 190.02 | 167.93 | 145.78 | 4.99 | 17.22 | 14.79 | 1.12 |
| 4 | 0 | — | 178.37 | 196.94 | 176.40 | 151.59 | 5.49 | 18.68 | 16.22 | 0.84 |
| 4 | 8,192 | — | 201.85 | 224.70 | 202.17 | 172.57 | 5.89 | 23.73 | 20.42 | 1.26 |

## 2. Width 1: the verify pass's GPU time by part, ms a pass, all nine cards (w1g1, w1g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 17.48, 17.46 | 17.54, 17.52 | 0.05, 0.07 |
| shared expert | 3.01, 3.02 | 3.02, 3.07 | 0.01, 0.05 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.19, 1.20 | 1.18, 1.18 | -0.01, -0.02 |
| attention and recurrent projections | 18.77, 18.81 | 18.63, 18.71 | -0.14, -0.09 |
| Gated DeltaNet state and conv | 3.70, 3.64 | 3.73, 3.69 | 0.03, 0.05 |
| attention (FLASH_ATTN_EXT) | 0.56, 0.56 | 10.78, 10.79 | 10.22, 10.22 |
| output layer | 3.01, 3.01 | 3.01, 3.01 | 0.00, 0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 4.10, 4.15 | 4.17, 4.10 | 0.07, -0.05 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 51.84, 51.85 | 62.06, 62.08 | 10.23, 10.23 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 3.14, 3.15 | 3.19, 3.20 |
| cards 0 to 7: routed experts | 16.10, 16.08 | 16.16, 16.14 |
| cards 0 to 7: shared expert, router, hyper-connections | 3.85, 3.87 | 3.85, 3.90 |
| cards 0 to 7: everything else | 21.50, 21.48 | 30.73, 30.72 |
| card 8: the head's product | 3.01, 3.01 | 3.01, 3.01 |
| card 8: the sampling chain after the product | 0.21, 0.21 | 0.21, 0.21 |
| card 8: GATED_DELTA_NET | 0.32, 0.30 | 0.30, 0.31 |
| card 8: routed experts | 1.39, 1.38 | 1.38, 1.39 |
| card 8: shared expert, router, hyper-connections | 0.35, 0.35 | 0.35, 0.35 |
| card 8: everything else | 1.97, 2.03 | 2.88, 2.85 |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 30.41 | 30.77 | 0.36 |
| shared expert | 5.17 | 5.16 | -0.01 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 1.49 | 1.49 | 0.00 |
| attention and recurrent projections | 37.35 | 37.34 | -0.01 |
| Gated DeltaNet state and conv | 6.42 | 6.42 | 0.01 |
| attention (FLASH_ATTN_EXT) | 0.76 | 13.28 | 12.52 |
| output layer | 5.54 | 5.56 | 0.02 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 8.73 | 8.90 | 0.17 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 95.86 | 108.92 | 13.06 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.88 | 5.90 |
| cards 0 to 7: routed experts | 28.66 | 28.99 |
| cards 0 to 7: shared expert, router, hyper-connections | 6.23 | 6.22 |
| cards 0 to 7: everything else | 43.66 | 55.29 |
| card 8: the head's product | 5.54 | 5.56 |
| card 8: the sampling chain after the product | 0.39 | 0.41 |
| card 8: GATED_DELTA_NET | 0.34 | 0.34 |
| card 8: routed experts | 1.75 | 1.79 |
| card 8: shared expert, router, hyper-connections | 0.43 | 0.43 |
| card 8: everything else | 2.97 | 4.01 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 43.59 | 44.22 | 0.63 |
| shared expert | 6.50 | 6.48 | -0.01 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 2.06 | 2.07 | 0.01 |
| attention and recurrent projections | 46.03 | 46.11 | 0.08 |
| Gated DeltaNet state and conv | 9.21 | 9.23 | 0.02 |
| attention (FLASH_ATTN_EXT) | 1.16 | 20.23 | 19.07 |
| output layer | 7.02 | 7.08 | 0.06 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 9.52 | 9.56 | 0.05 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 125.08 | 144.99 | 19.91 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 8.48 | 8.51 |
| cards 0 to 7: routed experts | 41.09 | 41.67 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.02 | 8.01 |
| cards 0 to 7: everything else | 52.73 | 70.27 |
| card 8: the head's product | 7.02 | 7.08 |
| card 8: the sampling chain after the product | 0.63 | 0.64 |
| card 8: GATED_DELTA_NET | 0.50 | 0.50 |
| card 8: routed experts | 2.50 | 2.56 |
| card 8: shared expert, router, hyper-connections | 0.54 | 0.55 |
| card 8: everything else | 3.59 | 5.21 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 55.58 | 57.41 | 1.84 |
| shared expert | 8.30 | 8.45 | 0.14 |
| hyper-connections (BF16) | 0.00 | 0.00 | 0.00 |
| router | 1.48 | 1.48 | 0.00 |
| attention and recurrent projections | 53.42 | 53.69 | 0.27 |
| Gated DeltaNet state and conv | 11.89 | 11.95 | 0.06 |
| attention (FLASH_ATTN_EXT) | 1.16 | 20.26 | 19.10 |
| output layer | 8.48 | 8.56 | 0.08 |
| token embeddings | 0.00 | 0.00 | 0.00 |
| other ops without weights | 9.88 | 9.75 | -0.13 |
| other weights | 0.00 | 0.00 | 0.00 |
| sum of the parts | 150.18 | 171.54 | 21.36 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 11.03 | 11.08 |
| cards 0 to 7: routed experts | 52.41 | 54.10 |
| cards 0 to 7: shared expert, router, hyper-connections | 9.17 | 9.31 |
| cards 0 to 7: everything else | 59.71 | 77.33 |
| card 8: the head's product | 8.48 | 8.56 |
| card 8: the sampling chain after the product | 0.86 | 0.87 |
| card 8: GATED_DELTA_NET | 0.65 | 0.65 |
| card 8: routed experts | 3.16 | 3.31 |
| card 8: shared expert, router, hyper-connections | 0.61 | 0.61 |
| card 8: everything else | 4.10 | 5.72 |

## 4a. Width 1: what grows from depth 0 to 8,192, ms a pass (each load: w1g1, w1g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.56, 0.56 | 10.78, 10.79 | 10.22, 10.22 |
| `MUL_MAT ssm_beta:q5_0:3072x64` | 0.52, 0.53 | 0.38, 0.45 | -0.14, -0.09 |
| `SIGMOID beta_sigmoid` | 0.10, 0.13 | 0.22, 0.20 | 0.12, 0.08 |
| `SOFTPLUS+MUL a_softplus` | 0.32, 0.34 | 0.26, 0.25 | -0.06, -0.09 |
| `GATED_DELTA_NET+CPY node` | 3.46, 3.44 | 3.49, 3.50 | 0.03, 0.06 |
| `RMS_NORM+MUL norm` | 0.52, 0.52 | 0.48, 0.48 | -0.03, -0.04 |
| `SILU+MUL node` | 0.09, 0.09 | 0.07, 0.06 | -0.02, -0.03 |
| `SOFT_MAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | 0.43, 0.43 | 0.41, 0.40 | -0.01, -0.03 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82, 3.81 | 3.84, 3.83 | 0.02, 0.02 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81, 5.81 | 5.83, 5.82 | 0.02, 0.02 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.20, 6.19 | 6.21, 6.21 | 0.01, 0.02 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 1.19, 1.20 | 1.18, 1.18 | -0.01, -0.02 |
| every label (the pass) | 51.84, 51.85 | 62.06, 62.08 | 10.22, 10.23 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.76 | 13.28 | 12.52 |
| `RMS_NORM+SCALE node` | 0.19 | 0.40 | 0.21 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.09 | 9.23 | 0.15 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.67 | 7.79 | 0.13 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.80 | 0.70 | -0.09 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.37 | 3.42 | 0.05 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.29 | 10.33 | 0.04 |
| `SCALE node` | 0.34 | 0.31 | -0.04 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.50 | 10.52 | 0.03 |
| `RMS_NORM+MUL norm` | 0.65 | 0.68 | 0.03 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.07 | 10.09 | 0.03 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.56 | 4.59 | 0.02 |
| every label (the pass) | 95.86 | 108.92 | 13.06 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.16 | 20.23 | 19.07 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 10.76 | 11.06 | 0.30 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.10 | 13.33 | 0.23 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.45 | 12.52 | 0.07 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 14.99 | 15.05 | 0.06 |
| `MUL_MAT output:q5_0:3072x248320` | 7.02 | 7.08 | 0.06 |
| `RMS_NORM+MUL norm` | 0.87 | 0.93 | 0.06 |
| `ADD attn_residual` | 0.24 | 0.29 | 0.05 |
| `RMS_NORM+SCALE node` | 0.25 | 0.20 | -0.05 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.59 | 0.54 | -0.05 |
| `SCALE node` | 0.19 | 0.15 | -0.05 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.74 | 4.79 | 0.04 |
| every label (the pass) | 125.08 | 144.99 | 19.91 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.16 | 20.26 | 19.10 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 16.68 | 17.31 | 0.63 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 13.59 | 14.19 | 0.60 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 19.41 | 19.77 | 0.36 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.90 | 6.14 | 0.25 |
| `SIGMOID shared_expert_gate_sigmoid` | 0.22 | 0.07 | -0.15 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.71 | 2.82 | 0.11 |
| `SWIGLU ffn_swiglu` | 0.18 | 0.27 | 0.09 |
| `MUL ffn_shexp_gated` | 0.20 | 0.11 | -0.09 |
| `MUL_MAT output:q5_0:3072x248320` | 8.48 | 8.56 | 0.08 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 15.76 | 15.83 | 0.07 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.62 | 0.69 | 0.07 |
| every label (the pass) | 150.18 | 171.54 | 21.36 |

## 4b. Depth 0: what grows from 1 token to 2, ms a pass (loads paired in order: w1g1 → w2g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 10.07 | 10.07 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.72 | 10.50 | 4.78 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.20 | 10.29 | 4.09 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82 | 7.67 | 3.85 |
| `CONCAT conv_input` | 0.47 | 4.01 | 3.54 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81 | 9.09 | 3.27 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 3.37 | 3.26 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.87 | 7.02 | 3.15 |
| `GATED_DELTA_NET+CPY node` | 3.46 | 6.23 | 2.77 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 5.54 | 2.53 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.44 | 4.56 | 2.12 |
| every label (the pass) | 51.84 | 95.86 | 44.03 |

## 4b. Depth 8,192: what grows from 1 token to 2, ms a pass (loads paired in order: w1g1 → w2g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 10.09 | 10.09 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.70 | 10.52 | 4.83 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 10.33 | 4.11 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.84 | 7.79 | 3.95 |
| `CONCAT conv_input` | 0.48 | 4.03 | 3.55 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.83 | 9.23 | 3.40 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 3.37 | 3.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.88 | 7.04 | 3.17 |
| `GATED_DELTA_NET+CPY node` | 3.49 | 6.24 | 2.75 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 5.56 | 2.54 |
| `FLASH_ATTN_EXT node` | 10.78 | 13.28 | 2.50 |
| every label (the pass) | 62.06 | 108.92 | 46.86 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.29 | 14.99 | 4.70 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.09 | 13.10 | 4.01 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.67 | 10.76 | 3.09 |
| `GATED_DELTA_NET+CPY node` | 6.23 | 8.98 | 2.75 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.50 | 13.09 | 2.60 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.07 | 12.45 | 2.38 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.02 | 8.82 | 1.80 |
| `MUL_MAT output:q5_0:3072x248320` | 5.54 | 7.02 | 1.48 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.37 | 4.74 | 1.38 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.56 | 5.73 | 1.17 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.37 | 4.17 | 0.79 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 1.48 | 2.06 | 0.58 |
| every label (the pass) | 95.86 | 125.08 | 29.22 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 13.28 | 20.23 | 6.95 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 10.33 | 15.05 | 4.72 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 9.23 | 13.33 | 4.10 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 7.79 | 11.06 | 3.26 |
| `GATED_DELTA_NET+CPY node` | 6.24 | 9.01 | 2.77 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 10.52 | 13.13 | 2.61 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 10.09 | 12.52 | 2.42 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.04 | 8.84 | 1.80 |
| `MUL_MAT output:q5_0:3072x248320` | 5.56 | 7.08 | 1.52 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 3.42 | 4.79 | 1.37 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.59 | 5.75 | 1.17 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 3.37 | 4.15 | 0.78 |
| every label (the pass) | 108.92 | 144.99 | 36.07 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 14.99 | 19.41 | 4.42 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.10 | 16.68 | 3.58 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 10.76 | 13.59 | 2.83 |
| `GATED_DELTA_NET+CPY node` | 8.98 | 11.68 | 2.70 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 13.09 | 15.76 | 2.67 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 8.82 | 10.57 | 1.75 |
| `MUL_MAT output:q5_0:3072x248320` | 7.02 | 8.48 | 1.46 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.45 | 13.62 | 1.17 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.73 | 6.90 | 1.17 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.74 | 5.90 | 1.15 |
| `MUL_MAT ffn_gate_inp_shexp:f32:3072x1` | 0.25 | 0.86 | 0.61 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 2.06 | 1.48 | -0.59 |
| every label (the pass) | 125.08 | 150.18 | 25.10 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 15.05 | 19.77 | 4.72 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 13.33 | 17.31 | 3.98 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 11.06 | 14.19 | 3.13 |
| `GATED_DELTA_NET+CPY node` | 9.01 | 11.73 | 2.72 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 13.13 | 15.83 | 2.70 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 8.84 | 10.61 | 1.77 |
| `MUL_MAT output:q5_0:3072x248320` | 7.08 | 8.56 | 1.48 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.79 | 6.14 | 1.36 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.75 | 6.93 | 1.18 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 12.52 | 13.67 | 1.15 |
| `MUL_MAT ffn_gate_inp:f32:3072x256` | 2.07 | 1.48 | -0.60 |
| `MUL_MAT ffn_gate_inp_shexp:f32:3072x1` | 0.25 | 0.83 | 0.58 |
| every label (the pass) | 144.99 | 171.54 | 26.55 |

## 4b. Depth 0: what grows from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 13.62 | 13.62 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.20 | 19.41 | 13.21 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.81 | 16.68 | 10.87 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.72 | 15.76 | 10.04 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.82 | 13.59 | 9.77 |
| `GATED_DELTA_NET+CPY node` | 3.46 | 11.68 | 8.22 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.87 | 10.57 | 6.70 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 8.48 | 5.47 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.44 | 6.90 | 4.46 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 4.57 | 4.46 |
| `MUL_MAT+ADD ssm_out:q5_0:8192x3072` | 4.31 | 0.00 | -4.31 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 1.65 | 5.90 | 4.25 |
| every label (the pass) | 51.84 | 150.18 | 98.35 |

## 4b. Depth 8,192: what grows from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 0.00 | 13.67 | 13.67 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n2p:3072x1024 ffn_up_exps:n2p:3072x1024` | 6.21 | 19.77 | 13.56 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:n3p:3072x1024 ffn_up_exps:n3p:3072x1024` | 5.83 | 17.31 | 11.48 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 3.84 | 14.19 | 10.35 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 5.70 | 15.83 | 10.14 |
| `FLASH_ATTN_EXT node` | 10.78 | 20.26 | 9.48 |
| `GATED_DELTA_NET+CPY node` | 3.49 | 11.73 | 8.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 3.88 | 10.61 | 6.74 |
| `MUL_MAT output:q5_0:3072x248320` | 3.01 | 8.56 | 5.54 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 1.65 | 6.14 | 4.49 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 2.45 | 6.93 | 4.48 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 0.12 | 4.57 | 4.45 |
| every label (the pass) | 62.06 | 171.54 | 109.48 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p1; each load by value)

| family | 1 token, depth 0 | 1 token, depth 8,192 | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 12.01, 12.00 | 12.04, 12.03 | 19.38 | 19.56 | 28.09 | 28.38 | 36.09 | 37.08 |
| routed experts: down | 5.47, 5.46 | 5.49, 5.49 | 11.03 | 11.21 | 15.50 | 15.85 | 19.49 | 20.33 |
| shared expert: gate, up and GLU | 1.72, 1.73 | 1.72, 1.75 | 3.21 | 3.21 | 4.00 | 3.98 | 4.73 | 4.80 |
| shared expert: down | 1.15, 1.15 | 1.15, 1.17 | 1.76 | 1.77 | 2.24 | 2.25 | 2.71 | 2.82 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.14, 0.14 | 0.15, 0.16 | 0.20 | 0.18 | 0.25 | 0.25 | 0.86 | 0.83 |
| router (ffn_gate_inp) and its top-k | 1.62, 1.63 | 1.59, 1.58 | 1.85 | 1.86 | 2.45 | 2.46 | 1.83 | 1.83 |
| attention layers: q, k, v, output products | 4.22, 4.23 | 4.23, 4.22 | 8.67 | 8.70 | 10.72 | 10.73 | 12.41 | 12.44 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 14.55, 14.58 | 14.40, 14.49 | 28.68 | 28.64 | 35.31 | 35.38 | 41.01 | 41.25 |
| recurrent layers: GATED_DELTA_NET | 3.46, 3.44 | 3.49, 3.50 | 6.23 | 6.24 | 8.98 | 9.01 | 11.68 | 11.73 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.24, 0.20 | 0.25, 0.19 | 0.19 | 0.18 | 0.23 | 0.22 | 0.21 | 0.22 |
| attention: FLASH_ATTN_EXT | 0.56, 0.56 | 10.78, 10.79 | 0.76 | 13.28 | 1.16 | 20.23 | 1.16 | 20.26 |
| output head | 3.01, 3.01 | 3.01, 3.01 | 5.54 | 5.56 | 7.02 | 7.08 | 8.48 | 8.56 |
| everything else (norms, adds, copies, the sampling chain) | 3.68, 3.72 | 3.76, 3.70 | 8.36 | 8.53 | 9.13 | 9.18 | 9.52 | 9.39 |
| **the pass** | 51.84, 51.85 | 62.06, 62.08 | 95.86 | 108.92 | 125.08 | 144.99 | 150.18 | 171.54 |

## 6. The one-token step's host part, ms, median of the request's steps (host_split.py; w1h1, w1h2; request p1)

| part | depth 0 | depth 8,192 |
|---|---:|---:|
| the step (trace) | 53.76, 53.76 | 64.42, 64.44 |
| host, before the inputs are set | 0.13, 0.13 | 0.19, 0.19 |
| from the inputs set to the token on the host | 53.40, 53.41 | 64.03, 64.05 |
|   of it, the graphs queued | 2.35, 2.36 | 3.14, 3.14 |
|   the wait for the last card | 51.02, 51.03 | 60.86, 60.88 |
| the nine cards' GPU time (host profile) | 52.15, 52.16 | 62.71, 62.73 |
| the cards' part less their GPU time: hops, launch latency, gaps (est.) | 1.24, 1.24 | 1.31, 1.31 |
| host, after the token | 0.19, 0.19 | 0.19, 0.19 |

