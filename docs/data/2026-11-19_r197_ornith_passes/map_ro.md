## 1. The step, ms (request p2: the repeat), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 45.63 | — | 45.60 | 44.05 | 1.55 | — | — | — |
| 1 | 8,192 | — | 55.08 | — | 55.17 | 53.46 | 1.71 | — | — | — |
| 2 | 0 | — | — | 116.16, 116.35 | — | — | — | — | — | — |
| 2 | 8,192 | — | — | 131.63, 131.84 | — | — | — | — | — | — |
| 3 | 0 | — | — | 146.60, 146.66 | — | — | — | — | — | — |
| 3 | 8,192 | — | — | 170.33, 170.03 | — | — | — | — | — | — |
| 4 | 0 | — | — | 182.90, 182.78 | — | — | — | — | — | — |
| 4 | 8,192 | — | — | 207.57, 207.73 | — | — | — | — | — | — |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1, w2g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 26.64, 26.67 | 26.79, 26.80 | 0.15, 0.13 |
| shared expert | 4.65, 4.64 | 4.66, 4.65 | 0.01, 0.01 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.35, 1.36 | 1.35, 1.34 | 0.00, -0.01 |
| attention and recurrent projections | 30.51, 30.50 | 30.51, 30.51 | 0.01, 0.01 |
| Gated DeltaNet state and conv | 5.74, 5.76 | 5.75, 5.76 | 0.01, 0.00 |
| attention (FLASH_ATTN_EXT) | 0.70, 0.70 | 11.83, 11.85 | 11.13, 11.15 |
| output layer | 5.06, 5.04 | 5.07, 5.06 | 0.01, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.34, 8.40 | 8.35, 8.44 | 0.01, 0.04 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 82.97, 83.08 | 94.30, 94.41 | 11.32, 11.33 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.26, 5.27 | 5.28, 5.28 |
| cards 0 to 7: routed experts | 25.22, 25.26 | 25.36, 25.37 |
| cards 0 to 7: shared expert, router, hyper-connections | 5.61, 5.62 | 5.63, 5.61 |
| cards 0 to 7: everything else | 36.82, 36.89 | 47.03, 47.14 |
| card 8: the head's product | 5.06, 5.04 | 5.07, 5.06 |
| card 8: the sampling chain after the product | 0.37, 0.37 | 0.38, 0.38 |
| card 8: GATED_DELTA_NET | 0.31, 0.31 | 0.31, 0.31 |
| card 8: routed experts | 1.42, 1.41 | 1.43, 1.43 |
| card 8: shared expert, router, hyper-connections | 0.38, 0.38 | 0.38, 0.38 |
| card 8: everything else | 2.53, 2.52 | 3.45, 3.44 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1, w3g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 37.56, 37.59 | 37.98, 37.98 | 0.42, 0.40 |
| shared expert | 5.53, 5.53 | 5.49, 5.49 | -0.04, -0.03 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.91, 1.91 | 1.94, 1.92 | 0.03, 0.01 |
| attention and recurrent projections | 38.21, 38.20 | 38.32, 38.32 | 0.11, 0.12 |
| Gated DeltaNet state and conv | 8.23, 8.24 | 8.25, 8.25 | 0.01, 0.01 |
| attention (FLASH_ATTN_EXT) | 1.05, 1.05 | 17.84, 17.86 | 16.79, 16.80 |
| output layer | 5.98, 5.96 | 5.99, 6.00 | 0.02, 0.03 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.88, 8.94 | 8.88, 8.89 | -0.00, -0.04 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 107.36, 107.42 | 124.68, 124.72 | 17.33, 17.29 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 7.58, 7.59 | 7.60, 7.61 |
| cards 0 to 7: routed experts | 35.58, 35.61 | 35.95, 35.96 |
| cards 0 to 7: shared expert, router, hyper-connections | 6.97, 6.97 | 6.96, 6.94 |
| cards 0 to 7: everything else | 44.69, 44.76 | 60.18, 60.22 |
| card 8: the head's product | 5.98, 5.96 | 5.99, 6.00 |
| card 8: the sampling chain after the product | 0.58, 0.58 | 0.58, 0.58 |
| card 8: GATED_DELTA_NET | 0.45, 0.45 | 0.45, 0.45 |
| card 8: routed experts | 1.98, 1.98 | 2.03, 2.03 |
| card 8: shared expert, router, hyper-connections | 0.47, 0.47 | 0.47, 0.47 |
| card 8: everything else | 3.08, 3.07 | 4.47, 4.47 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 48.75, 48.72 | 48.59, 48.60 | -0.16, -0.12 |
| shared expert | 7.80, 7.76 | 7.77, 7.78 | -0.02, 0.01 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.37, 1.38 | 1.37, 1.38 | 0.00, -0.01 |
| attention and recurrent projections | 50.70, 50.63 | 50.74, 50.74 | 0.04, 0.11 |
| Gated DeltaNet state and conv | 10.65, 10.64 | 10.66, 10.66 | 0.01, 0.03 |
| attention (FLASH_ATTN_EXT) | 1.09, 1.09 | 17.88, 17.88 | 16.79, 16.79 |
| output layer | 8.38, 8.37 | 8.40, 8.40 | 0.02, 0.04 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 9.08, 9.08 | 9.11, 9.11 | 0.02, 0.02 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 137.82, 137.67 | 154.53, 154.55 | 16.71, 16.88 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 9.88, 9.87 | 9.89, 9.89 |
| cards 0 to 7: routed experts | 46.09, 46.06 | 46.02, 46.02 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.61, 8.58 | 8.58, 8.59 |
| cards 0 to 7: everything else | 56.38, 56.31 | 71.84, 71.84 |
| card 8: the head's product | 8.38, 8.37 | 8.40, 8.40 |
| card 8: the sampling chain after the product | 0.78, 0.79 | 0.78, 0.79 |
| card 8: GATED_DELTA_NET | 0.58, 0.58 | 0.58, 0.58 |
| card 8: routed experts | 2.66, 2.66 | 2.58, 2.57 |
| card 8: shared expert, router, hyper-connections | 0.56, 0.56 | 0.56, 0.56 |
| card 8: everything else | 3.90, 3.89 | 5.30, 5.29 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1, w2g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.70, 0.70 | 11.83, 11.85 | 11.13, 11.15 |
| `RMS_NORM+SCALE node` | 0.47, 0.51 | 0.61, 0.66 | 0.13, 0.16 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.64, 0.62 | 0.57, 0.54 | -0.07, -0.08 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 4.88, 4.88 | 0.08, 0.08 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 5.08, 5.09 | 0.08, 0.07 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.83 | 6.90, 6.89 | 0.07, 0.07 |
| `SCALE node` | 0.41, 0.43 | 0.36, 0.36 | -0.05, -0.07 |
| `RMS_NORM+MUL norm` | 0.58, 0.58 | 0.56, 0.53 | -0.03, -0.05 |
| `SOFT_MAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV ffn_moe_probs` | 0.36, 0.38 | 0.35, 0.34 | -0.02, -0.05 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.66, 3.67 | 3.63, 3.63 | -0.03, -0.03 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.91, 2.92 | 2.88, 2.89 | -0.03, -0.03 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.54, 9.55 | 9.57, 9.58 | 0.03, 0.02 |
| every label (the pass) | 82.98, 83.08 | 94.30, 94.41 | 11.32, 11.33 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1, w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.05, 1.05 | 17.84, 17.86 | 16.79, 16.80 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.02, 7.03 | 7.18, 7.18 | 0.15, 0.15 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.85, 6.86 | 7.00, 7.00 | 0.15, 0.14 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.67, 9.67 | 9.81, 9.80 | 0.14, 0.13 |
| `RMS_NORM+MUL norm` | 0.88, 0.90 | 0.97, 0.94 | 0.09, 0.05 |
| `SCALE node` | 0.22, 0.26 | 0.19, 0.20 | -0.03, -0.06 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.94, 9.94 | 9.98, 9.99 | 0.04, 0.06 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.15, 11.16 | 11.20, 11.19 | 0.05, 0.03 |
| `SILU+MUL node` | 0.21, 0.23 | 0.24, 0.26 | 0.03, 0.04 |
| `ADD attn_residual` | 0.25, 0.23 | 0.28, 0.26 | 0.03, 0.03 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 1.85, 1.85 | 1.82, 1.82 | -0.03, -0.03 |
| `CPY s_slot_rev` | 0.09, 0.09 | 0.06, 0.07 | -0.03, -0.02 |
| every label (the pass) | 107.36, 107.42 | 124.68, 124.72 | 17.33, 17.30 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.09, 1.09 | 17.88, 17.88 | 16.79, 16.79 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.53, 6.54 | 6.37, 6.36 | -0.17, -0.17 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.37, 6.36 | 6.20, 6.20 | -0.16, -0.16 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 8.85, 8.84 | 8.97, 8.98 | 0.13, 0.14 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 9.01, 9.00 | 9.13, 9.14 | 0.13, 0.14 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.45, 5.45 | 5.32, 5.32 | -0.13, -0.13 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.49, 2.47 | 2.37, 2.39 | -0.12, -0.07 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.47, 2.47 | 2.58, 2.56 | 0.10, 0.09 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.39, 0.38 | 0.45, 0.44 | 0.05, 0.06 |
| `SWIGLU ffn_swiglu` | 0.22, 0.25 | 0.17, 0.19 | -0.05, -0.06 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.54, 12.54 | 12.59, 12.60 | 0.05, 0.06 |
| `MUL_MAT ffn_gate_inp_shexp:f32:3072x1` | 0.73, 0.71 | 0.67, 0.67 | -0.06, -0.04 |
| every label (the pass) | 137.82, 137.67 | 154.53, 154.55 | 16.71, 16.88 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.71, 6.72 | 9.94, 9.94 | 3.23, 3.22 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.83 | 9.67, 9.67 | 2.85, 2.85 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.58 | 8.03, 8.04 | 2.46, 2.45 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 6.85, 6.86 | 2.05, 2.06 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 7.02, 7.03 | 2.02, 2.02 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.54, 9.55 | 11.15, 11.16 | 1.61, 1.60 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.44, 3.44 | 4.85, 4.85 | 1.41, 1.41 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.66, 3.67 | 5.03, 5.03 | 1.37, 1.36 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.91, 2.92 | 4.14, 4.14 | 1.23, 1.23 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.39, 6.40 | 7.51, 7.52 | 1.12, 1.12 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 2.34, 2.34 | 3.37, 3.37 | 1.03, 1.03 |
| `MUL_MAT output:q5_0:3072x248320` | 5.06, 5.04 | 5.98, 5.96 | 0.92, 0.92 |
| every label (the pass) | 82.98, 83.08 | 107.36, 107.42 | 24.38, 24.35 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.83, 11.85 | 17.84, 17.86 | 6.01, 6.01 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.72, 6.73 | 9.98, 9.99 | 3.26, 3.27 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.90, 6.89 | 9.81, 9.80 | 2.91, 2.91 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.59 | 8.05, 8.06 | 2.46, 2.47 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.88, 4.88 | 7.00, 7.00 | 2.12, 2.12 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.08, 5.09 | 7.18, 7.18 | 2.09, 2.09 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.57, 9.58 | 11.20, 11.19 | 1.62, 1.61 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.42, 3.42 | 4.84, 4.84 | 1.42, 1.43 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 5.02, 5.02 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.89 | 4.13, 4.14 | 1.25, 1.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.42, 6.42 | 7.54, 7.54 | 1.13, 1.12 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 2.33, 2.33 | 3.38, 3.38 | 1.05, 1.05 |
| every label (the pass) | 94.30, 94.41 | 124.68, 124.72 | 30.39, 30.31 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.15, 11.16 | 15.58, 15.56 | 4.43, 4.40 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.51, 7.52 | 10.47, 10.46 | 2.96, 2.94 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.67, 9.67 | 12.54, 12.54 | 2.87, 2.86 |
| `GATED_DELTA_NET+CPY node` | 8.03, 8.04 | 10.45, 10.44 | 2.43, 2.41 |
| `MUL_MAT output:q5_0:3072x248320` | 5.98, 5.96 | 8.38, 8.37 | 2.41, 2.41 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.94, 9.94 | 11.98, 11.97 | 2.04, 2.03 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.85, 6.86 | 8.85, 8.84 | 2.00, 1.98 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.02, 7.03 | 9.01, 9.00 | 1.99, 1.97 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.87, 4.87 | 6.82, 6.81 | 1.95, 1.94 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.85, 4.85 | 6.37, 6.36 | 1.52, 1.51 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.03, 5.03 | 6.53, 6.54 | 1.51, 1.51 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.14, 4.14 | 5.45, 5.45 | 1.31, 1.31 |
| every label (the pass) | 107.36, 107.42 | 137.82, 137.67 | 30.46, 30.24 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.20, 11.19 | 15.60, 15.60 | 4.41, 4.42 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.54, 7.54 | 10.48, 10.48 | 2.94, 2.94 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.81, 9.80 | 12.59, 12.60 | 2.78, 2.79 |
| `GATED_DELTA_NET+CPY node` | 8.05, 8.06 | 10.47, 10.47 | 2.42, 2.41 |
| `MUL_MAT output:q5_0:3072x248320` | 5.99, 6.00 | 8.40, 8.40 | 2.41, 2.41 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.98, 9.99 | 12.00, 12.00 | 2.02, 2.00 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 7.00, 7.00 | 8.97, 8.98 | 1.98, 1.98 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.18, 7.18 | 9.13, 9.14 | 1.96, 1.96 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.89, 4.89 | 6.84, 6.84 | 1.95, 1.95 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.84, 4.84 | 6.20, 6.20 | 1.36, 1.35 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.02 | 6.37, 6.36 | 1.35, 1.34 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.13, 4.14 | 5.32, 5.32 | 1.19, 1.18 |
| every label (the pass) | 124.68, 124.72 | 154.53, 154.55 | 29.85, 29.83 |

## 4b. Depth 0: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.54, 9.55 | 15.58, 15.56 | 6.03, 6.00 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.83 | 12.54, 12.54 | 5.72, 5.71 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.71, 6.72 | 11.98, 11.97 | 5.27, 5.25 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.58 | 10.45, 10.44 | 4.88, 4.86 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.39, 6.40 | 10.47, 10.46 | 4.08, 4.06 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 8.85, 8.84 | 4.05, 4.03 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 9.01, 9.00 | 4.00, 3.99 |
| `MUL_MAT output:q5_0:3072x248320` | 5.06, 5.04 | 8.38, 8.37 | 3.32, 3.32 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.44, 3.44 | 6.37, 6.36 | 2.93, 2.92 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.66, 3.67 | 6.53, 6.54 | 2.88, 2.87 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.14, 4.15 | 6.82, 6.81 | 2.68, 2.67 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.91, 2.92 | 5.45, 5.45 | 2.54, 2.53 |
| every label (the pass) | 82.98, 83.08 | 137.82, 137.67 | 54.85, 54.59 |

## 4b. Depth 8,192: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.83, 11.85 | 17.88, 17.88 | 6.05, 6.03 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.57, 9.58 | 15.60, 15.60 | 6.03, 6.03 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.90, 6.89 | 12.59, 12.60 | 5.69, 5.70 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.72, 6.73 | 12.00, 12.00 | 5.28, 5.27 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.59 | 10.47, 10.47 | 4.88, 4.88 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.88, 4.88 | 8.97, 8.98 | 4.10, 4.10 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.42, 6.42 | 10.48, 10.48 | 4.07, 4.06 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.08, 5.09 | 9.13, 9.14 | 4.05, 4.06 |
| `MUL_MAT output:q5_0:3072x248320` | 5.06, 5.06 | 8.40, 8.40 | 3.34, 3.34 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.42, 3.42 | 6.20, 6.20 | 2.79, 2.78 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 6.37, 6.36 | 2.74, 2.73 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.15, 4.15 | 6.84, 6.84 | 2.69, 2.69 |
| every label (the pass) | 94.30, 94.41 | 154.53, 154.55 | 60.23, 60.14 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p2; each load by value)

| family | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 16.90, 16.92 | 17.01, 17.01 | 23.75, 23.77 | 24.04, 24.04 | 30.76, 30.73 | 30.68, 30.68 |
| routed experts: down | 9.74, 9.75 | 9.78, 9.78 | 13.81, 13.82 | 13.94, 13.94 | 17.99, 17.99 | 17.91, 17.92 |
| shared expert: gate, up and GLU | 2.87, 2.85 | 2.88, 2.87 | 3.35, 3.35 | 3.32, 3.32 | 4.58, 4.59 | 4.73, 4.72 |
| shared expert: down | 1.60, 1.62 | 1.61, 1.61 | 1.95, 1.94 | 1.95, 1.95 | 2.49, 2.47 | 2.37, 2.39 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.17, 0.17 | 0.16, 0.16 | 0.23, 0.23 | 0.22, 0.23 | 0.73, 0.71 | 0.67, 0.67 |
| router (ffn_gate_inp) and its top-k | 1.71, 1.74 | 1.70, 1.68 | 2.26, 2.26 | 2.29, 2.27 | 1.70, 1.71 | 1.70, 1.70 |
| attention layers: q, k, v, output products | 6.97, 6.97 | 6.97, 6.99 | 8.79, 8.79 | 8.81, 8.82 | 11.61, 11.60 | 11.61, 11.62 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 23.54, 23.53 | 23.54, 23.52 | 29.42, 29.41 | 29.51, 29.51 | 39.09, 39.03 | 39.13, 39.13 |
| recurrent layers: GATED_DELTA_NET | 5.57, 5.58 | 5.59, 5.59 | 8.03, 8.04 | 8.05, 8.06 | 10.45, 10.44 | 10.47, 10.47 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.17, 0.18 | 0.16, 0.17 | 0.21, 0.21 | 0.20, 0.19 | 0.19, 0.19 | 0.19, 0.19 |
| attention: FLASH_ATTN_EXT | 0.70, 0.70 | 11.83, 11.85 | 1.05, 1.05 | 17.84, 17.86 | 1.09, 1.09 | 17.88, 17.88 |
| output head | 5.06, 5.04 | 5.06, 5.06 | 5.98, 5.96 | 5.99, 6.00 | 8.38, 8.37 | 8.40, 8.40 |
| everything else (norms, adds, copies, the sampling chain) | 7.97, 8.02 | 8.00, 8.10 | 8.53, 8.59 | 8.53, 8.55 | 8.75, 8.75 | 8.78, 8.79 |
| **the pass** | 82.98, 83.08 | 94.30, 94.41 | 107.36, 107.42 | 124.68, 124.72 | 137.82, 137.67 | 154.53, 154.55 |

## 6. The one-token step's host part, ms, median of the request's steps (host_split.py; w1h1; request p2)

| part | depth 0 | depth 8,192 |
|---|---:|---:|
| the step (trace) | 45.60 | 55.17 |
| host, before the inputs are set | 0.13 | 0.19 |
| from the inputs set to the token on the host | 45.27 | 54.78 |
|   of it, the graphs queued | 2.46 | 3.23 |
|   the wait for the last card | 42.80 | 51.52 |
| the nine cards' GPU time (host profile) | 44.05 | 53.46 |
| the cards' part less their GPU time: hops, launch latency, gaps (est.) | 1.22 | 1.32 |
| host, after the token | 0.19 | 0.20 |

