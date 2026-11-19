## 1. The step, ms (request p1: the first request at the depth), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 46.79 | — | 45.61 | 44.02 | 1.56 | — | — | — |
| 1 | 8,192 | — | 55.30 | — | 55.16 | 53.45 | 1.71 | — | — | — |
| 2 | 0 | — | — | 117.86, 117.81 | — | — | — | — | — | — |
| 2 | 8,192 | — | — | 131.46, 131.96 | — | — | — | — | — | — |
| 3 | 0 | — | — | 148.44, 148.18 | — | — | — | — | — | — |
| 3 | 8,192 | — | — | 170.10, 169.98 | — | — | — | — | — | — |
| 4 | 0 | — | — | 183.82, 184.11 | — | — | — | — | — | — |
| 4 | 8,192 | — | — | 207.63, 207.28 | — | — | — | — | — | — |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1, w2g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 26.61, 26.62 | 26.77, 26.78 | 0.16, 0.15 |
| shared expert | 4.63, 4.63 | 4.65, 4.65 | 0.02, 0.02 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.35, 1.36 | 1.35, 1.34 | 0.00, -0.02 |
| attention and recurrent projections | 30.48, 30.44 | 30.48, 30.48 | 0.00, 0.04 |
| Gated DeltaNet state and conv | 5.74, 5.75 | 5.74, 5.75 | 0.00, 0.00 |
| attention (FLASH_ATTN_EXT) | 0.70, 0.70 | 11.82, 11.84 | 11.12, 11.13 |
| output layer | 5.04, 5.04 | 5.06, 5.06 | 0.02, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.32, 8.40 | 8.35, 8.45 | 0.03, 0.05 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 82.86, 82.95 | 94.22, 94.35 | 11.36, 11.40 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.26, 5.26 | 5.27, 5.28 |
| cards 0 to 7: routed experts | 25.20, 25.21 | 25.34, 25.35 |
| cards 0 to 7: shared expert, router, hyper-connections | 5.60, 5.61 | 5.62, 5.62 |
| cards 0 to 7: everything else | 36.77, 36.84 | 46.98, 47.11 |
| card 8: the head's product | 5.04, 5.04 | 5.06, 5.06 |
| card 8: the sampling chain after the product | 0.37, 0.37 | 0.37, 0.38 |
| card 8: GATED_DELTA_NET | 0.31, 0.31 | 0.31, 0.31 |
| card 8: routed experts | 1.41, 1.41 | 1.43, 1.43 |
| card 8: shared expert, router, hyper-connections | 0.38, 0.38 | 0.38, 0.38 |
| card 8: everything else | 2.52, 2.52 | 3.45, 3.44 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1, w3g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 37.52, 37.52 | 37.93, 37.97 | 0.40, 0.45 |
| shared expert | 5.52, 5.52 | 5.48, 5.49 | -0.04, -0.03 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.90, 1.91 | 1.93, 1.92 | 0.03, 0.01 |
| attention and recurrent projections | 38.13, 38.14 | 38.27, 38.30 | 0.13, 0.16 |
| Gated DeltaNet state and conv | 8.23, 8.22 | 8.25, 8.25 | 0.02, 0.03 |
| attention (FLASH_ATTN_EXT) | 1.05, 1.05 | 17.82, 17.84 | 16.77, 16.79 |
| output layer | 5.96, 5.96 | 5.98, 5.98 | 0.03, 0.03 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.90, 8.90 | 8.89, 8.89 | -0.01, -0.01 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 107.21, 107.22 | 124.54, 124.65 | 17.33, 17.43 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 7.58, 7.58 | 7.60, 7.60 |
| cards 0 to 7: routed experts | 35.55, 35.55 | 35.90, 35.94 |
| cards 0 to 7: shared expert, router, hyper-connections | 6.96, 6.96 | 6.94, 6.94 |
| cards 0 to 7: everything else | 44.64, 44.65 | 60.12, 60.19 |
| card 8: the head's product | 5.96, 5.96 | 5.98, 5.98 |
| card 8: the sampling chain after the product | 0.58, 0.58 | 0.58, 0.58 |
| card 8: GATED_DELTA_NET | 0.45, 0.45 | 0.45, 0.45 |
| card 8: routed experts | 1.98, 1.97 | 2.02, 2.02 |
| card 8: shared expert, router, hyper-connections | 0.47, 0.47 | 0.47, 0.47 |
| card 8: everything else | 3.06, 3.06 | 4.46, 4.47 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 48.72, 48.65 | 48.58, 48.62 | -0.14, -0.03 |
| shared expert | 7.79, 7.75 | 7.80, 7.79 | 0.01, 0.04 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.37, 1.38 | 1.38, 1.37 | 0.00, -0.00 |
| attention and recurrent projections | 50.60, 50.51 | 50.74, 50.77 | 0.14, 0.27 |
| Gated DeltaNet state and conv | 10.64, 10.61 | 10.66, 10.67 | 0.02, 0.05 |
| attention (FLASH_ATTN_EXT) | 1.08, 1.08 | 17.87, 17.89 | 16.78, 16.81 |
| output layer | 8.36, 8.37 | 8.40, 8.39 | 0.04, 0.03 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 9.04, 9.09 | 9.07, 9.06 | 0.02, -0.03 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 137.60, 137.44 | 154.47, 154.57 | 16.87, 17.13 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 9.87, 9.85 | 9.88, 9.89 |
| cards 0 to 7: routed experts | 46.06, 45.99 | 46.01, 46.05 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.60, 8.57 | 8.61, 8.60 |
| cards 0 to 7: everything else | 56.26, 56.18 | 71.81, 71.84 |
| card 8: the head's product | 8.36, 8.37 | 8.40, 8.39 |
| card 8: the sampling chain after the product | 0.78, 0.79 | 0.77, 0.79 |
| card 8: GATED_DELTA_NET | 0.58, 0.58 | 0.58, 0.58 |
| card 8: routed experts | 2.66, 2.66 | 2.57, 2.57 |
| card 8: shared expert, router, hyper-connections | 0.56, 0.56 | 0.56, 0.56 |
| card 8: everything else | 3.88, 3.89 | 5.28, 5.29 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1, w2g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.70, 0.70 | 11.82, 11.84 | 11.12, 11.13 |
| `RMS_NORM+SCALE node` | 0.47, 0.52 | 0.64, 0.67 | 0.17, 0.14 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.65, 0.61 | 0.54, 0.52 | -0.11, -0.09 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 5.08, 5.08 | 0.08, 0.08 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 4.87, 4.87 | 0.07, 0.08 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.82 | 6.89, 6.89 | 0.07, 0.07 |
| `RMS_NORM+MUL norm` | 0.60, 0.60 | 0.54, 0.52 | -0.06, -0.08 |
| `SCALE node` | 0.40, 0.41 | 0.36, 0.38 | -0.04, -0.03 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.53, 9.53 | 9.57, 9.57 | 0.04, 0.03 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.38, 6.39 | 6.41, 6.42 | 0.03, 0.03 |
| `MUL_MAT attn_k:q8_0:3072x512` | 0.22, 0.21 | 0.24, 0.24 | 0.02, 0.03 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.43, 3.44 | 3.41, 3.41 | -0.02, -0.03 |
| every label (the pass) | 82.86, 82.95 | 94.22, 94.35 | 11.36, 11.40 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1, w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.05, 1.05 | 17.82, 17.84 | 16.77, 16.79 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.02, 7.02 | 7.17, 7.17 | 0.15, 0.16 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.84, 6.84 | 6.99, 6.99 | 0.14, 0.15 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.66, 9.66 | 9.80, 9.80 | 0.13, 0.15 |
| `ADD attn_residual` | 0.22, 0.22 | 0.29, 0.28 | 0.07, 0.07 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.92, 9.92 | 9.97, 10.00 | 0.04, 0.09 |
| `RMS_NORM+MUL norm` | 0.88, 0.90 | 0.96, 0.93 | 0.07, 0.04 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.13, 11.13 | 11.17, 11.18 | 0.04, 0.05 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.50, 7.50 | 7.53, 7.53 | 0.03, 0.03 |
| `SCALE node` | 0.23, 0.25 | 0.21, 0.21 | -0.02, -0.04 |
| `MUL_MAT output:q5_0:3072x248320` | 5.96, 5.95 | 5.98, 5.98 | 0.03, 0.03 |
| `GATED_DELTA_NET+CPY node` | 8.02, 8.02 | 8.05, 8.05 | 0.02, 0.03 |
| every label (the pass) | 107.21, 107.22 | 124.54, 124.65 | 17.32, 17.43 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.08, 1.08 | 17.87, 17.89 | 16.78, 16.80 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.54, 6.53 | 6.36, 6.36 | -0.17, -0.17 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.36, 6.36 | 6.20, 6.20 | -0.16, -0.16 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 8.83, 8.82 | 8.97, 8.99 | 0.14, 0.17 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 9.00, 8.99 | 9.14, 9.15 | 0.13, 0.16 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.44, 5.44 | 5.32, 5.32 | -0.13, -0.12 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.47, 2.47 | 2.59, 2.55 | 0.12, 0.08 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.49, 2.48 | 2.37, 2.40 | -0.11, -0.08 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.54, 12.52 | 12.59, 12.61 | 0.05, 0.09 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 15.55, 15.53 | 15.60, 15.61 | 0.05, 0.08 |
| `MUL_MAT ffn_up_shexp:q5_0:3072x1024` | 2.10, 2.11 | 2.16, 2.18 | 0.06, 0.07 |
| `SWIGLU ffn_swiglu` | 0.22, 0.26 | 0.18, 0.18 | -0.04, -0.08 |
| every label (the pass) | 137.60, 137.44 | 154.47, 154.57 | 16.87, 17.13 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.70, 6.71 | 9.92, 9.92 | 3.22, 3.21 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.82 | 9.66, 9.66 | 2.84, 2.84 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.57 | 8.02, 8.02 | 2.45, 2.45 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 6.84, 6.84 | 2.05, 2.04 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 7.02, 7.02 | 2.01, 2.01 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.53, 9.53 | 11.13, 11.13 | 1.60, 1.60 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.43, 3.44 | 4.84, 4.85 | 1.41, 1.41 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.65, 3.65 | 5.02, 5.02 | 1.38, 1.37 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.90, 2.91 | 4.13, 4.14 | 1.23, 1.22 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.38, 6.39 | 7.50, 7.50 | 1.11, 1.11 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 2.33, 2.33 | 3.37, 3.37 | 1.04, 1.04 |
| `MUL_MAT output:q5_0:3072x248320` | 5.04, 5.04 | 5.96, 5.95 | 0.92, 0.91 |
| every label (the pass) | 82.86, 82.95 | 107.21, 107.22 | 24.35, 24.27 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.82, 11.84 | 17.82, 17.84 | 5.99, 6.01 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.72, 6.72 | 9.97, 10.00 | 3.25, 3.28 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.89, 6.89 | 9.80, 9.80 | 2.90, 2.91 |
| `GATED_DELTA_NET+CPY node` | 5.58, 5.58 | 8.05, 8.05 | 2.46, 2.47 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.87, 4.87 | 6.99, 6.99 | 2.11, 2.12 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.08, 5.08 | 7.17, 7.17 | 2.08, 2.09 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.57, 9.57 | 11.17, 11.18 | 1.61, 1.62 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.41, 3.41 | 4.84, 4.84 | 1.42, 1.43 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 5.02, 5.02 | 1.39, 1.40 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.89 | 4.12, 4.13 | 1.25, 1.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.41, 6.42 | 7.53, 7.53 | 1.12, 1.12 |
| `MUL_MAT attn_output:q5_0:8192x3072` | 2.32, 2.33 | 3.37, 3.38 | 1.05, 1.05 |
| every label (the pass) | 94.22, 94.35 | 124.54, 124.65 | 30.32, 30.30 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.13, 11.13 | 15.55, 15.53 | 4.42, 4.40 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.50, 7.50 | 10.45, 10.43 | 2.95, 2.93 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.66, 9.66 | 12.54, 12.52 | 2.87, 2.86 |
| `GATED_DELTA_NET+CPY node` | 8.02, 8.02 | 10.45, 10.43 | 2.42, 2.41 |
| `MUL_MAT output:q5_0:3072x248320` | 5.96, 5.95 | 8.36, 8.37 | 2.40, 2.41 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.92, 9.92 | 11.96, 11.94 | 2.04, 2.02 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.84, 6.84 | 8.83, 8.82 | 1.99, 1.98 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.02, 7.02 | 9.00, 8.99 | 1.98, 1.97 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.86, 4.86 | 6.81, 6.80 | 1.95, 1.94 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.84, 4.85 | 6.36, 6.36 | 1.52, 1.51 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.02 | 6.54, 6.53 | 1.51, 1.51 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.13, 4.14 | 5.44, 5.44 | 1.31, 1.31 |
| every label (the pass) | 107.21, 107.22 | 137.60, 137.44 | 30.39, 30.22 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.17, 11.18 | 15.60, 15.61 | 4.43, 4.43 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.53, 7.53 | 10.48, 10.49 | 2.95, 2.95 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.80, 9.80 | 12.59, 12.61 | 2.79, 2.80 |
| `GATED_DELTA_NET+CPY node` | 8.05, 8.05 | 10.47, 10.47 | 2.42, 2.42 |
| `MUL_MAT output:q5_0:3072x248320` | 5.98, 5.98 | 8.40, 8.39 | 2.41, 2.41 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.97, 10.00 | 12.00, 12.01 | 2.03, 2.00 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.99, 6.99 | 8.97, 8.99 | 1.99, 2.00 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.17, 7.17 | 9.14, 9.15 | 1.97, 1.97 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.88, 4.89 | 6.84, 6.84 | 1.96, 1.96 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.84, 4.84 | 6.20, 6.20 | 1.36, 1.36 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.02 | 6.36, 6.36 | 1.35, 1.34 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.12, 4.13 | 5.32, 5.32 | 1.19, 1.18 |
| every label (the pass) | 124.54, 124.65 | 154.47, 154.57 | 29.94, 29.92 |

## 4b. Depth 0: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.53, 9.53 | 15.55, 15.53 | 6.02, 5.99 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.82, 6.82 | 12.54, 12.52 | 5.72, 5.70 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.70, 6.71 | 11.96, 11.94 | 5.26, 5.23 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.57 | 10.45, 10.43 | 4.88, 4.86 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.38, 6.39 | 10.45, 10.43 | 4.07, 4.04 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.80, 4.80 | 8.83, 8.82 | 4.04, 4.02 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.01, 5.01 | 9.00, 8.99 | 3.99, 3.98 |
| `MUL_MAT output:q5_0:3072x248320` | 5.04, 5.04 | 8.36, 8.37 | 3.32, 3.33 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.43, 3.44 | 6.36, 6.36 | 2.93, 2.92 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.65, 3.65 | 6.54, 6.53 | 2.89, 2.88 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.14, 4.14 | 6.81, 6.80 | 2.67, 2.66 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.90, 2.91 | 5.44, 5.44 | 2.54, 2.53 |
| every label (the pass) | 82.86, 82.95 | 137.60, 137.44 | 54.74, 54.49 |

## 4b. Depth 8,192: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.82, 11.84 | 17.87, 17.89 | 6.04, 6.05 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.57, 9.57 | 15.60, 15.61 | 6.03, 6.04 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.89, 6.89 | 12.59, 12.61 | 5.70, 5.71 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 6.72, 6.72 | 12.00, 12.01 | 5.28, 5.28 |
| `GATED_DELTA_NET+CPY node` | 5.58, 5.58 | 10.47, 10.47 | 4.88, 4.89 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.87, 4.87 | 8.97, 8.99 | 4.10, 4.11 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.41, 6.42 | 10.48, 10.49 | 4.07, 4.07 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.08, 5.08 | 9.14, 9.15 | 4.05, 4.06 |
| `MUL_MAT output:q5_0:3072x248320` | 5.06, 5.06 | 8.40, 8.39 | 3.33, 3.33 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.41, 3.41 | 6.20, 6.20 | 2.79, 2.78 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 6.36, 6.36 | 2.74, 2.73 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.15, 4.15 | 6.84, 6.84 | 2.69, 2.69 |
| every label (the pass) | 94.22, 94.35 | 154.47, 154.57 | 60.25, 60.22 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p1; each load by value)

| family | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 16.89, 16.89 | 17.00, 17.00 | 23.73, 23.73 | 24.01, 24.03 | 30.74, 30.69 | 30.67, 30.69 |
| routed experts: down | 9.72, 9.73 | 9.77, 9.78 | 13.79, 13.79 | 13.92, 13.94 | 17.98, 17.96 | 17.91, 17.93 |
| shared expert: gate, up and GLU | 2.87, 2.86 | 2.89, 2.88 | 3.34, 3.34 | 3.32, 3.32 | 4.57, 4.58 | 4.75, 4.73 |
| shared expert: down | 1.60, 1.60 | 1.61, 1.61 | 1.95, 1.94 | 1.94, 1.95 | 2.49, 2.48 | 2.37, 2.40 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.16, 0.16 | 0.16, 0.16 | 0.23, 0.24 | 0.22, 0.23 | 0.73, 0.70 | 0.67, 0.67 |
| router (ffn_gate_inp) and its top-k | 1.71, 1.72 | 1.71, 1.69 | 2.25, 2.26 | 2.28, 2.27 | 1.70, 1.70 | 1.70, 1.70 |
| attention layers: q, k, v, output products | 6.95, 6.95 | 6.98, 6.99 | 8.78, 8.77 | 8.80, 8.81 | 11.59, 11.57 | 11.61, 11.62 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 23.52, 23.49 | 23.50, 23.48 | 29.35, 29.37 | 29.46, 29.49 | 39.02, 38.93 | 39.13, 39.15 |
| recurrent layers: GATED_DELTA_NET | 5.57, 5.57 | 5.58, 5.58 | 8.02, 8.02 | 8.05, 8.05 | 10.45, 10.43 | 10.47, 10.47 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.17, 0.18 | 0.16, 0.17 | 0.21, 0.20 | 0.20, 0.19 | 0.19, 0.19 | 0.19, 0.19 |
| attention: FLASH_ATTN_EXT | 0.70, 0.70 | 11.82, 11.84 | 1.05, 1.05 | 17.82, 17.84 | 1.08, 1.08 | 17.87, 17.89 |
| output head | 5.04, 5.04 | 5.06, 5.06 | 5.96, 5.95 | 5.98, 5.98 | 8.36, 8.37 | 8.40, 8.39 |
| everything else (norms, adds, copies, the sampling chain) | 7.96, 8.05 | 7.99, 8.11 | 8.55, 8.55 | 8.54, 8.55 | 8.72, 8.76 | 8.74, 8.74 |
| **the pass** | 82.86, 82.95 | 94.22, 94.35 | 107.21, 107.22 | 124.54, 124.65 | 137.60, 137.44 | 154.47, 154.57 |

## 6. The one-token step's host part, ms, median of the request's steps (host_split.py; w1h1; request p1)

| part | depth 0 | depth 8,192 |
|---|---:|---:|
| the step (trace) | 45.63 | 55.16 |
| host, before the inputs are set | 0.13 | 0.19 |
| from the inputs set to the token on the host | 45.26 | 54.77 |
|   of it, the graphs queued | 2.50 | 3.23 |
|   the wait for the last card | 42.74 | 51.50 |
| the nine cards' GPU time (host profile) | 44.02 | 53.45 |
| the cards' part less their GPU time: hops, launch latency, gaps (est.) | 1.24 | 1.31 |
| host, after the token | 0.20 | 0.20 |

