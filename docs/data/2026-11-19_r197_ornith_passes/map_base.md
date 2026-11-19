## 1. The step, ms (request p2: the repeat), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 45.63 | — | 45.60 | 44.05 | 1.55 | — | — | — |
| 1 | 8,192 | — | 55.08 | — | 55.17 | 53.46 | 1.71 | — | — | — |
| 2 | 0 | — | — | 119.35, 119.31 | — | — | — | — | — | — |
| 2 | 8,192 | — | — | 134.66, 134.79 | — | — | — | — | — | — |
| 3 | 0 | — | — | 150.78, 150.96 | — | — | — | — | — | — |
| 3 | 8,192 | — | — | 174.65, 174.71 | — | — | — | — | — | — |
| 4 | 0 | — | — | 177.43, 177.48 | — | — | — | — | — | — |
| 4 | 8,192 | — | — | 204.75, 204.90 | — | — | — | — | — | — |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1, w2g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 26.46, 26.49 | 26.87, 26.87 | 0.41, 0.38 |
| shared expert | 4.67, 4.66 | 4.65, 4.67 | -0.02, 0.01 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.37, 1.39 | 1.37, 1.37 | 0.00, -0.01 |
| attention and recurrent projections | 33.81, 33.76 | 33.79, 33.73 | -0.02, -0.03 |
| Gated DeltaNet state and conv | 5.77, 5.76 | 5.78, 5.77 | 0.01, 0.00 |
| attention (FLASH_ATTN_EXT) | 0.70, 0.70 | 11.87, 11.87 | 11.17, 11.16 |
| output layer | 4.97, 4.95 | 4.98, 4.97 | 0.01, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.24, 8.31 | 8.36, 8.43 | 0.12, 0.12 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 85.99, 86.02 | 97.67, 97.68 | 11.68, 11.65 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.28, 5.28 | 5.29, 5.29 |
| cards 0 to 7: routed experts | 25.06, 25.09 | 25.44, 25.44 |
| cards 0 to 7: shared expert, router, hyper-connections | 5.66, 5.66 | 5.64, 5.65 |
| cards 0 to 7: everything else | 39.83, 39.87 | 50.16, 50.18 |
| card 8: the head's product | 4.97, 4.95 | 4.98, 4.97 |
| card 8: the sampling chain after the product | 0.37, 0.36 | 0.38, 0.37 |
| card 8: GATED_DELTA_NET | 0.31, 0.31 | 0.31, 0.31 |
| card 8: routed experts | 1.40, 1.40 | 1.44, 1.44 |
| card 8: shared expert, router, hyper-connections | 0.39, 0.38 | 0.38, 0.38 |
| card 8: everything else | 2.73, 2.72 | 3.65, 3.65 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1, w3g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 37.40, 37.45 | 38.23, 38.26 | 0.83, 0.81 |
| shared expert | 5.80, 5.80 | 5.79, 5.80 | -0.00, 0.00 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.91, 1.90 | 1.91, 1.90 | 0.00, 0.00 |
| attention and recurrent projections | 41.77, 41.81 | 41.87, 41.90 | 0.09, 0.09 |
| Gated DeltaNet state and conv | 8.24, 8.26 | 8.26, 8.26 | 0.02, 0.01 |
| attention (FLASH_ATTN_EXT) | 1.06, 1.06 | 17.99, 18.01 | 16.93, 16.95 |
| output layer | 6.31, 6.30 | 6.32, 6.33 | 0.00, 0.04 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 9.01, 8.93 | 8.96, 8.91 | -0.05, -0.01 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 111.50, 111.49 | 129.32, 129.38 | 17.82, 17.89 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 7.58, 7.60 | 7.62, 7.62 |
| cards 0 to 7: routed experts | 35.42, 35.47 | 36.22, 36.25 |
| cards 0 to 7: shared expert, router, hyper-connections | 7.21, 7.21 | 7.21, 7.21 |
| cards 0 to 7: everything else | 48.18, 48.15 | 63.72, 63.74 |
| card 8: the head's product | 6.31, 6.30 | 6.32, 6.33 |
| card 8: the sampling chain after the product | 0.58, 0.57 | 0.58, 0.57 |
| card 8: GATED_DELTA_NET | 0.45, 0.45 | 0.45, 0.45 |
| card 8: routed experts | 1.98, 1.98 | 2.01, 2.01 |
| card 8: shared expert, router, hyper-connections | 0.49, 0.49 | 0.49, 0.49 |
| card 8: everything else | 3.29, 3.29 | 4.70, 4.71 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p2)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 47.26, 47.25 | 49.22, 49.30 | 1.96, 2.05 |
| shared expert | 7.54, 7.54 | 7.51, 7.50 | -0.03, -0.04 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.39, 1.38 | 1.37, 1.37 | -0.02, -0.00 |
| attention and recurrent projections | 48.31, 48.32 | 48.37, 48.46 | 0.06, 0.13 |
| Gated DeltaNet state and conv | 10.65, 10.64 | 10.67, 10.68 | 0.02, 0.04 |
| attention (FLASH_ATTN_EXT) | 1.06, 1.06 | 18.04, 18.06 | 16.98, 17.00 |
| output layer | 7.62, 7.60 | 7.63, 7.62 | 0.01, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 9.05, 9.02 | 9.13, 9.08 | 0.08, 0.06 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 132.88, 132.82 | 151.95, 152.07 | 19.06, 19.25 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 9.88, 9.87 | 9.90, 9.91 |
| cards 0 to 7: routed experts | 44.77, 44.76 | 46.63, 46.71 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.38, 8.37 | 8.33, 8.32 |
| cards 0 to 7: everything else | 54.09, 54.08 | 69.78, 69.84 |
| card 8: the head's product | 7.62, 7.60 | 7.63, 7.62 |
| card 8: the sampling chain after the product | 0.78, 0.78 | 0.79, 0.79 |
| card 8: GATED_DELTA_NET | 0.58, 0.58 | 0.58, 0.58 |
| card 8: routed experts | 2.49, 2.49 | 2.59, 2.59 |
| card 8: shared expert, router, hyper-connections | 0.55, 0.55 | 0.55, 0.55 |
| card 8: everything else | 3.75, 3.74 | 5.16, 5.16 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1, w2g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.70, 0.71 | 11.86, 11.87 | 11.17, 11.16 |
| `RMS_NORM+SCALE node` | 0.47, 0.36 | 0.64, 0.67 | 0.17, 0.31 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.66, 0.63 | 0.55, 0.51 | -0.12, -0.12 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.77, 6.79 | 6.90, 6.89 | 0.12, 0.10 |
| `SCALE node` | 0.36, 0.41 | 0.32, 0.30 | -0.05, -0.11 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.41, 3.41 | 3.47, 3.48 | 0.06, 0.07 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 3.68, 3.69 | 0.06, 0.07 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.79 | 4.84, 4.84 | 0.06, 0.05 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 5.00 | 5.05, 5.04 | 0.06, 0.05 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 2.93, 2.93 | 0.05, 0.05 |
| `RMS_NORM+MUL norm` | 0.59, 0.62 | 0.54, 0.59 | -0.06, -0.03 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.42, 9.41 | 9.45, 9.45 | 0.03, 0.04 |
| every label (the pass) | 85.99, 86.02 | 97.67, 97.68 | 11.67, 11.65 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1, w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.06, 1.06 | 17.99, 18.01 | 16.93, 16.95 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.61, 9.62 | 9.88, 9.89 | 0.27, 0.27 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 6.98, 6.99 | 7.18, 7.19 | 0.20, 0.20 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.81, 6.82 | 7.00, 7.01 | 0.19, 0.18 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.03 | 5.10, 5.10 | 0.08, 0.07 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.83, 4.84 | 4.91, 4.91 | 0.07, 0.07 |
| `RMS_NORM+MUL norm` | 0.90, 0.84 | 0.93, 0.92 | 0.04, 0.08 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.44 | 11.49, 11.50 | 0.06, 0.05 |
| `ADD attn_residual` | 0.21, 0.24 | 0.24, 0.32 | 0.03, 0.07 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.26, 0.26 | 0.21, 0.22 | -0.05, -0.04 |
| `SILU+MUL node` | 0.22, 0.21 | 0.28, 0.24 | 0.06, 0.03 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.94, 7.94 | 7.98, 7.97 | 0.04, 0.03 |
| every label (the pass) | 111.50, 111.49 | 129.32, 129.38 | 17.82, 17.89 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.06, 1.06 | 18.04, 18.06 | 16.97, 16.99 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.25, 12.25 | 12.76, 12.80 | 0.51, 0.55 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 8.84, 8.84 | 9.22, 9.23 | 0.38, 0.39 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 8.70, 8.69 | 9.04, 9.06 | 0.35, 0.37 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.21, 6.21 | 6.48, 6.49 | 0.27, 0.28 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.06, 6.07 | 6.32, 6.32 | 0.25, 0.26 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.19, 5.20 | 5.40, 5.41 | 0.20, 0.21 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.39, 0.38 | 0.48, 0.47 | 0.09, 0.09 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.49, 2.46 | 2.39, 2.37 | -0.09, -0.09 |
| `SWIGLU ffn_swiglu` | 0.23, 0.23 | 0.17, 0.13 | -0.06, -0.10 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.36, 2.37 | 2.45, 2.44 | 0.09, 0.07 |
| `RMS_NORM+MUL norm` | 0.55, 0.54 | 0.62, 0.61 | 0.07, 0.07 |
| every label (the pass) | 132.89, 132.82 | 151.95, 152.07 | 19.06, 19.25 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.77, 6.79 | 9.61, 9.62 | 2.84, 2.83 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.59 | 8.03, 8.05 | 2.44, 2.46 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.42, 9.41 | 11.78, 11.80 | 2.36, 2.39 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.26 | 11.43, 11.44 | 2.16, 2.18 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.79 | 6.81, 6.82 | 2.03, 2.04 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 5.00 | 6.98, 6.99 | 2.00, 1.99 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 7.94, 7.94 | 1.62, 1.63 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.41, 3.41 | 4.83, 4.84 | 1.42, 1.43 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 5.02, 5.03 | 1.40, 1.40 |
| `MUL_MAT output:q5_0:3072x248320` | 4.97, 4.95 | 6.31, 6.30 | 1.34, 1.34 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 4.13, 4.15 | 1.25, 1.27 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 5.15, 5.16 | 1.07, 1.08 |
| every label (the pass) | 85.99, 86.02 | 111.50, 111.49 | 25.51, 25.47 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.86, 11.87 | 17.99, 18.01 | 6.12, 6.14 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.90, 6.89 | 9.88, 9.89 | 2.99, 2.99 |
| `GATED_DELTA_NET+CPY node` | 5.60, 5.60 | 8.06, 8.07 | 2.46, 2.46 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.45, 9.45 | 11.82, 11.83 | 2.37, 2.38 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.28, 9.28 | 11.49, 11.50 | 2.21, 2.21 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.84, 4.84 | 7.00, 7.01 | 2.15, 2.17 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.05, 5.04 | 7.18, 7.19 | 2.14, 2.15 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 7.98, 7.97 | 1.66, 1.65 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.48 | 4.91, 4.91 | 1.43, 1.43 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.68, 3.69 | 5.10, 5.10 | 1.41, 1.40 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 6.32, 6.33 | 1.34, 1.36 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.93, 2.93 | 4.16, 4.17 | 1.23, 1.24 |
| every label (the pass) | 97.67, 97.68 | 129.32, 129.38 | 31.65, 31.70 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.61, 9.62 | 12.25, 12.25 | 2.64, 2.63 |
| `GATED_DELTA_NET+CPY node` | 8.03, 8.05 | 10.46, 10.45 | 2.42, 2.40 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.78, 11.80 | 14.16, 14.17 | 2.38, 2.37 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.81, 6.82 | 8.70, 8.69 | 1.88, 1.87 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 6.98, 6.99 | 8.84, 8.84 | 1.85, 1.84 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.94, 7.94 | 9.48, 9.48 | 1.55, 1.54 |
| `MUL_MAT output:q5_0:3072x248320` | 6.31, 6.30 | 7.62, 7.60 | 1.31, 1.31 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.83, 4.84 | 6.06, 6.07 | 1.23, 1.23 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.03 | 6.21, 6.21 | 1.19, 1.19 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.13, 4.15 | 5.19, 5.20 | 1.06, 1.05 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.44 | 12.49, 12.48 | 1.06, 1.04 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.15, 5.16 | 6.18, 6.18 | 1.03, 1.02 |
| every label (the pass) | 111.50, 111.49 | 132.89, 132.82 | 21.39, 21.33 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.88, 9.89 | 12.76, 12.80 | 2.88, 2.92 |
| `GATED_DELTA_NET+CPY node` | 8.06, 8.07 | 10.48, 10.49 | 2.42, 2.42 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.82, 11.83 | 14.19, 14.21 | 2.37, 2.38 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 7.00, 7.01 | 9.04, 9.06 | 2.04, 2.05 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.18, 7.19 | 9.22, 9.23 | 2.03, 2.03 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.98, 7.97 | 9.51, 9.52 | 1.53, 1.55 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.91, 4.91 | 6.32, 6.32 | 1.41, 1.41 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.10, 5.10 | 6.48, 6.49 | 1.39, 1.40 |
| `MUL_MAT output:q5_0:3072x248320` | 6.32, 6.33 | 7.64, 7.63 | 1.32, 1.29 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.16, 4.17 | 5.40, 5.41 | 1.23, 1.24 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.17, 5.18 | 6.20, 6.20 | 1.03, 1.02 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.49, 11.50 | 12.51, 12.52 | 1.02, 1.03 |
| every label (the pass) | 129.32, 129.38 | 151.95, 152.07 | 22.63, 22.69 |

## 4b. Depth 0: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.77, 6.79 | 12.25, 12.25 | 5.48, 5.46 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.59 | 10.46, 10.45 | 4.87, 4.86 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.42, 9.41 | 14.16, 14.17 | 4.74, 4.75 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.79 | 8.70, 8.69 | 3.92, 3.90 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 5.00 | 8.84, 8.84 | 3.85, 3.84 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.26 | 12.49, 12.48 | 3.22, 3.22 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 9.48, 9.48 | 3.17, 3.17 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.41, 3.41 | 6.06, 6.07 | 2.65, 2.66 |
| `MUL_MAT output:q5_0:3072x248320` | 4.97, 4.95 | 7.62, 7.60 | 2.65, 2.65 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.63, 3.63 | 6.21, 6.21 | 2.59, 2.59 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 5.19, 5.20 | 2.31, 2.32 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 6.18, 6.18 | 2.10, 2.10 |
| every label (the pass) | 85.99, 86.02 | 132.89, 132.82 | 46.89, 46.80 |

## 4b. Depth 8,192: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.86, 11.87 | 18.04, 18.06 | 6.17, 6.19 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.90, 6.89 | 12.76, 12.80 | 5.87, 5.91 |
| `GATED_DELTA_NET+CPY node` | 5.60, 5.60 | 10.48, 10.49 | 4.88, 4.89 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.45, 9.45 | 14.19, 14.21 | 4.74, 4.76 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.84, 4.84 | 9.04, 9.06 | 4.20, 4.21 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.05, 5.04 | 9.22, 9.23 | 4.17, 4.18 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.28, 9.28 | 12.51, 12.52 | 3.24, 3.24 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 9.51, 9.52 | 3.18, 3.19 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.48 | 6.32, 6.32 | 2.84, 2.85 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.68, 3.69 | 6.48, 6.49 | 2.80, 2.80 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 7.64, 7.63 | 2.66, 2.65 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.93, 2.93 | 5.40, 5.41 | 2.47, 2.48 |
| every label (the pass) | 97.67, 97.68 | 151.95, 152.07 | 54.28, 54.40 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p2; each load by value)

| family | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 16.80, 16.82 | 17.05, 17.05 | 23.65, 23.68 | 24.18, 24.21 | 29.81, 29.80 | 31.06, 31.10 |
| routed experts: down | 9.66, 9.67 | 9.82, 9.82 | 13.75, 13.77 | 14.04, 14.05 | 17.44, 17.45 | 18.16, 18.21 |
| shared expert: gate, up and GLU | 2.95, 2.94 | 2.93, 2.94 | 3.59, 3.58 | 3.58, 3.58 | 4.35, 4.39 | 4.45, 4.45 |
| shared expert: down | 1.57, 1.56 | 1.56, 1.58 | 1.99, 1.99 | 1.99, 1.99 | 2.49, 2.46 | 2.39, 2.37 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.16, 0.16 | 0.16, 0.15 | 0.22, 0.22 | 0.22, 0.23 | 0.71, 0.69 | 0.67, 0.68 |
| router (ffn_gate_inp) and its top-k | 1.74, 1.76 | 1.74, 1.73 | 2.26, 2.25 | 2.27, 2.26 | 1.72, 1.71 | 1.70, 1.70 |
| attention layers: q, k, v, output products | 7.87, 7.86 | 7.91, 7.86 | 9.75, 9.76 | 9.73, 9.75 | 11.23, 11.23 | 11.23, 11.23 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.94, 25.89 | 25.88, 25.86 | 32.02, 32.05 | 32.13, 32.15 | 37.08, 37.09 | 37.14, 37.23 |
| recurrent layers: GATED_DELTA_NET | 5.59, 5.59 | 5.60, 5.60 | 8.03, 8.05 | 8.06, 8.07 | 10.46, 10.45 | 10.48, 10.49 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.18, 0.18 | 0.17, 0.17 | 0.21, 0.21 | 0.19, 0.20 | 0.19, 0.19 | 0.19, 0.19 |
| attention: FLASH_ATTN_EXT | 0.70, 0.71 | 11.86, 11.87 | 1.06, 1.06 | 17.99, 18.01 | 1.06, 1.06 | 18.04, 18.06 |
| output head | 4.97, 4.95 | 4.98, 4.97 | 6.31, 6.30 | 6.32, 6.33 | 7.62, 7.60 | 7.64, 7.63 |
| everything else (norms, adds, copies, the sampling chain) | 7.88, 7.93 | 7.99, 8.07 | 8.66, 8.57 | 8.61, 8.56 | 8.73, 8.69 | 8.80, 8.75 |
| **the pass** | 85.99, 86.02 | 97.67, 97.68 | 111.50, 111.49 | 129.32, 129.38 | 132.89, 132.82 | 151.95, 152.07 |

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

