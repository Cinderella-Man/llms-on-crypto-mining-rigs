## 1. The step, ms (request p1: the first request at the depth), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 46.79 | — | 45.61 | 44.02 | 1.56 | — | — | — |
| 1 | 8,192 | — | 55.30 | — | 55.16 | 53.45 | 1.71 | — | — | — |
| 2 | 0 | — | — | 121.60, 121.69 | — | — | — | — | — | — |
| 2 | 8,192 | — | — | 134.45, 134.51 | — | — | — | — | — | — |
| 3 | 0 | — | — | 153.59, 153.47 | — | — | — | — | — | — |
| 3 | 8,192 | — | — | 174.17, 174.42 | — | — | — | — | — | — |
| 4 | 0 | — | — | 179.69, 179.56 | — | — | — | — | — | — |
| 4 | 8,192 | — | — | 204.72, 204.76 | — | — | — | — | — | — |

## 2. Width 2: the verify pass's GPU time by part, ms a pass, all nine cards (w2g1, w2g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 26.44, 26.45 | 26.86, 26.86 | 0.42, 0.40 |
| shared expert | 4.67, 4.66 | 4.63, 4.66 | -0.04, 0.00 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.37, 1.38 | 1.38, 1.37 | 0.01, -0.02 |
| attention and recurrent projections | 33.78, 33.72 | 33.73, 33.67 | -0.05, -0.05 |
| Gated DeltaNet state and conv | 5.76, 5.75 | 5.77, 5.76 | 0.01, 0.02 |
| attention (FLASH_ATTN_EXT) | 0.70, 0.70 | 11.86, 11.86 | 11.16, 11.16 |
| output layer | 4.96, 4.95 | 4.98, 4.97 | 0.02, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.25, 8.33 | 8.38, 8.46 | 0.14, 0.13 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 85.92, 85.94 | 97.59, 97.60 | 11.67, 11.66 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 5.26, 5.26 | 5.28, 5.29 |
| cards 0 to 7: routed experts | 25.04, 25.05 | 25.42, 25.42 |
| cards 0 to 7: shared expert, router, hyper-connections | 5.66, 5.65 | 5.63, 5.64 |
| cards 0 to 7: everything else | 39.81, 39.84 | 50.12, 50.14 |
| card 8: the head's product | 4.96, 4.95 | 4.98, 4.97 |
| card 8: the sampling chain after the product | 0.37, 0.37 | 0.38, 0.37 |
| card 8: GATED_DELTA_NET | 0.31, 0.31 | 0.31, 0.31 |
| card 8: routed experts | 1.40, 1.40 | 1.44, 1.43 |
| card 8: shared expert, router, hyper-connections | 0.39, 0.38 | 0.38, 0.38 |
| card 8: everything else | 2.73, 2.72 | 3.65, 3.64 |

## 2. Width 3: the verify pass's GPU time by part, ms a pass, all nine cards (w3g1, w3g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 37.39, 37.38 | 38.20, 38.23 | 0.81, 0.85 |
| shared expert | 5.79, 5.77 | 5.78, 5.79 | -0.01, 0.01 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.91, 1.90 | 1.93, 1.91 | 0.02, 0.01 |
| attention and recurrent projections | 41.77, 41.74 | 41.79, 41.84 | 0.01, 0.10 |
| Gated DeltaNet state and conv | 8.24, 8.23 | 8.25, 8.25 | 0.01, 0.02 |
| attention (FLASH_ATTN_EXT) | 1.05, 1.05 | 17.97, 17.99 | 16.91, 16.94 |
| output layer | 6.29, 6.29 | 6.33, 6.32 | 0.04, 0.03 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.92, 8.95 | 8.96, 8.91 | 0.04, -0.03 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 111.37, 111.31 | 129.20, 129.23 | 17.83, 17.92 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 7.59, 7.58 | 7.61, 7.61 |
| cards 0 to 7: routed experts | 35.41, 35.40 | 36.19, 36.22 |
| cards 0 to 7: shared expert, router, hyper-connections | 7.21, 7.19 | 7.22, 7.20 |
| cards 0 to 7: everything else | 48.10, 48.09 | 63.62, 63.67 |
| card 8: the head's product | 6.29, 6.29 | 6.33, 6.32 |
| card 8: the sampling chain after the product | 0.57, 0.57 | 0.58, 0.57 |
| card 8: GATED_DELTA_NET | 0.45, 0.45 | 0.45, 0.45 |
| card 8: routed experts | 1.98, 1.98 | 2.01, 2.01 |
| card 8: shared expert, router, hyper-connections | 0.49, 0.49 | 0.49, 0.49 |
| card 8: everything else | 3.28, 3.28 | 4.71, 4.70 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p1)

| part | depth 0 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|
| routed experts | 47.24, 47.17 | 49.20, 49.28 | 1.95, 2.11 |
| shared expert | 7.53, 7.55 | 7.52, 7.48 | -0.01, -0.07 |
| hyper-connections (BF16) | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| router | 1.38, 1.38 | 1.38, 1.37 | 0.00, -0.00 |
| attention and recurrent projections | 48.28, 48.20 | 48.37, 48.44 | 0.09, 0.24 |
| Gated DeltaNet state and conv | 10.64, 10.62 | 10.67, 10.68 | 0.03, 0.06 |
| attention (FLASH_ATTN_EXT) | 1.06, 1.06 | 18.03, 18.05 | 16.97, 17.00 |
| output layer | 7.61, 7.60 | 7.64, 7.62 | 0.03, 0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 9.07, 9.00 | 9.13, 9.11 | 0.06, 0.11 |
| other weights | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| sum of the parts | 132.81, 132.57 | 151.94, 152.04 | 19.13, 19.48 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 8,192 |
|---|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 9.87, 9.85 | 9.89, 9.91 |
| cards 0 to 7: routed experts | 44.75, 44.68 | 46.60, 46.69 |
| cards 0 to 7: shared expert, router, hyper-connections | 8.36, 8.38 | 8.35, 8.30 |
| cards 0 to 7: everything else | 54.08, 53.94 | 69.78, 69.87 |
| card 8: the head's product | 7.61, 7.60 | 7.64, 7.62 |
| card 8: the sampling chain after the product | 0.78, 0.78 | 0.79, 0.79 |
| card 8: GATED_DELTA_NET | 0.58, 0.58 | 0.58, 0.58 |
| card 8: routed experts | 2.49, 2.48 | 2.60, 2.59 |
| card 8: shared expert, router, hyper-connections | 0.55, 0.55 | 0.55, 0.55 |
| card 8: everything else | 3.74, 3.73 | 5.16, 5.15 |

## 4a. Width 2: what grows from depth 0 to 8,192, ms a pass (each load: w2g1, w2g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.70, 0.70 | 11.86, 11.86 | 11.16, 11.16 |
| `RMS_NORM+SCALE node` | 0.45, 0.37 | 0.67, 0.68 | 0.22, 0.31 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.66, 0.63 | 0.52, 0.51 | -0.15, -0.12 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.78, 6.78 | 6.89, 6.89 | 0.12, 0.11 |
| `SCALE node` | 0.36, 0.41 | 0.32, 0.31 | -0.05, -0.11 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.77 | 4.84, 4.84 | 0.07, 0.07 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.40, 3.41 | 3.47, 3.47 | 0.06, 0.06 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.62, 3.62 | 3.68, 3.69 | 0.06, 0.06 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 4.99 | 5.05, 5.04 | 0.06, 0.05 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 2.92, 2.93 | 0.05, 0.05 |
| `RMS_NORM+MUL norm` | 0.60, 0.63 | 0.55, 0.59 | -0.05, -0.04 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.41, 9.40 | 9.44, 9.44 | 0.03, 0.04 |
| every label (the pass) | 85.92, 85.94 | 97.59, 97.60 | 11.67, 11.66 |

## 4a. Width 3: what grows from depth 0 to 8,192, ms a pass (each load: w3g1, w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.05, 1.05 | 17.97, 17.99 | 16.91, 16.94 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.61, 9.60 | 9.87, 9.88 | 0.26, 0.28 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 6.98, 6.98 | 7.18, 7.18 | 0.20, 0.21 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.81, 6.81 | 6.99, 7.00 | 0.18, 0.19 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.02 | 5.09, 5.10 | 0.07, 0.08 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.83, 4.83 | 4.90, 4.91 | 0.07, 0.07 |
| `ADD attn_residual` | 0.19, 0.23 | 0.21, 0.33 | 0.02, 0.10 |
| `SILU+MUL node` | 0.22, 0.22 | 0.29, 0.24 | 0.07, 0.02 |
| `MUL_MAT ssm_alpha:q5_0:3072x64` | 0.55, 0.54 | 0.50, 0.51 | -0.05, -0.03 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.93, 7.92 | 7.97, 7.96 | 0.04, 0.04 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.43 | 11.45, 11.48 | 0.02, 0.05 |
| `RMS_NORM+MUL norm` | 0.87, 0.87 | 0.93, 0.88 | 0.06, 0.01 |
| every label (the pass) | 111.37, 111.31 | 129.20, 129.23 | 17.83, 17.92 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.06, 1.06 | 18.03, 18.05 | 16.97, 16.99 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 12.25, 12.24 | 12.75, 12.79 | 0.50, 0.55 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 8.84, 8.82 | 9.21, 9.23 | 0.37, 0.41 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 8.69, 8.67 | 9.04, 9.05 | 0.34, 0.38 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 6.21, 6.20 | 6.49, 6.49 | 0.28, 0.29 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 6.06, 6.06 | 6.32, 6.32 | 0.25, 0.26 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 5.19, 5.18 | 5.40, 5.40 | 0.21, 0.22 |
| `MUL_MAT ffn_down_shexp:q5_0:1024x3072` | 2.49, 2.46 | 2.38, 2.37 | -0.11, -0.09 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.38, 0.38 | 0.47, 0.48 | 0.09, 0.10 |
| `MUL_MAT ffn_gate_shexp:q5_0:3072x1024` | 2.36, 2.36 | 2.44, 2.46 | 0.08, 0.09 |
| `SWIGLU ffn_swiglu` | 0.24, 0.22 | 0.16, 0.15 | -0.08, -0.08 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 14.14, 14.13 | 14.19, 14.20 | 0.04, 0.07 |
| every label (the pass) | 132.81, 132.57 | 151.94, 152.04 | 19.13, 19.48 |

## 4b. Depth 0: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.78, 6.78 | 9.61, 9.60 | 2.84, 2.82 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.57 | 8.03, 8.02 | 2.46, 2.45 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.41, 9.40 | 11.78, 11.78 | 2.37, 2.38 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.26, 9.26 | 11.43, 11.43 | 2.17, 2.16 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.77 | 6.81, 6.81 | 2.04, 2.03 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 4.99 | 6.98, 6.98 | 1.99, 1.99 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 7.93, 7.92 | 1.62, 1.62 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.40, 3.41 | 4.83, 4.83 | 1.43, 1.43 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.62, 3.62 | 5.02, 5.02 | 1.40, 1.40 |
| `MUL_MAT output:q5_0:3072x248320` | 4.96, 4.95 | 6.29, 6.29 | 1.34, 1.34 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 4.13, 4.14 | 1.26, 1.26 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 5.15, 5.15 | 1.07, 1.07 |
| every label (the pass) | 85.92, 85.94 | 111.37, 111.31 | 25.45, 25.37 |

## 4b. Depth 8,192: what grows from 2 tokens to 3, ms a pass (loads paired in order: w2g1 → w3g1, w2g2 → w3g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.86, 11.86 | 17.97, 17.99 | 6.11, 6.14 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.89, 6.89 | 9.87, 9.88 | 2.98, 2.99 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.60 | 8.06, 8.06 | 2.46, 2.46 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.44, 9.44 | 11.81, 11.82 | 2.37, 2.38 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.27 | 11.45, 11.48 | 2.18, 2.21 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.84, 4.84 | 6.99, 7.00 | 2.15, 2.16 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.05, 5.04 | 7.18, 7.18 | 2.13, 2.14 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 7.97, 7.96 | 1.65, 1.65 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.47 | 4.90, 4.91 | 1.44, 1.44 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.68, 3.69 | 5.09, 5.10 | 1.41, 1.41 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 6.33, 6.32 | 1.35, 1.35 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.92, 2.93 | 4.16, 4.17 | 1.24, 1.24 |
| every label (the pass) | 97.59, 97.60 | 129.20, 129.23 | 31.61, 31.63 |

## 4b. Depth 0: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.61, 9.60 | 12.25, 12.24 | 2.64, 2.64 |
| `GATED_DELTA_NET+CPY node` | 8.03, 8.02 | 10.45, 10.43 | 2.42, 2.40 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.78, 11.78 | 14.14, 14.13 | 2.36, 2.35 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.81, 6.81 | 8.69, 8.67 | 1.88, 1.86 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 6.98, 6.98 | 8.84, 8.82 | 1.86, 1.84 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.93, 7.92 | 9.48, 9.46 | 1.54, 1.54 |
| `MUL_MAT output:q5_0:3072x248320` | 6.29, 6.29 | 7.61, 7.60 | 1.32, 1.30 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.83, 4.83 | 6.06, 6.06 | 1.23, 1.22 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.02, 5.02 | 6.21, 6.20 | 1.19, 1.18 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.13, 4.14 | 5.19, 5.18 | 1.05, 1.04 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.43, 11.43 | 12.48, 12.46 | 1.05, 1.04 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.15, 5.15 | 6.18, 6.17 | 1.03, 1.02 |
| every label (the pass) | 111.37, 111.31 | 132.81, 132.57 | 21.44, 21.25 |

## 4b. Depth 8,192: what grows from 3 tokens to 4, ms a pass (loads paired in order: w3g1 → w4g1, w3g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 9.87, 9.88 | 12.75, 12.79 | 2.88, 2.91 |
| `GATED_DELTA_NET+CPY node` | 8.06, 8.06 | 10.48, 10.49 | 2.42, 2.43 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 11.81, 11.82 | 14.19, 14.20 | 2.38, 2.38 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 6.99, 7.00 | 9.04, 9.05 | 2.05, 2.05 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 7.18, 7.18 | 9.21, 9.23 | 2.03, 2.05 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 7.97, 7.96 | 9.50, 9.52 | 1.53, 1.55 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 4.90, 4.91 | 6.32, 6.32 | 1.41, 1.41 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 5.09, 5.10 | 6.49, 6.49 | 1.39, 1.40 |
| `MUL_MAT output:q5_0:3072x248320` | 6.33, 6.32 | 7.64, 7.62 | 1.31, 1.30 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 4.16, 4.17 | 5.40, 5.40 | 1.24, 1.23 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 11.45, 11.48 | 12.51, 12.52 | 1.06, 1.04 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 5.16, 5.17 | 6.20, 6.20 | 1.04, 1.03 |
| every label (the pass) | 129.20, 129.23 | 151.94, 152.04 | 22.74, 22.81 |

## 4b. Depth 0: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.78, 6.78 | 12.25, 12.24 | 5.47, 5.46 |
| `GATED_DELTA_NET+CPY node` | 5.57, 5.57 | 10.45, 10.43 | 4.88, 4.85 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.41, 9.40 | 14.14, 14.13 | 4.74, 4.73 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.78, 4.77 | 8.69, 8.67 | 3.92, 3.89 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 4.99, 4.99 | 8.84, 8.82 | 3.85, 3.83 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.26, 9.26 | 12.48, 12.46 | 3.22, 3.20 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.31, 6.31 | 9.48, 9.46 | 3.17, 3.16 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.40, 3.41 | 6.06, 6.06 | 2.66, 2.65 |
| `MUL_MAT output:q5_0:3072x248320` | 4.96, 4.95 | 7.61, 7.60 | 2.65, 2.64 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.62, 3.62 | 6.21, 6.20 | 2.59, 2.58 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.88, 2.88 | 5.19, 5.18 | 2.31, 2.30 |
| `MUL_MAT attn_q:q5_0:3072x16384` | 4.08, 4.08 | 6.18, 6.17 | 2.10, 2.09 |
| every label (the pass) | 85.92, 85.94 | 132.81, 132.57 | 46.89, 46.62 |

## 4b. Depth 8,192: what grows from 2 tokens to 4, ms a pass (loads paired in order: w2g1 → w4g1, w2g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 11.86, 11.86 | 18.03, 18.05 | 6.17, 6.20 |
| `MUL_MAT_ID ffn_down_exps:n2p:1024x3072` | 6.89, 6.89 | 12.75, 12.79 | 5.86, 5.90 |
| `GATED_DELTA_NET+CPY node` | 5.59, 5.60 | 10.48, 10.49 | 4.88, 4.89 |
| `MUL_MAT attn_qkv:q5_0:3072x12288` | 9.44, 9.44 | 14.19, 14.20 | 4.75, 4.76 |
| `MUL_MAT_ID ffn_up_exps:n2p:3072x1024` | 4.84, 4.84 | 9.04, 9.05 | 4.20, 4.21 |
| `MUL_MAT_ID ffn_gate_exps:n2p:3072x1024` | 5.05, 5.04 | 9.21, 9.23 | 4.16, 4.19 |
| `MUL_MAT ssm_out:q5_0:8192x3072` | 9.27, 9.27 | 12.51, 12.52 | 3.24, 3.25 |
| `MUL_MAT attn_gate:q5_0:3072x8192` | 6.32, 6.32 | 9.50, 9.52 | 3.18, 3.20 |
| `MUL_MAT_ID ffn_up_exps:n3p:3072x1024` | 3.47, 3.47 | 6.32, 6.32 | 2.85, 2.85 |
| `MUL_MAT_ID ffn_gate_exps:n3p:3072x1024` | 3.68, 3.69 | 6.49, 6.49 | 2.80, 2.81 |
| `MUL_MAT output:q5_0:3072x248320` | 4.98, 4.97 | 7.64, 7.62 | 2.66, 2.65 |
| `MUL_MAT_ID ffn_down_exps:n3p:1024x3072` | 2.92, 2.93 | 5.40, 5.40 | 2.48, 2.47 |
| every label (the pass) | 97.59, 97.60 | 151.94, 152.04 | 54.35, 54.44 |

## 5. The pass by family of launch groups, ms a pass, all nine cards (request p1; each load by value)

| family | 2 tokens, depth 0 | 2 tokens, depth 8,192 | 3 tokens, depth 0 | 3 tokens, depth 8,192 | 4 tokens, depth 0 | 4 tokens, depth 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| routed experts: gate, up and GLU | 16.79, 16.80 | 17.04, 17.04 | 23.64, 23.64 | 24.17, 24.18 | 29.81, 29.75 | 31.05, 31.10 |
| routed experts: down | 9.65, 9.66 | 9.82, 9.81 | 13.75, 13.74 | 14.03, 14.04 | 17.44, 17.42 | 18.15, 18.19 |
| shared expert: gate, up and GLU | 2.95, 2.94 | 2.92, 2.93 | 3.59, 3.57 | 3.57, 3.58 | 4.33, 4.38 | 4.47, 4.44 |
| shared expert: down | 1.56, 1.56 | 1.56, 1.57 | 1.99, 1.99 | 1.99, 1.99 | 2.49, 2.46 | 2.38, 2.37 |
| shared expert: its gate (ffn_gate_inp_shexp) | 0.16, 0.16 | 0.15, 0.15 | 0.22, 0.22 | 0.22, 0.22 | 0.71, 0.70 | 0.67, 0.67 |
| router (ffn_gate_inp) and its top-k | 1.74, 1.76 | 1.75, 1.73 | 2.26, 2.26 | 2.28, 2.26 | 1.71, 1.71 | 1.72, 1.71 |
| attention layers: q, k, v, output products | 7.86, 7.86 | 7.90, 7.86 | 9.75, 9.74 | 9.73, 9.74 | 11.22, 11.21 | 11.23, 11.23 |
| recurrent layers: products (attn_qkv, attn_gate, ssm_*) | 25.92, 25.86 | 25.83, 25.81 | 32.03, 32.00 | 32.06, 32.10 | 37.05, 37.00 | 37.15, 37.21 |
| recurrent layers: GATED_DELTA_NET | 5.57, 5.57 | 5.59, 5.60 | 8.03, 8.02 | 8.06, 8.06 | 10.45, 10.43 | 10.48, 10.49 |
| recurrent layers: conv (SSM_CONV, GDN_CONV) | 0.18, 0.18 | 0.17, 0.16 | 0.21, 0.20 | 0.20, 0.19 | 0.19, 0.19 | 0.19, 0.19 |
| attention: FLASH_ATTN_EXT | 0.70, 0.70 | 11.86, 11.86 | 1.05, 1.05 | 17.97, 17.99 | 1.06, 1.06 | 18.03, 18.05 |
| output head | 4.96, 4.95 | 4.98, 4.97 | 6.29, 6.29 | 6.33, 6.32 | 7.61, 7.60 | 7.64, 7.62 |
| everything else (norms, adds, copies, the sampling chain) | 7.88, 7.95 | 8.02, 8.10 | 8.57, 8.58 | 8.60, 8.56 | 8.73, 8.67 | 8.79, 8.78 |
| **the pass** | 85.92, 85.94 | 97.59, 97.60 | 111.37, 111.31 | 129.20, 129.23 | 132.81, 132.57 | 151.94, 152.04 |

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

