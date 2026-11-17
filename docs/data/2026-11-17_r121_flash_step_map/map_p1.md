## 1. The step, ms (request p1: the first request at the depth), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 52.80, 52.81 | 76.85, 76.79 | 51.41, 51.45 | 49.36, 49.34 | 1.77, 1.77 | — | — | — |
| 1 | 4,096 | — | 54.40, 55.32 | 81.19, 80.02 | 53.95, 53.99 | 52.17, 52.18 | 1.78, 1.78 | — | — | — |
| 1 | 8,192 | — | 54.62, 54.78 | 80.00, 80.89 | 54.16, 54.18 | 52.35, 52.37 | 1.79, 1.80 | — | — | — |
| 4 | 0 | 102.93 | 103.36, 104.56 | 131.80, 131.58 | 100.90, 100.90 | 84.15, 84.12 | 3.59, 3.59 | 13.00, 13.11 | 8.33, 8.35 | 1.69, 1.73 |
| 4 | 4,096 | 113.79 | 114.63, 114.48 | 154.07, 144.33 | 112.61, 112.44 | 94.24, 94.12 | 3.97, 3.96 | 14.20, 13.87 | 9.41, 9.38 | 1.50, 1.12 |
| 4 | 8,192 | 118.52 | 119.44, 119.39 | 147.44, 147.23 | 116.39, 116.47 | 97.48, 97.56 | 4.41, 4.42 | 14.22, 14.23 | 9.55, 9.52 | 0.98, 0.97 |
| 5 | 0 | — | 143.26, 143.39 | 174.03, 173.89 | 140.27, 140.47 | 118.79, 118.93 | 4.29, 4.29 | 16.76, 16.56 | 10.98, 11.04 | 1.75, 1.67 |
| 5 | 4,096 | — | 158.80, 159.02 | 208.16, 188.13 | 156.07, 156.18 | 133.06, 133.10 | 4.66, 4.66 | 18.24, 18.38 | 12.33, 12.32 | 1.71, 1.90 |
| 5 | 8,192 | — | 172.43, 172.34 | 201.75, 201.14 | 170.71, 170.88 | 147.05, 147.23 | 5.09, 5.12 | 18.37, 18.53 | 12.53, 12.55 | 1.24, 1.35 |

## 2. Width 1: the verify pass's GPU time by part, ms a pass, all nine cards (w1g1, w1g2; request p1)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 7.17, 7.19 | 7.18, 7.19 | 7.18, 7.22 | 0.02, 0.03 |
| shared expert | 2.66, 2.64 | 2.66, 2.66 | 2.67, 2.66 | 0.01, 0.02 |
| hyper-connections (BF16) | 9.61, 9.66 | 9.50, 9.60 | 9.50, 9.50 | -0.11, -0.16 |
| router | 1.82, 1.82 | 1.82, 1.84 | 1.82, 1.85 | 0.00, 0.03 |
| attention and recurrent projections | 15.72, 15.74 | 15.89, 15.84 | 15.84, 15.83 | 0.12, 0.09 |
| Gated DeltaNet state and conv | 2.64, 2.64 | 2.66, 2.65 | 2.66, 2.66 | 0.02, 0.01 |
| attention (FLASH_ATTN_EXT) | 0.59, 0.58 | 2.90, 2.90 | 2.91, 2.93 | 2.32, 2.34 |
| output layer | 2.86, 2.86 | 2.86, 2.86 | 2.86, 2.86 | -0.00, 0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 5.16, 5.17 | 5.43, 5.45 | 5.60, 5.62 | 0.44, 0.45 |
| other weights | 0.57, 0.56 | 0.58, 0.57 | 0.59, 0.60 | 0.03, 0.03 |
| sum of the parts | 48.80, 48.87 | 51.48, 51.56 | 51.64, 51.72 | 2.84, 2.85 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 2.57, 2.57 | 2.59, 2.57 | 2.59, 2.58 |
| cards 0 to 7: GDN_CONV | 0.62, 0.44 | 0.39, 0.41 | 0.44, 0.42 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.05, 0.05 | 0.05, 0.05 | 0.05, 0.05 |
| cards 0 to 7: routed experts | 6.87, 6.89 | 6.89, 6.88 | 6.89, 6.92 |
| cards 0 to 7: shared expert, router, hyper-connections | 13.43, 13.45 | 13.30, 13.42 | 13.31, 13.32 |
| cards 0 to 7: everything else | 20.12, 20.34 | 22.90, 22.86 | 22.97, 23.05 |
| card 8: the head's product | 2.86, 2.86 | 2.86, 2.86 | 2.86, 2.86 |
| card 8: the sampling chain after the product | 0.20, 0.20 | 0.20, 0.20 | 0.20, 0.20 |
| card 8: the output's hyper-connection products | 0.08, 0.08 | 0.08, 0.08 | 0.08, 0.08 |
| card 8: GATED_DELTA_NET | 0.07, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| card 8: GDN_CONV | 0.09, 0.04 | 0.07, 0.01 | 0.06, 0.01 |
| card 8: routed experts | 0.30, 0.30 | 0.29, 0.30 | 0.29, 0.30 |
| card 8: shared expert, router, hyper-connections | 0.66, 0.68 | 0.68, 0.68 | 0.68, 0.69 |
| card 8: everything else | 0.88, 0.92 | 1.11, 1.16 | 1.14, 1.17 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p1)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 19.96, 19.95 | 19.99, 20.00 | 20.19, 20.20 | 0.23, 0.24 |
| shared expert | 3.64, 3.61 | 3.63, 3.65 | 3.64, 3.66 | 0.01, 0.05 |
| hyper-connections (BF16) | 12.80, 12.84 | 12.83, 12.85 | 12.82, 12.82 | 0.02, -0.02 |
| router | 2.43, 2.43 | 2.43, 2.43 | 2.42, 2.44 | -0.02, 0.00 |
| attention and recurrent projections | 23.24, 23.18 | 23.33, 23.31 | 23.35, 23.33 | 0.11, 0.15 |
| Gated DeltaNet state and conv | 8.91, 8.90 | 8.98, 9.00 | 8.98, 8.98 | 0.07, 0.08 |
| attention (FLASH_ATTN_EXT) | 0.97, 0.97 | 10.15, 10.15 | 12.63, 12.63 | 11.67, 11.67 |
| output layer | 3.10, 3.11 | 3.11, 3.11 | 3.11, 3.11 | 0.01, -0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 7.19, 7.18 | 8.01, 8.02 | 8.60, 8.61 | 1.41, 1.43 |
| other weights | 0.91, 0.96 | 0.93, 0.96 | 0.92, 0.95 | 0.01, -0.01 |
| sum of the parts | 83.16, 83.13 | 93.39, 93.48 | 96.67, 96.73 | 13.51, 13.60 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 8.66, 8.65 | 8.73, 8.74 | 8.71, 8.73 |
| cards 0 to 7: GDN_CONV | 0.73, 0.73 | 0.68, 0.70 | 0.69, 0.76 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.07, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| cards 0 to 7: routed experts | 19.14, 19.12 | 19.16, 19.17 | 19.36, 19.37 |
| cards 0 to 7: shared expert, router, hyper-connections | 18.06, 18.06 | 18.09, 18.11 | 18.08, 18.10 |
| cards 0 to 7: everything else | 29.19, 29.16 | 38.51, 38.52 | 41.40, 41.32 |
| card 8: the head's product | 3.10, 3.11 | 3.11, 3.11 | 3.11, 3.11 |
| card 8: the sampling chain after the product | 0.91, 0.91 | 0.91, 0.91 | 0.90, 0.90 |
| card 8: the output's hyper-connection products | 0.10, 0.10 | 0.10, 0.10 | 0.10, 0.10 |
| card 8: GATED_DELTA_NET | 0.25, 0.25 | 0.26, 0.26 | 0.27, 0.25 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 0.82, 0.83 | 0.82, 0.82 | 0.83, 0.83 |
| card 8: shared expert, router, hyper-connections | 0.81, 0.81 | 0.81, 0.81 | 0.80, 0.81 |
| card 8: everything else | 1.30, 1.31 | 2.12, 2.12 | 2.33, 2.36 |

## 2. Width 5: the verify pass's GPU time by part, ms a pass, all nine cards (w5g1, w5g2; request p1)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 24.14, 24.14 | 24.46, 24.45 | 24.66, 24.66 | 0.52, 0.52 |
| shared expert | 4.47, 4.47 | 4.53, 4.57 | 4.55, 4.52 | 0.07, 0.05 |
| hyper-connections (BF16) | 18.45, 18.45 | 18.49, 18.51 | 18.46, 18.45 | 0.01, -0.00 |
| router | 4.66, 4.66 | 4.68, 4.68 | 4.67, 4.67 | 0.01, 0.01 |
| attention and recurrent projections | 36.10, 36.11 | 36.11, 36.12 | 36.31, 36.23 | 0.21, 0.12 |
| Gated DeltaNet state and conv | 11.01, 10.99 | 11.05, 11.06 | 11.03, 11.02 | 0.02, 0.03 |
| attention (FLASH_ATTN_EXT) | 1.47, 1.47 | 14.60, 14.61 | 27.54, 27.55 | 26.07, 26.08 |
| output layer | 8.72, 8.73 | 8.70, 8.73 | 8.71, 8.70 | -0.01, -0.04 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 7.85, 7.78 | 8.88, 8.72 | 9.52, 9.63 | 1.67, 1.86 |
| other weights | 1.10, 1.11 | 1.12, 1.11 | 1.11, 1.11 | 0.01, -0.00 |
| sum of the parts | 117.97, 117.91 | 132.63, 132.55 | 146.56, 146.54 | 28.59, 28.64 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 10.69, 10.68 | 10.73, 10.74 | 10.72, 10.71 |
| cards 0 to 7: GDN_CONV | 0.80, 0.79 | 0.80, 0.79 | 0.84, 0.85 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.08, 0.08 | 0.09, 0.08 | 0.08, 0.08 |
| cards 0 to 7: routed experts | 23.13, 23.12 | 23.43, 23.43 | 23.64, 23.64 |
| cards 0 to 7: shared expert, router, hyper-connections | 26.34, 26.34 | 26.46, 26.52 | 26.44, 26.40 |
| cards 0 to 7: everything else | 42.23, 42.17 | 55.28, 55.12 | 67.86, 67.91 |
| card 8: the head's product | 8.72, 8.73 | 8.70, 8.73 | 8.71, 8.70 |
| card 8: the sampling chain after the product | 1.05, 1.06 | 1.05, 1.05 | 1.05, 1.04 |
| card 8: the output's hyper-connection products | 0.16, 0.16 | 0.16, 0.16 | 0.16, 0.16 |
| card 8: GATED_DELTA_NET | 0.32, 0.31 | 0.32, 0.31 | 0.31, 0.32 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 1.01, 1.01 | 1.03, 1.03 | 1.02, 1.02 |
| card 8: shared expert, router, hyper-connections | 1.24, 1.24 | 1.24, 1.24 | 1.24, 1.24 |
| card 8: everything else | 2.17, 2.19 | 3.32, 3.33 | 4.47, 4.46 |

## 4a. Width 1: what grows from depth 0 to 8,192, ms a pass (each load: w1g1, w1g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.59, 0.58 | 2.91, 2.93 | 2.32, 2.34 |
| `LIGHTNING_INDEXER indexer_score` | 0.14, 0.14 | 0.44, 0.42 | 0.30, 0.28 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.33, 0.32 | 0.19, 0.18 |
| `SOFTPLUS+MUL a_softplus` | 0.10, 0.14 | 0.24, 0.27 | 0.15, 0.13 |
| `GDN_CONV+CPY gdn_conv` | 0.71, 0.48 | 0.50, 0.43 | -0.21, -0.05 |
| `DSV4_HC_PRE hc_mixed` | 0.50, 0.54 | 0.42, 0.43 | -0.08, -0.11 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.41, 0.42 | 0.45, 0.46 | 0.05, 0.04 |
| `TOP_K_MASK indexer_sel` | 0.10, 0.09 | 0.13, 0.14 | 0.03, 0.05 |
| `ADD node` | 0.30, 0.49 | 0.32, 0.39 | 0.02, -0.10 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 0.10, 0.10 | 0.13, 0.14 | 0.03, 0.04 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.10, 0.11 | 0.13, 0.14 | 0.03, 0.03 |
| `SCALE node` | 0.21, 0.25 | 0.19, 0.22 | -0.02, -0.03 |
| every label (the pass) | 48.80, 48.87 | 51.64, 51.72 | 2.84, 2.85 |

### Width 1: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.59, 0.58 | 2.90, 2.90 | 2.31, 2.32 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.32, 0.32 | 0.18, 0.18 |
| `GDN_CONV+CPY gdn_conv` | 0.71, 0.48 | 0.45, 0.42 | -0.26, -0.06 |
| `SOFTPLUS+MUL a_softplus` | 0.10, 0.14 | 0.25, 0.24 | 0.15, 0.11 |
| `LIGHTNING_INDEXER indexer_score` | 0.14, 0.14 | 0.24, 0.24 | 0.10, 0.10 |
| `DSV4_HC_PRE hc_mixed` | 0.50, 0.54 | 0.39, 0.52 | -0.11, -0.02 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.41, 0.42 | 0.49, 0.44 | 0.09, 0.02 |
| `TOP_K_MASK indexer_sel` | 0.10, 0.09 | 0.14, 0.13 | 0.04, 0.05 |
| `ADD node` | 0.30, 0.49 | 0.33, 0.38 | 0.03, -0.10 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.49, 1.48 | 1.52, 1.52 | 0.03, 0.03 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.10, 0.11 | 0.15, 0.14 | 0.04, 0.02 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 0.10, 0.10 | 0.12, 0.12 | 0.02, 0.02 |
| every label (the pass) | 48.80, 48.87 | 51.48, 51.56 | 2.69, 2.68 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.97, 0.97 | 12.63, 12.64 | 11.67, 11.67 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 1.15, 1.15 | 1.03, 1.03 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.55, 0.55 | 0.42, 0.41 |
| `MUL gate` | 0.44, 0.42 | 0.30, 0.30 | -0.14, -0.12 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 11.19, 11.20 | 0.11, 0.13 |
| `SIGMOID beta_sigmoid` | 0.12, 0.15 | 0.25, 0.24 | 0.13, 0.09 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.14, 7.14 | 7.24, 7.23 | 0.10, 0.09 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.90 | 8.98, 8.98 | 0.07, 0.08 |
| `SOFTPLUS a_softplus` | 0.16, 0.17 | 0.09, 0.11 | -0.07, -0.06 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.59, 4.58 | 4.63, 4.63 | 0.04, 0.05 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 3.94, 3.93 | 3.96, 3.97 | 0.03, 0.04 |
| `MUL_MAT attn_q:q3_K:2560x12288` | 1.80, 1.80 | 1.83, 1.84 | 0.03, 0.04 |
| every label (the pass) | 83.16, 83.13 | 96.67, 96.73 | 13.51, 13.60 |

### Width 4: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.97, 0.97 | 10.15, 10.15 | 9.18, 9.18 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 0.62, 0.62 | 0.50, 0.50 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.54, 0.54 | 0.40, 0.40 |
| `MUL gate` | 0.44, 0.42 | 0.29, 0.28 | -0.15, -0.15 |
| `SIGMOID beta_sigmoid` | 0.12, 0.15 | 0.26, 0.26 | 0.14, 0.11 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.90 | 8.98, 9.00 | 0.07, 0.10 |
| `SOFTPLUS a_softplus` | 0.16, 0.17 | 0.09, 0.09 | -0.07, -0.08 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.59, 4.58 | 4.63, 4.62 | 0.04, 0.04 |
| `GDN_CONV+CPY gdn_conv` | 0.75, 0.75 | 0.70, 0.72 | -0.05, -0.02 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 11.10, 11.11 | 0.02, 0.04 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 3.94, 3.93 | 3.97, 3.96 | 0.03, 0.03 |
| `MUL_MAT attn_q:q3_K:2560x12288` | 1.80, 1.80 | 1.83, 1.84 | 0.02, 0.04 |
| every label (the pass) | 83.16, 83.13 | 93.39, 93.48 | 10.24, 10.35 |

## 4a. Width 5: what grows from depth 0 to 8,192, ms a pass (each load: w5g1, w5g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.47, 1.47 | 27.54, 27.55 | 26.07, 26.09 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 1.39, 1.39 | 1.27, 1.27 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.56, 0.56 | 0.41, 0.42 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.58, 13.58 | 13.89, 13.87 | 0.31, 0.30 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.43, 8.43 | 8.60, 8.60 | 0.16, 0.17 |
| `MUL gate` | 0.14, 0.18 | 0.23, 0.28 | 0.09, 0.10 |
| `SIGMOID beta_sigmoid` | 0.27, 0.23 | 0.17, 0.14 | -0.10, -0.09 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 5.11, 5.10 | 5.17, 5.17 | 0.06, 0.07 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 2.13, 2.13 | 2.18, 2.18 | 0.05, 0.05 |
| `GDN_CONV+CPY gdn_conv` | 0.82, 0.81 | 0.86, 0.87 | 0.04, 0.07 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 4.05, 4.05 | 4.09, 4.09 | 0.04, 0.04 |
| `CONCAT node` | 0.15, 0.15 | 0.21, 0.15 | 0.06, 0.00 |
| every label (the pass) | 117.97, 117.91 | 146.56, 146.55 | 28.59, 28.64 |

### Width 5: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.47, 1.47 | 14.60, 14.61 | 13.14, 13.14 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 0.76, 0.76 | 0.64, 0.64 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.55, 0.55 | 0.40, 0.40 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.58, 13.58 | 13.77, 13.77 | 0.19, 0.19 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.66, 0.71 | 0.54, 0.52 | -0.12, -0.20 |
| `SIGMOID beta_sigmoid` | 0.27, 0.23 | 0.37, 0.34 | 0.10, 0.11 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.43, 8.43 | 8.53, 8.53 | 0.10, 0.10 |
| `MUL gate` | 0.14, 0.18 | 0.09, 0.09 | -0.05, -0.09 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 5.11, 5.10 | 5.16, 5.16 | 0.05, 0.06 |
| `MUL_MAT ffn_gate_shexp:q3_K:2560x640` | 0.14, 0.14 | 0.18, 0.20 | 0.04, 0.06 |
| `GATED_DELTA_NET+CPY node` | 11.01, 10.99 | 11.05, 11.06 | 0.04, 0.06 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 4.33, 4.32 | 4.36, 4.37 | 0.03, 0.06 |
| every label (the pass) | 117.97, 117.91 | 132.63, 132.55 | 14.66, 14.65 |

## 4b. Depth 0: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.10, 3.11 | 8.73, 8.74 | 5.63, 5.62 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 13.58, 13.58 | 2.50, 2.51 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.89, 1.89 | 4.38, 4.39 | 2.49, 2.49 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.83, 1.83 | 4.26, 4.27 | 2.43, 2.43 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.66, 4.66 | 2.22, 2.23 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.90 | 11.01, 10.99 | 2.10, 2.09 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.47, 2.48 | 3.91, 3.92 | 1.43, 1.44 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.94, 2.93 | 1.39, 1.39 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.66, 2.66 | 4.05, 4.05 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.14, 7.14 | 8.43, 8.43 | 1.29, 1.29 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.60, 2.59 | 3.59, 3.60 | 0.99, 1.01 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 2.58, 2.59 | 3.56, 3.56 | 0.98, 0.97 |
| every label (the pass) | 83.16, 83.13 | 117.97, 117.91 | 34.82, 34.78 |

## 4b. Depth 4,096: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.11 | 8.70, 8.73 | 5.59, 5.62 |
| `FLASH_ATTN_EXT node` | 10.15, 10.15 | 14.60, 14.61 | 4.46, 4.46 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.10, 11.11 | 13.77, 13.77 | 2.67, 2.66 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.40, 4.40 | 2.50, 2.50 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.83, 1.84 | 4.28, 4.27 | 2.44, 2.44 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.68, 4.68 | 2.25, 2.25 |
| `GATED_DELTA_NET+CPY node` | 8.98, 9.00 | 11.05, 11.06 | 2.06, 2.06 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.47, 2.48 | 3.92, 3.93 | 1.45, 1.45 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.64, 2.65 | 4.06, 4.05 | 1.42, 1.40 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.93, 2.94 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.16, 7.16 | 8.53, 8.53 | 1.38, 1.37 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 2.58, 2.58 | 3.58, 3.58 | 1.00, 1.01 |
| every label (the pass) | 93.39, 93.48 | 132.63, 132.55 | 39.24, 39.07 |

## 4b. Depth 8,192: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 12.63, 12.64 | 27.54, 27.55 | 14.90, 14.92 |
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.11 | 8.71, 8.70 | 5.60, 5.59 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.19, 11.20 | 13.89, 13.87 | 2.70, 2.67 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.39, 4.39 | 2.49, 2.49 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.85, 1.84 | 4.28, 4.27 | 2.43, 2.44 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.42, 2.43 | 4.67, 4.67 | 2.26, 2.24 |
| `GATED_DELTA_NET+CPY node` | 8.98, 8.98 | 11.03, 11.02 | 2.05, 2.04 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.63, 2.64 | 4.09, 4.09 | 1.46, 1.44 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.48, 2.47 | 3.91, 3.91 | 1.43, 1.44 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.55 | 2.93, 2.94 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.24, 7.23 | 8.60, 8.60 | 1.36, 1.37 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.60, 2.60 | 3.61, 3.61 | 1.01, 1.01 |
| every label (the pass) | 96.67, 96.73 | 146.56, 146.55 | 49.89, 49.81 |

## 4c. Depth 0: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.61 | 11.08, 11.07 | 7.47, 7.45 |
| `GATED_DELTA_NET+CPY node` | 2.64, 2.64 | 8.91, 8.90 | 6.27, 6.26 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.00, 3.02 | 7.14, 7.14 | 4.14, 4.13 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.23, 3.25 | 4.59, 4.58 | 1.35, 1.34 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.55, 0.56 | 1.74, 1.74 | 1.19, 1.19 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.86, 2.87 | 3.94, 3.93 | 1.07, 1.06 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.42, 0.43 | 1.25, 1.27 | 0.82, 0.85 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.81, 0.81 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.95 | 2.58, 2.59 | 0.63, 0.63 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 1.82, 1.82 | 2.43, 2.43 | 0.61, 0.61 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.28 | 1.89, 1.89 | 0.60, 0.61 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.98, 1.99 | 2.60, 2.59 | 0.61, 0.60 |
| every label (the pass) | 48.80, 48.87 | 83.16, 83.13 | 34.36, 34.26 |

## 4c. Depth 4,096: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.62 | 11.10, 11.11 | 7.49, 7.49 |
| `FLASH_ATTN_EXT node` | 2.90, 2.90 | 10.15, 10.15 | 7.24, 7.25 |
| `GATED_DELTA_NET+CPY node` | 2.66, 2.65 | 8.98, 9.00 | 6.32, 6.35 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.02, 3.01 | 7.16, 7.16 | 4.14, 4.15 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.22, 3.25 | 4.63, 4.62 | 1.42, 1.38 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.73, 1.73 | 1.17, 1.18 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.88, 2.87 | 3.97, 3.96 | 1.09, 1.09 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.42, 0.42 | 1.28, 1.27 | 0.86, 0.85 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.82, 0.81 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.96 | 2.58, 2.58 | 0.64, 0.62 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.99, 1.98 | 2.60, 2.60 | 0.62, 0.62 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.90, 1.90 | 0.60, 0.61 |
| every label (the pass) | 51.48, 51.56 | 93.39, 93.48 | 41.91, 41.92 |

## 4c. Depth 8,192: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p1)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 2.91, 2.93 | 12.63, 12.64 | 9.72, 9.71 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.62, 3.64 | 11.19, 11.20 | 7.57, 7.56 |
| `GATED_DELTA_NET+CPY node` | 2.66, 2.66 | 8.98, 8.98 | 6.32, 6.32 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.02 | 7.24, 7.23 | 4.23, 4.21 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.23, 3.23 | 4.63, 4.63 | 1.39, 1.40 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.76, 1.76 | 1.20, 1.21 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.87, 2.88 | 3.96, 3.97 | 1.09, 1.09 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.41, 0.41 | 1.27, 1.27 | 0.86, 0.86 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.55 | 0.81, 0.82 |
| `LIGHTNING_INDEXER indexer_score` | 0.44, 0.42 | 1.15, 1.15 | 0.71, 0.73 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.94, 1.97 | 2.58, 2.59 | 0.64, 0.62 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.90, 1.90 | 0.61, 0.61 |
| every label (the pass) | 51.64, 51.72 | 96.67, 96.73 | 45.03, 45.01 |

