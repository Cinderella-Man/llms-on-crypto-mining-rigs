## 1. The step, ms (request p2: the repeat), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 51.19, 51.19 | 76.88, 76.84 | 51.06, 51.08 | 49.35, 49.36 | 1.71, 1.71 | — | — | — |
| 1 | 4,096 | — | 53.90, 53.89 | 80.08, 82.31 | 53.93, 53.94 | 52.16, 52.18 | 1.76, 1.76 | — | — | — |
| 1 | 8,192 | — | 54.15, 54.18 | 79.82, 80.90 | 54.19, 54.19 | 52.39, 52.39 | 1.79, 1.79 | — | — | — |
| 4 | 0 | 98.62 | 99.32, 99.27 | 130.48, 129.21 | 99.86, 99.80 | 84.02, 84.03 | 3.59, 3.59 | 12.13, 12.15 | 8.33, 8.36 | 0.96, 0.95 |
| 4 | 4,096 | 109.85 | 110.75, 110.34 | 140.96, 140.63 | 111.91, 111.92 | 94.36, 94.35 | 3.98, 3.97 | 13.59, 13.58 | 9.39, 9.39 | 0.96, 0.96 |
| 4 | 8,192 | 114.84 | 114.71, 114.91 | 146.12, 147.05 | 116.01, 116.16 | 97.43, 97.57 | 4.40, 4.43 | 14.17, 14.17 | 9.55, 9.52 | 0.97, 0.97 |
| 5 | 0 | — | 137.86, 137.98 | 170.31, 169.84 | 139.04, 139.22 | 118.78, 118.95 | 4.29, 4.29 | 15.77, 15.80 | 10.99, 11.03 | 1.12, 1.12 |
| 5 | 4,096 | — | 155.50, 155.38 | 191.57, 191.19 | 155.45, 155.20 | 133.19, 133.16 | 4.67, 4.66 | 17.60, 17.58 | 12.32, 12.36 | 1.14, 1.13 |
| 5 | 8,192 | — | 169.04, 166.78 | 200.03, 200.63 | 170.71, 170.43 | 147.18, 147.15 | 5.10, 5.12 | 18.32, 18.24 | 12.55, 12.54 | 1.14, 1.13 |

## 2. Width 1: the verify pass's GPU time by part, ms a pass, all nine cards (w1g1, w1g2; request p2)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 7.17, 7.19 | 7.18, 7.20 | 7.18, 7.22 | 0.01, 0.03 |
| shared expert | 2.67, 2.64 | 2.65, 2.66 | 2.68, 2.65 | 0.01, 0.01 |
| hyper-connections (BF16) | 9.59, 9.66 | 9.51, 9.59 | 9.51, 9.50 | -0.09, -0.16 |
| router | 1.82, 1.83 | 1.82, 1.85 | 1.82, 1.85 | -0.00, 0.02 |
| attention and recurrent projections | 15.73, 15.77 | 15.90, 15.84 | 15.84, 15.85 | 0.11, 0.08 |
| Gated DeltaNet state and conv | 2.65, 2.65 | 2.66, 2.65 | 2.66, 2.66 | 0.01, 0.01 |
| attention (FLASH_ATTN_EXT) | 0.59, 0.58 | 2.90, 2.90 | 2.91, 2.93 | 2.32, 2.34 |
| output layer | 2.86, 2.86 | 2.87, 2.86 | 2.86, 2.87 | 0.00, 0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 5.19, 5.14 | 5.45, 5.46 | 5.63, 5.64 | 0.45, 0.50 |
| other weights | 0.57, 0.57 | 0.58, 0.57 | 0.59, 0.59 | 0.02, 0.02 |
| sum of the parts | 48.84, 48.90 | 51.52, 51.58 | 51.69, 51.75 | 2.84, 2.85 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 2.57, 2.57 | 2.59, 2.58 | 2.59, 2.58 |
| cards 0 to 7: GDN_CONV | 0.64, 0.43 | 0.38, 0.42 | 0.44, 0.42 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.05, 0.05 | 0.05, 0.05 | 0.05, 0.05 |
| cards 0 to 7: routed experts | 6.88, 6.89 | 6.89, 6.89 | 6.89, 6.92 |
| cards 0 to 7: shared expert, router, hyper-connections | 13.43, 13.45 | 13.30, 13.40 | 13.32, 13.31 |
| cards 0 to 7: everything else | 20.14, 20.36 | 22.94, 22.87 | 23.01, 23.08 |
| card 8: the head's product | 2.86, 2.86 | 2.87, 2.86 | 2.86, 2.87 |
| card 8: the sampling chain after the product | 0.20, 0.20 | 0.20, 0.20 | 0.20, 0.20 |
| card 8: the output's hyper-connection products | 0.08, 0.08 | 0.08, 0.08 | 0.08, 0.08 |
| card 8: GATED_DELTA_NET | 0.07, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| card 8: GDN_CONV | 0.09, 0.04 | 0.07, 0.01 | 0.06, 0.01 |
| card 8: routed experts | 0.30, 0.30 | 0.30, 0.30 | 0.30, 0.30 |
| card 8: shared expert, router, hyper-connections | 0.66, 0.68 | 0.68, 0.69 | 0.69, 0.69 |
| card 8: everything else | 0.89, 0.91 | 1.11, 1.15 | 1.14, 1.17 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request p2)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 19.98, 19.97 | 20.00, 20.00 | 20.20, 20.22 | 0.23, 0.25 |
| shared expert | 3.58, 3.60 | 3.65, 3.65 | 3.66, 3.67 | 0.07, 0.07 |
| hyper-connections (BF16) | 12.83, 12.85 | 12.82, 12.85 | 12.83, 12.85 | 0.01, 0.00 |
| router | 2.44, 2.43 | 2.42, 2.42 | 2.43, 2.44 | -0.00, 0.00 |
| attention and recurrent projections | 23.27, 23.22 | 23.33, 23.34 | 23.40, 23.34 | 0.14, 0.12 |
| Gated DeltaNet state and conv | 8.91, 8.91 | 9.00, 8.99 | 8.98, 9.02 | 0.08, 0.11 |
| attention (FLASH_ATTN_EXT) | 0.97, 0.97 | 10.15, 10.15 | 12.66, 12.65 | 11.69, 11.68 |
| output layer | 3.10, 3.12 | 3.11, 3.12 | 3.11, 3.11 | 0.00, -0.01 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 7.25, 7.17 | 8.03, 8.02 | 8.62, 8.57 | 1.36, 1.40 |
| other weights | 0.90, 0.94 | 0.91, 0.95 | 0.91, 0.97 | 0.01, 0.03 |
| sum of the parts | 83.22, 83.19 | 93.42, 93.48 | 96.80, 96.84 | 13.58, 13.66 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 8.66, 8.66 | 8.75, 8.73 | 8.72, 8.76 |
| cards 0 to 7: GDN_CONV | 0.75, 0.78 | 0.70, 0.72 | 0.69, 0.74 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.08, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| cards 0 to 7: routed experts | 19.15, 19.14 | 19.17, 19.17 | 19.38, 19.39 |
| cards 0 to 7: shared expert, router, hyper-connections | 18.03, 18.06 | 18.08, 18.11 | 18.11, 18.14 |
| cards 0 to 7: everything else | 29.22, 29.10 | 38.48, 38.51 | 41.44, 41.34 |
| card 8: the head's product | 3.10, 3.12 | 3.11, 3.12 | 3.11, 3.11 |
| card 8: the sampling chain after the product | 0.91, 0.92 | 0.91, 0.91 | 0.90, 0.91 |
| card 8: the output's hyper-connection products | 0.10, 0.10 | 0.10, 0.10 | 0.10, 0.10 |
| card 8: GATED_DELTA_NET | 0.24, 0.25 | 0.25, 0.26 | 0.26, 0.25 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 0.82, 0.83 | 0.83, 0.83 | 0.83, 0.83 |
| card 8: shared expert, router, hyper-connections | 0.81, 0.81 | 0.81, 0.81 | 0.82, 0.81 |
| card 8: everything else | 1.31, 1.32 | 2.13, 2.13 | 2.36, 2.37 |

## 2. Width 5: the verify pass's GPU time by part, ms a pass, all nine cards (w5g1, w5g2; request p2)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 24.17, 24.17 | 24.49, 24.47 | 24.66, 24.67 | 0.49, 0.50 |
| shared expert | 4.51, 4.49 | 4.53, 4.53 | 4.55, 4.51 | 0.04, 0.03 |
| hyper-connections (BF16) | 18.39, 18.38 | 18.46, 18.45 | 18.45, 18.46 | 0.07, 0.07 |
| router | 4.67, 4.67 | 4.69, 4.68 | 4.67, 4.67 | 0.01, 0.00 |
| attention and recurrent projections | 36.19, 36.21 | 36.10, 36.10 | 36.32, 36.34 | 0.13, 0.13 |
| Gated DeltaNet state and conv | 11.01, 11.01 | 11.06, 11.04 | 11.03, 11.01 | 0.02, 0.00 |
| attention (FLASH_ATTN_EXT) | 1.48, 1.48 | 14.60, 14.61 | 27.55, 27.56 | 26.07, 26.08 |
| output layer | 8.72, 8.73 | 8.71, 8.69 | 8.71, 8.71 | -0.01, -0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 7.85, 7.85 | 8.90, 8.79 | 9.51, 9.59 | 1.66, 1.75 |
| other weights | 1.10, 1.10 | 1.12, 1.12 | 1.11, 1.10 | 0.01, -0.00 |
| sum of the parts | 118.08, 118.09 | 132.67, 132.49 | 146.56, 146.62 | 28.49, 28.53 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 10.70, 10.70 | 10.75, 10.73 | 10.72, 10.70 |
| cards 0 to 7: GDN_CONV | 0.83, 0.82 | 0.84, 0.80 | 0.84, 0.88 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.08, 0.08 | 0.09, 0.08 | 0.08, 0.08 |
| cards 0 to 7: routed experts | 23.16, 23.16 | 23.46, 23.44 | 23.64, 23.65 |
| cards 0 to 7: shared expert, router, hyper-connections | 26.32, 26.30 | 26.45, 26.42 | 26.44, 26.40 |
| cards 0 to 7: everything else | 42.29, 42.31 | 55.23, 55.16 | 67.86, 67.93 |
| card 8: the head's product | 8.72, 8.73 | 8.71, 8.69 | 8.71, 8.71 |
| card 8: the sampling chain after the product | 1.06, 1.06 | 1.05, 1.05 | 1.05, 1.05 |
| card 8: the output's hyper-connection products | 0.16, 0.16 | 0.16, 0.16 | 0.16, 0.16 |
| card 8: GATED_DELTA_NET | 0.31, 0.31 | 0.31, 0.31 | 0.31, 0.31 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 1.02, 1.01 | 1.02, 1.03 | 1.02, 1.02 |
| card 8: shared expert, router, hyper-connections | 1.24, 1.24 | 1.24, 1.24 | 1.24, 1.24 |
| card 8: everything else | 2.18, 2.19 | 3.33, 3.35 | 4.47, 4.47 |

## 4a. Width 1: what grows from depth 0 to 8,192, ms a pass (each load: w1g1, w1g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.59, 0.59 | 2.91, 2.93 | 2.32, 2.34 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.14 | 0.44, 0.42 | 0.30, 0.28 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.33, 0.32 | 0.19, 0.18 |
| `SOFTPLUS+MUL a_softplus` | 0.10, 0.14 | 0.24, 0.28 | 0.15, 0.14 |
| `GDN_CONV+CPY gdn_conv` | 0.72, 0.47 | 0.50, 0.43 | -0.22, -0.04 |
| `DSV4_HC_PRE hc_mixed` | 0.51, 0.54 | 0.43, 0.43 | -0.08, -0.11 |
| `TOP_K_MASK indexer_sel` | 0.09, 0.09 | 0.13, 0.13 | 0.04, 0.04 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.10, 0.12 | 0.14, 0.16 | 0.03, 0.04 |
| `ADD node` | 0.31, 0.47 | 0.33, 0.39 | 0.03, -0.09 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.40, 0.43 | 0.44, 0.45 | 0.05, 0.02 |
| `SCALE node` | 0.22, 0.25 | 0.20, 0.21 | -0.02, -0.04 |
| `MUL_MAT indexer.k_proj:bf16:2560x128` | 0.10, 0.10 | 0.13, 0.13 | 0.02, 0.03 |
| every label (the pass) | 48.84, 48.90 | 51.69, 51.75 | 2.84, 2.85 |

### Width 1: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.59, 0.59 | 2.90, 2.90 | 2.31, 2.32 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.32, 0.32 | 0.18, 0.18 |
| `GDN_CONV+CPY gdn_conv` | 0.72, 0.47 | 0.45, 0.42 | -0.27, -0.04 |
| `SOFTPLUS+MUL a_softplus` | 0.10, 0.14 | 0.26, 0.25 | 0.17, 0.11 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.14 | 0.24, 0.24 | 0.11, 0.10 |
| `DSV4_HC_PRE hc_mixed` | 0.51, 0.54 | 0.40, 0.51 | -0.11, -0.03 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.40, 0.43 | 0.47, 0.44 | 0.08, 0.01 |
| `ADD node` | 0.31, 0.47 | 0.33, 0.36 | 0.02, -0.11 |
| `TOP_K_MASK indexer_sel` | 0.09, 0.09 | 0.14, 0.13 | 0.04, 0.04 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.10, 0.12 | 0.16, 0.14 | 0.05, 0.02 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.50, 1.49 | 1.54, 1.51 | 0.04, 0.02 |
| `SET_ROWS kpool_dev_idxs` | 0.04, 0.04 | 0.06, 0.06 | 0.02, 0.02 |
| every label (the pass) | 48.84, 48.90 | 51.52, 51.58 | 2.68, 2.68 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.97, 0.97 | 12.66, 12.65 | 11.68, 11.68 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 1.15, 1.15 | 1.03, 1.04 |
| `TOP_K indexer_top_k` | 0.13, 0.14 | 0.56, 0.55 | 0.42, 0.42 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 11.21, 11.21 | 0.13, 0.15 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.91 | 8.98, 9.02 | 0.08, 0.11 |
| `MUL gate` | 0.44, 0.39 | 0.33, 0.32 | -0.12, -0.07 |
| `SOFTPLUS a_softplus` | 0.18, 0.16 | 0.09, 0.10 | -0.09, -0.06 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.15, 7.16 | 7.23, 7.24 | 0.07, 0.08 |
| `SIGMOID beta_sigmoid` | 0.15, 0.20 | 0.25, 0.24 | 0.10, 0.03 |
| `GDN_CONV+CPY gdn_conv` | 0.77, 0.80 | 0.71, 0.76 | -0.07, -0.04 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.37, 0.35 | 0.38, 0.41 | 0.01, 0.07 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.61, 4.60 | 4.64, 4.63 | 0.03, 0.04 |
| every label (the pass) | 83.22, 83.19 | 96.80, 96.84 | 13.58, 13.66 |

### Width 4: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.97, 0.97 | 10.15, 10.15 | 9.18, 9.18 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 0.62, 0.62 | 0.50, 0.50 |
| `TOP_K indexer_top_k` | 0.13, 0.14 | 0.53, 0.53 | 0.40, 0.40 |
| `MUL gate` | 0.44, 0.39 | 0.30, 0.30 | -0.14, -0.09 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.91 | 9.00, 8.99 | 0.09, 0.08 |
| `SOFTPLUS a_softplus` | 0.18, 0.16 | 0.09, 0.09 | -0.09, -0.07 |
| `SIGMOID beta_sigmoid` | 0.15, 0.20 | 0.26, 0.24 | 0.10, 0.04 |
| `GDN_CONV+CPY gdn_conv` | 0.77, 0.80 | 0.72, 0.74 | -0.05, -0.06 |
| `MUL_MAT ffn_gate_inp_shexp:f32:2560x1` | 0.25, 0.26 | 0.29, 0.28 | 0.03, 0.03 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 11.11, 11.11 | 0.02, 0.04 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.61, 4.60 | 4.63, 4.63 | 0.02, 0.03 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.35, 0.36 | 0.38, 0.38 | 0.03, 0.01 |
| every label (the pass) | 83.22, 83.19 | 93.42, 93.48 | 10.20, 10.29 |

## 4a. Width 5: what grows from depth 0 to 8,192, ms a pass (each load: w5g1, w5g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.48, 1.48 | 27.55, 27.56 | 26.07, 26.08 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 1.39, 1.39 | 1.27, 1.27 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.56, 0.56 | 0.41, 0.41 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.60, 13.60 | 13.88, 13.87 | 0.28, 0.27 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.45, 8.45 | 8.60, 8.62 | 0.15, 0.17 |
| `SIGMOID beta_sigmoid` | 0.25, 0.23 | 0.17, 0.13 | -0.08, -0.10 |
| `MUL gate` | 0.16, 0.18 | 0.20, 0.28 | 0.04, 0.10 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 2.13, 2.12 | 2.18, 2.18 | 0.05, 0.06 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.68, 0.71 | 0.74, 0.75 | 0.06, 0.04 |
| `CPY hop_wide` | 0.30, 0.30 | 0.25, 0.25 | -0.05, -0.05 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 5.13, 5.12 | 5.17, 5.17 | 0.04, 0.05 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 4.05, 4.05 | 4.08, 4.10 | 0.03, 0.05 |
| every label (the pass) | 118.08, 118.09 | 146.56, 146.62 | 28.49, 28.53 |

### Width 5: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.48, 1.48 | 14.60, 14.61 | 13.13, 13.14 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 0.76, 0.77 | 0.64, 0.64 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.55, 0.55 | 0.40, 0.40 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.68, 0.71 | 0.49, 0.50 | -0.19, -0.21 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.60, 13.60 | 13.78, 13.78 | 0.19, 0.17 |
| `SIGMOID beta_sigmoid` | 0.25, 0.23 | 0.40, 0.32 | 0.15, 0.08 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.45, 8.45 | 8.54, 8.54 | 0.09, 0.09 |
| `MUL gate` | 0.16, 0.18 | 0.09, 0.10 | -0.06, -0.08 |
| `MUL_MAT ffn_gate_shexp:q3_K:2560x640` | 0.14, 0.13 | 0.17, 0.19 | 0.04, 0.05 |
| `ADD node` | 0.38, 0.39 | 0.33, 0.35 | -0.05, -0.04 |
| `GATED_DELTA_NET+CPY node` | 11.01, 11.01 | 11.06, 11.04 | 0.06, 0.03 |
| `CPY hop_wide` | 0.30, 0.30 | 0.26, 0.26 | -0.05, -0.04 |
| every label (the pass) | 118.08, 118.09 | 132.67, 132.49 | 14.59, 14.40 |

## 4b. Depth 0: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.10, 3.12 | 8.72, 8.73 | 5.62, 5.61 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.08, 11.07 | 13.60, 13.60 | 2.51, 2.53 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.89, 1.89 | 4.39, 4.39 | 2.49, 2.50 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.85, 1.84 | 4.27, 4.27 | 2.41, 2.43 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.67, 4.67 | 2.23, 2.23 |
| `GATED_DELTA_NET+CPY node` | 8.91, 8.91 | 11.01, 11.01 | 2.10, 2.10 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.48, 2.49 | 3.91, 3.91 | 1.43, 1.43 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.66, 2.66 | 4.05, 4.05 | 1.39, 1.39 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.94, 2.94 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.15, 7.16 | 8.45, 8.45 | 1.30, 1.29 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.57, 2.58 | 3.58, 3.58 | 1.01, 1.00 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 2.59, 2.60 | 3.56, 3.56 | 0.97, 0.96 |
| every label (the pass) | 83.22, 83.19 | 118.08, 118.09 | 34.85, 34.90 |

## 4b. Depth 4,096: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.12 | 8.71, 8.69 | 5.60, 5.58 |
| `FLASH_ATTN_EXT node` | 10.15, 10.15 | 14.60, 14.61 | 4.46, 4.46 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.11, 11.11 | 13.78, 13.78 | 2.68, 2.67 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.40, 4.40 | 2.50, 2.50 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.84, 1.84 | 4.28, 4.27 | 2.44, 2.43 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.42, 2.42 | 4.69, 4.68 | 2.27, 2.26 |
| `GATED_DELTA_NET+CPY node` | 9.00, 8.99 | 11.06, 11.04 | 2.06, 2.06 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.48, 2.47 | 3.92, 3.92 | 1.45, 1.45 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.64, 2.65 | 4.08, 4.05 | 1.43, 1.40 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.94, 2.93 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.16, 7.16 | 8.54, 8.54 | 1.39, 1.38 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.59, 2.58 | 3.59, 3.60 | 1.00, 1.02 |
| every label (the pass) | 93.42, 93.48 | 132.67, 132.49 | 39.24, 39.01 |

## 4b. Depth 8,192: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 12.66, 12.65 | 27.55, 27.56 | 14.89, 14.90 |
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.11 | 8.72, 8.71 | 5.61, 5.60 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.21, 11.21 | 13.88, 13.87 | 2.67, 2.66 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.39, 4.39 | 2.49, 2.49 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.86, 1.84 | 4.28, 4.27 | 2.42, 2.44 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.67, 4.67 | 2.24, 2.23 |
| `GATED_DELTA_NET+CPY node` | 8.98, 9.02 | 11.02, 11.01 | 2.04, 2.00 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.48, 2.47 | 3.91, 3.91 | 1.43, 1.43 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.65, 2.67 | 4.08, 4.10 | 1.43, 1.43 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.55 | 2.94, 2.94 | 1.38, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.23, 7.24 | 8.60, 8.62 | 1.37, 1.38 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.59, 2.60 | 3.61, 3.60 | 1.02, 1.00 |
| every label (the pass) | 96.80, 96.84 | 146.56, 146.62 | 49.76, 49.78 |

## 4c. Depth 0: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.62 | 11.08, 11.07 | 7.47, 7.45 |
| `GATED_DELTA_NET+CPY node` | 2.65, 2.65 | 8.91, 8.91 | 6.26, 6.26 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.02 | 7.15, 7.16 | 4.15, 4.14 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.23, 3.24 | 4.61, 4.60 | 1.37, 1.36 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.55, 0.55 | 1.74, 1.74 | 1.19, 1.19 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.87, 2.87 | 3.95, 3.94 | 1.08, 1.07 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.41, 0.42 | 1.27, 1.27 | 0.86, 0.84 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.82, 0.81 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.95 | 2.59, 2.60 | 0.64, 0.65 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 1.82, 1.83 | 2.43, 2.43 | 0.61, 0.60 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.89, 1.89 | 0.60, 0.60 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.98, 2.00 | 2.57, 2.58 | 0.60, 0.58 |
| every label (the pass) | 48.84, 48.90 | 83.22, 83.19 | 34.38, 34.28 |

## 4c. Depth 4,096: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.63 | 11.11, 11.11 | 7.49, 7.48 |
| `FLASH_ATTN_EXT node` | 2.90, 2.90 | 10.15, 10.15 | 7.25, 7.25 |
| `GATED_DELTA_NET+CPY node` | 2.66, 2.65 | 9.00, 8.99 | 6.34, 6.34 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.02 | 7.16, 7.16 | 4.14, 4.15 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.21, 3.25 | 4.63, 4.63 | 1.42, 1.38 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.73, 1.73 | 1.18, 1.18 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.88, 2.88 | 3.96, 3.96 | 1.09, 1.09 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.43, 0.41 | 1.27, 1.29 | 0.84, 0.87 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.81, 0.81 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.96 | 2.58, 2.59 | 0.63, 0.63 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.90, 1.90 | 0.61, 0.61 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.98, 1.99 | 2.59, 2.58 | 0.61, 0.60 |
| every label (the pass) | 51.52, 51.58 | 93.42, 93.48 | 41.90, 41.90 |

## 4c. Depth 8,192: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request p2)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 2.91, 2.93 | 12.66, 12.65 | 9.74, 9.73 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.62, 3.64 | 11.21, 11.21 | 7.59, 7.57 |
| `GATED_DELTA_NET+CPY node` | 2.66, 2.66 | 8.98, 9.02 | 6.32, 6.36 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.02 | 7.23, 7.24 | 4.22, 4.22 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.24, 3.24 | 4.64, 4.63 | 1.40, 1.40 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.77, 1.77 | 1.21, 1.21 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.87, 2.88 | 3.97, 3.97 | 1.10, 1.10 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.41, 0.40 | 1.28, 1.27 | 0.87, 0.87 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.55 | 0.82, 0.81 |
| `LIGHTNING_INDEXER indexer_score` | 0.44, 0.42 | 1.15, 1.15 | 0.72, 0.73 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.94, 1.97 | 2.58, 2.58 | 0.64, 0.60 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.97, 1.99 | 2.59, 2.60 | 0.62, 0.61 |
| every label (the pass) | 51.69, 51.75 | 96.80, 96.84 | 45.12, 45.09 |

