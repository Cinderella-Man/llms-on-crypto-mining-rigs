## 1. The step, ms (request s: the sampled request), each load by value

| width | depth | server, unprofiled | server, host profile and trace | server, group profile | split: step | verify GPU | verify rest | draft span | draft GPU | server turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | — | 52.54, 52.53 | 77.02, 76.96 | 51.95, 52.14 | 49.44, 49.45 | 2.47, 2.61 | — | — | — |
| 1 | 4,096 | — | 54.70, 54.92 | 81.19, 80.56 | 54.32, 54.88 | 52.12, 52.13 | 1.93, 2.75 | — | — | — |
| 1 | 8,192 | — | 55.30, 55.32 | 84.25, 84.73 | 55.05, 55.04 | 52.37, 52.37 | 2.71, 2.69 | — | — | — |
| 4 | 0 | 104.49 | 104.06, 105.25 | 133.56, 133.91 | 100.35, 101.22 | 84.17, 84.13 | 3.61, 3.62 | 12.30, 13.23 | 8.35, 8.37 | 1.14, 1.81 |
| 4 | 4,096 | 112.08 | 112.09, 112.47 | 169.79, 143.72 | 112.75, 112.57 | 94.19, 94.19 | 4.00, 3.98 | 13.79, 13.75 | 9.41, 9.40 | 1.01, 1.01 |
| 4 | 8,192 | 116.73 | 117.66, 117.93 | 148.19, 147.79 | 116.73, 117.45 | 97.97, 97.98 | 4.42, 4.46 | 14.21, 15.02 | 9.55, 9.52 | 0.97, 1.70 |
| 5 | 0 | — | 143.89, 145.58 | 179.32, 179.07 | 140.04, 139.90 | 117.93, 118.14 | 4.33, 4.32 | 16.81, 16.80 | 11.04, 11.03 | 1.82, 1.75 |
| 5 | 4,096 | — | 155.94, 156.02 | 187.49, 190.09 | 155.65, 155.42 | 132.82, 132.84 | 4.69, 4.68 | 17.74, 17.64 | 12.34, 12.34 | 1.22, 1.21 |
| 5 | 8,192 | — | 173.00, 173.73 | 218.24, 203.64 | 172.15, 172.21 | 147.90, 147.83 | 5.13, 5.15 | 19.01, 19.25 | 12.55, 12.55 | 1.78, 1.87 |

## 2. Width 1: the verify pass's GPU time by part, ms a pass, all nine cards (w1g1, w1g2; request s)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 7.17, 7.17 | 7.18, 7.20 | 7.18, 7.19 | 0.01, 0.02 |
| shared expert | 2.65, 2.66 | 2.66, 2.67 | 2.66, 2.65 | 0.01, -0.01 |
| hyper-connections (BF16) | 9.51, 9.48 | 9.58, 9.70 | 9.50, 9.52 | -0.02, 0.04 |
| router | 1.82, 1.82 | 1.83, 1.84 | 1.82, 1.83 | 0.00, 0.01 |
| attention and recurrent projections | 15.86, 15.84 | 15.92, 15.87 | 15.83, 15.87 | -0.03, 0.03 |
| Gated DeltaNet state and conv | 2.65, 2.65 | 2.65, 2.65 | 2.65, 2.65 | -0.00, 0.00 |
| attention (FLASH_ATTN_EXT) | 0.60, 0.60 | 2.90, 2.89 | 2.92, 2.93 | 2.32, 2.33 |
| output layer | 2.86, 2.87 | 2.86, 2.86 | 2.86, 2.86 | -0.00, -0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 5.14, 5.19 | 5.36, 5.37 | 5.65, 5.69 | 0.51, 0.50 |
| other weights | 0.57, 0.58 | 0.56, 0.56 | 0.59, 0.57 | 0.03, -0.01 |
| sum of the parts | 48.84, 48.87 | 51.49, 51.63 | 51.67, 51.78 | 2.83, 2.91 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 2.58, 2.58 | 2.58, 2.58 | 2.58, 2.58 |
| cards 0 to 7: GDN_CONV | 0.37, 0.34 | 0.40, 0.49 | 0.43, 0.39 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.05, 0.05 | 0.05, 0.05 | 0.05, 0.05 |
| cards 0 to 7: routed experts | 6.87, 6.87 | 6.89, 6.90 | 6.88, 6.90 |
| cards 0 to 7: shared expert, router, hyper-connections | 13.29, 13.28 | 13.37, 13.52 | 13.32, 13.32 |
| cards 0 to 7: everything else | 20.51, 20.57 | 22.82, 22.71 | 23.01, 23.13 |
| card 8: the head's product | 2.86, 2.87 | 2.86, 2.86 | 2.86, 2.86 |
| card 8: the sampling chain after the product | 0.21, 0.21 | 0.21, 0.21 | 0.21, 0.21 |
| card 8: the output's hyper-connection products | 0.08, 0.08 | 0.08, 0.08 | 0.08, 0.08 |
| card 8: GATED_DELTA_NET | 0.07, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| card 8: GDN_CONV | 0.07, 0.04 | 0.05, 0.01 | 0.09, 0.05 |
| card 8: routed experts | 0.30, 0.30 | 0.29, 0.30 | 0.29, 0.30 |
| card 8: shared expert, router, hyper-connections | 0.70, 0.69 | 0.69, 0.70 | 0.66, 0.69 |
| card 8: everything else | 0.88, 0.92 | 1.13, 1.15 | 1.13, 1.15 |

## 2. Width 4: the verify pass's GPU time by part, ms a pass, all nine cards (w4g1, w4g2; request s)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 20.07, 20.08 | 19.88, 19.89 | 20.53, 20.51 | 0.45, 0.43 |
| shared expert | 3.63, 3.64 | 3.58, 3.59 | 3.62, 3.62 | -0.01, -0.01 |
| hyper-connections (BF16) | 12.81, 12.84 | 12.82, 12.91 | 12.78, 12.85 | -0.03, 0.01 |
| router | 2.42, 2.42 | 2.43, 2.43 | 2.43, 2.43 | 0.01, 0.02 |
| attention and recurrent projections | 23.26, 23.29 | 23.32, 23.45 | 23.31, 23.48 | 0.04, 0.19 |
| Gated DeltaNet state and conv | 8.99, 8.94 | 8.99, 8.95 | 9.02, 8.96 | 0.03, 0.01 |
| attention (FLASH_ATTN_EXT) | 1.00, 1.01 | 10.15, 10.17 | 12.69, 12.68 | 11.68, 11.67 |
| output layer | 3.10, 3.11 | 3.11, 3.11 | 3.11, 3.11 | 0.02, 0.00 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 7.18, 7.24 | 8.16, 8.10 | 8.66, 8.59 | 1.49, 1.35 |
| other weights | 0.90, 0.95 | 0.91, 0.95 | 0.91, 0.97 | 0.01, 0.02 |
| sum of the parts | 83.37, 83.51 | 93.34, 93.55 | 97.06, 97.20 | 13.69, 13.69 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 8.71, 8.69 | 8.74, 8.70 | 8.76, 8.70 |
| cards 0 to 7: GDN_CONV | 0.68, 0.69 | 0.69, 0.67 | 0.72, 0.71 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.07, 0.07 | 0.07, 0.07 | 0.07, 0.07 |
| cards 0 to 7: routed experts | 19.23, 19.23 | 19.06, 19.07 | 19.70, 19.68 |
| cards 0 to 7: shared expert, router, hyper-connections | 18.06, 18.08 | 18.02, 18.11 | 18.03, 18.10 |
| cards 0 to 7: everything else | 29.21, 29.30 | 38.52, 38.67 | 41.28, 41.46 |
| card 8: the head's product | 3.10, 3.11 | 3.11, 3.11 | 3.11, 3.11 |
| card 8: the sampling chain after the product | 0.99, 1.00 | 0.99, 0.99 | 0.99, 0.99 |
| card 8: the output's hyper-connection products | 0.10, 0.10 | 0.10, 0.10 | 0.10, 0.10 |
| card 8: GATED_DELTA_NET | 0.27, 0.26 | 0.25, 0.25 | 0.25, 0.26 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 0.84, 0.85 | 0.82, 0.82 | 0.83, 0.82 |
| card 8: shared expert, router, hyper-connections | 0.80, 0.81 | 0.81, 0.81 | 0.81, 0.81 |
| card 8: everything else | 1.26, 1.29 | 2.14, 2.15 | 2.38, 2.36 |

## 2. Width 5: the verify pass's GPU time by part, ms a pass, all nine cards (w5g1, w5g2; request s)

| part | depth 0 | depth 4,096 | depth 8,192 | 8,192 − 0 |
|---|---:|---:|---:|---:|
| routed experts | 23.49, 23.49 | 24.13, 24.13 | 24.84, 24.85 | 1.35, 1.36 |
| shared expert | 4.53, 4.54 | 4.53, 4.57 | 4.55, 4.58 | 0.03, 0.04 |
| hyper-connections (BF16) | 18.43, 18.43 | 18.43, 18.44 | 18.43, 18.41 | -0.01, -0.02 |
| router | 4.67, 4.67 | 4.67, 4.67 | 4.67, 4.67 | -0.00, 0.00 |
| attention and recurrent projections | 36.00, 36.00 | 36.17, 36.12 | 36.18, 36.24 | 0.19, 0.24 |
| Gated DeltaNet state and conv | 11.04, 11.04 | 11.06, 11.05 | 11.06, 11.06 | 0.02, 0.02 |
| attention (FLASH_ATTN_EXT) | 1.50, 1.50 | 14.61, 14.61 | 27.55, 27.56 | 26.05, 26.06 |
| output layer | 8.72, 8.73 | 8.71, 8.71 | 8.72, 8.72 | 0.00, -0.02 |
| token embeddings | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| other ops without weights | 8.01, 7.93 | 8.96, 8.97 | 9.58, 9.51 | 1.57, 1.58 |
| other weights | 1.11, 1.11 | 1.11, 1.11 | 1.10, 1.10 | -0.00, -0.00 |
| sum of the parts | 117.50, 117.45 | 132.38, 132.37 | 146.69, 146.70 | 29.18, 29.25 |

Where it goes (verify_parts.py's buckets), ms a pass:

| cards: bucket | depth 0 | depth 4,096 | depth 8,192 |
|---|---:|---:|---:|
| cards 0 to 7: GATED_DELTA_NET | 10.72, 10.72 | 10.74, 10.74 | 10.74, 10.74 |
| cards 0 to 7: GDN_CONV | 0.81, 0.82 | 0.81, 0.78 | 0.78, 0.79 |
| cards 0 to 7: copies of the PLE conv history (cache_ple_r) | 0.08, 0.08 | 0.08, 0.08 | 0.08, 0.08 |
| cards 0 to 7: routed experts | 22.48, 22.48 | 23.13, 23.12 | 23.82, 23.83 |
| cards 0 to 7: shared expert, router, hyper-connections | 26.39, 26.40 | 26.38, 26.44 | 26.41, 26.43 |
| cards 0 to 7: everything else | 42.21, 42.13 | 55.27, 55.26 | 67.78, 67.80 |
| card 8: the head's product | 8.72, 8.73 | 8.71, 8.71 | 8.72, 8.72 |
| card 8: the sampling chain after the product | 1.16, 1.16 | 1.16, 1.16 | 1.15, 1.15 |
| card 8: the output's hyper-connection products | 0.16, 0.16 | 0.16, 0.15 | 0.15, 0.16 |
| card 8: GATED_DELTA_NET | 0.32, 0.32 | 0.32, 0.31 | 0.31, 0.32 |
| card 8: GDN_CONV | 0.02, 0.02 | 0.02, 0.02 | 0.02, 0.02 |
| card 8: routed experts | 1.01, 1.01 | 1.00, 1.01 | 1.02, 1.02 |
| card 8: shared expert, router, hyper-connections | 1.24, 1.24 | 1.24, 1.24 | 1.24, 1.23 |
| card 8: everything else | 2.17, 2.17 | 3.34, 3.35 | 4.44, 4.41 |

## 4a. Width 1: what grows from depth 0 to 8,192, ms a pass (each load: w1g1, w1g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.60, 0.60 | 2.92, 2.93 | 2.32, 2.33 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 0.43, 0.43 | 0.30, 0.30 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.33, 0.33 | 0.18, 0.18 |
| `GDN_CONV+CPY gdn_conv` | 0.44, 0.38 | 0.52, 0.44 | 0.08, 0.06 |
| `SOFTPLUS+MUL a_softplus` | 0.25, 0.28 | 0.19, 0.26 | -0.07, -0.02 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.56, 1.54 | 1.50, 1.52 | -0.05, -0.01 |
| `TOP_K_MASK indexer_sel` | 0.11, 0.10 | 0.15, 0.13 | 0.04, 0.03 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.45, 0.46 | 0.47, 0.49 | 0.02, 0.03 |
| `ADD node` | 0.42, 0.40 | 0.39, 0.38 | -0.03, -0.02 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.14, 0.13 | 0.11, 0.12 | -0.03, -0.01 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.97, 1.97 | 1.99, 2.00 | 0.01, 0.02 |
| `SIGMOID beta_sigmoid` | 0.09, 0.09 | 0.07, 0.07 | -0.02, -0.02 |
| every label (the pass) | 48.84, 48.87 | 51.67, 51.78 | 2.83, 2.91 |

### Width 1: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 0.60, 0.60 | 2.89, 2.89 | 2.29, 2.29 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.33, 0.31 | 0.18, 0.17 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 0.25, 0.25 | 0.12, 0.12 |
| `ADD node` | 0.42, 0.40 | 0.31, 0.30 | -0.11, -0.10 |
| `SOFTPLUS+MUL a_softplus` | 0.25, 0.28 | 0.20, 0.15 | -0.06, -0.13 |
| `MUL_MAT ssm_alpha:bf16:2560x48` | 0.45, 0.46 | 0.57, 0.52 | 0.11, 0.06 |
| `DSV4_HC_PRE hc_mixed` | 0.40, 0.41 | 0.43, 0.51 | 0.03, 0.10 |
| `GDN_CONV+CPY gdn_conv` | 0.44, 0.38 | 0.44, 0.50 | 0.00, 0.11 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.56, 1.54 | 1.50, 1.49 | -0.05, -0.05 |
| `RMS_NORM+MUL node` | 0.32, 0.31 | 0.33, 0.36 | 0.01, 0.05 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.97, 1.97 | 1.99, 2.00 | 0.02, 0.03 |
| `TOP_K_MASK indexer_sel` | 0.11, 0.10 | 0.14, 0.12 | 0.03, 0.01 |
| every label (the pass) | 48.84, 48.87 | 51.49, 51.63 | 2.65, 2.76 |

## 4a. Width 4: what grows from depth 0 to 8,192, ms a pass (each load: w4g1, w4g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.00, 1.01 | 12.69, 12.68 | 11.68, 11.67 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 1.15, 1.15 | 1.03, 1.03 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.56, 0.55 | 0.42, 0.41 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.16, 11.16 | 11.38, 11.37 | 0.22, 0.21 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.19, 7.19 | 7.36, 7.35 | 0.17, 0.16 |
| `SIGMOID beta_sigmoid` | 0.21, 0.19 | 0.32, 0.20 | 0.11, 0.01 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 1.73, 1.73 | 1.79, 1.79 | 0.06, 0.06 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.60, 4.59 | 4.63, 4.63 | 0.03, 0.05 |
| `GDN_CONV+CPY gdn_conv` | 0.70, 0.71 | 0.74, 0.73 | 0.04, 0.02 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 3.94, 3.94 | 3.97, 3.97 | 0.03, 0.03 |
| `MUL_MAT attn_q:q3_K:2560x12288` | 1.81, 1.80 | 1.83, 1.83 | 0.02, 0.03 |
| `MUL gate` | 0.29, 0.33 | 0.25, 0.31 | -0.03, -0.02 |
| every label (the pass) | 83.37, 83.51 | 97.06, 97.20 | 13.69, 13.69 |

### Width 4: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.00, 1.01 | 10.15, 10.17 | 9.14, 9.16 |
| `LIGHTNING_INDEXER indexer_score` | 0.12, 0.12 | 0.62, 0.62 | 0.50, 0.50 |
| `TOP_K indexer_top_k` | 0.14, 0.14 | 0.54, 0.54 | 0.40, 0.40 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.16, 11.16 | 11.03, 11.05 | -0.12, -0.11 |
| `MUL gate` | 0.29, 0.33 | 0.36, 0.39 | 0.07, 0.07 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.19, 7.19 | 7.13, 7.12 | -0.06, -0.07 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 4.60, 4.59 | 4.63, 4.63 | 0.04, 0.05 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 3.94, 3.94 | 3.97, 3.97 | 0.03, 0.03 |
| `MUL_MAT attn_q:q3_K:2560x12288` | 1.81, 1.80 | 1.83, 1.83 | 0.02, 0.03 |
| `MUL ffn_shexp_gated` | 0.11, 0.12 | 0.09, 0.10 | -0.02, -0.03 |
| `FILL node` | 0.14, 0.14 | 0.11, 0.12 | -0.02, -0.02 |
| `SIGMOID shared_expert_gate_sigmoid` | 0.09, 0.10 | 0.07, 0.08 | -0.02, -0.02 |
| every label (the pass) | 83.37, 83.51 | 93.34, 93.55 | 9.97, 10.04 |

## 4a. Width 5: what grows from depth 0 to 8,192, ms a pass (each load: w5g1, w5g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.50, 1.50 | 27.55, 27.56 | 26.05, 26.06 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 1.39, 1.39 | 1.27, 1.27 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.20, 13.20 | 13.95, 13.95 | 0.74, 0.75 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.20, 8.20 | 8.68, 8.69 | 0.48, 0.49 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.57, 0.57 | 0.42, 0.42 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 2.09, 2.09 | 2.22, 2.22 | 0.13, 0.13 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.51, 0.48 | 0.59, 0.63 | 0.09, 0.14 |
| `SIGMOID beta_sigmoid` | 0.37, 0.34 | 0.34, 0.25 | -0.03, -0.09 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 5.13, 5.12 | 5.16, 5.16 | 0.03, 0.04 |
| `GDN_CONV+CPY gdn_conv` | 0.83, 0.84 | 0.80, 0.81 | -0.03, -0.02 |
| `DSV4_HC_PRE hc_mixed` | 0.61, 0.56 | 0.56, 0.56 | -0.05, 0.00 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.39, 0.35 | 0.39, 0.40 | 0.00, 0.05 |
| every label (the pass) | 117.50, 117.45 | 146.69, 146.70 | 29.18, 29.25 |

### Width 5: depth 0 to 4,096

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 1.50, 1.50 | 14.61, 14.61 | 13.10, 13.11 |
| `LIGHTNING_INDEXER indexer_score` | 0.13, 0.13 | 0.76, 0.76 | 0.64, 0.64 |
| `TOP_K indexer_top_k` | 0.15, 0.15 | 0.55, 0.55 | 0.40, 0.40 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 13.20, 13.20 | 13.57, 13.57 | 0.36, 0.36 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 8.20, 8.20 | 8.42, 8.42 | 0.22, 0.22 |
| `MUL_MAT ssm_beta:bf16:2560x48` | 0.51, 0.48 | 0.58, 0.54 | 0.08, 0.06 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 2.09, 2.09 | 2.15, 2.14 | 0.06, 0.05 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 5.13, 5.12 | 5.16, 5.16 | 0.03, 0.03 |
| `SIGMOID beta_sigmoid` | 0.37, 0.34 | 0.30, 0.34 | -0.07, 0.01 |
| `MUL+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD+ADD ffn_moe_weighted` | 0.39, 0.35 | 0.42, 0.38 | 0.04, 0.02 |
| `SWIGLU ffn_swiglu` | 0.05, 0.05 | 0.06, 0.10 | 0.01, 0.05 |
| `MUL_MAT attn_q:q3_K:2560x12288` | 2.02, 2.02 | 2.04, 2.04 | 0.02, 0.02 |
| every label (the pass) | 117.50, 117.45 | 132.38, 132.37 | 14.88, 14.92 |

## 4b. Depth 0: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.10, 3.11 | 8.72, 8.73 | 5.63, 5.62 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.39, 4.39 | 2.49, 2.49 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.83, 1.83 | 4.27, 4.27 | 2.44, 2.44 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.42, 2.42 | 4.67, 4.67 | 2.25, 2.26 |
| `GATED_DELTA_NET+CPY node` | 8.99, 8.94 | 11.04, 11.04 | 2.05, 2.10 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.16, 11.16 | 13.20, 13.20 | 2.05, 2.04 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.47, 2.48 | 3.92, 3.92 | 1.44, 1.45 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.63, 2.66 | 4.05, 4.05 | 1.42, 1.39 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.56, 1.54 | 2.94, 2.94 | 1.38, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.19, 7.19 | 8.20, 8.20 | 1.01, 1.01 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 2.58, 2.58 | 3.57, 3.57 | 0.99, 0.99 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.61, 2.62 | 3.58, 3.58 | 0.97, 0.96 |
| every label (the pass) | 83.37, 83.51 | 117.50, 117.45 | 34.14, 33.94 |

## 4b. Depth 4,096: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.11 | 8.71, 8.71 | 5.60, 5.59 |
| `FLASH_ATTN_EXT node` | 10.15, 10.17 | 14.61, 14.61 | 4.46, 4.45 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.03, 11.05 | 13.57, 13.57 | 2.53, 2.52 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.40, 4.40 | 2.50, 2.50 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.83, 1.85 | 4.27, 4.27 | 2.44, 2.41 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.67, 4.68 | 2.24, 2.25 |
| `GATED_DELTA_NET+CPY node` | 8.99, 8.95 | 11.06, 11.05 | 2.08, 2.10 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.48, 2.49 | 3.92, 3.93 | 1.44, 1.44 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.63, 2.68 | 4.07, 4.06 | 1.44, 1.38 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.94, 2.94 | 1.39, 1.40 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.13, 7.12 | 8.42, 8.42 | 1.29, 1.29 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.60, 2.60 | 3.59, 3.61 | 0.99, 1.01 |
| every label (the pass) | 93.34, 93.55 | 132.38, 132.37 | 39.04, 38.82 |

## 4b. Depth 8,192: what grows from 4 tokens to 5, ms a pass (loads paired in order: w4g1 → w5g1, w4g2 → w5g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 12.69, 12.68 | 27.55, 27.56 | 14.86, 14.88 |
| `MUL_MAT output:q5_K:2560x248320` | 3.11, 3.11 | 8.72, 8.72 | 5.61, 5.60 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 11.38, 11.37 | 13.95, 13.95 | 2.57, 2.58 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.90, 1.90 | 4.40, 4.40 | 2.50, 2.50 |
| `MUL_MAT attn_qkv:iq4_xs:2560x10240` | 1.85, 1.85 | 4.27, 4.27 | 2.42, 2.42 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 2.43, 2.43 | 4.67, 4.67 | 2.24, 2.24 |
| `GATED_DELTA_NET+CPY node` | 9.02, 8.96 | 11.06, 11.06 | 2.04, 2.10 |
| `MUL_MAT hc_ffn_down:bf16:10240x320` | 2.47, 2.48 | 3.92, 3.92 | 1.45, 1.44 |
| `MUL_MAT hc_attn_down:bf16:10240x320` | 2.62, 2.67 | 4.06, 4.04 | 1.43, 1.36 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 1.55, 1.54 | 2.94, 2.94 | 1.39, 1.39 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 7.36, 7.35 | 8.68, 8.69 | 1.32, 1.33 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 2.60, 2.59 | 3.59, 3.61 | 0.98, 1.02 |
| every label (the pass) | 97.06, 97.20 | 146.69, 146.70 | 49.63, 49.49 |

## 4c. Depth 0: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.61 | 11.16, 11.16 | 7.54, 7.55 |
| `GATED_DELTA_NET+CPY node` | 2.65, 2.65 | 8.99, 8.94 | 6.33, 6.29 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.00, 3.01 | 7.19, 7.19 | 4.19, 4.18 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.22, 3.22 | 4.60, 4.59 | 1.38, 1.37 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.73, 1.73 | 1.17, 1.17 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.88, 2.88 | 3.94, 3.94 | 1.07, 1.06 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.43, 0.41 | 1.27, 1.27 | 0.85, 0.86 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.56, 1.54 | 0.82, 0.81 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.97, 1.97 | 2.61, 2.62 | 0.64, 0.65 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.95 | 2.58, 2.58 | 0.63, 0.63 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.90, 1.90 | 0.61, 0.60 |
| `MUL_MAT ffn_gate_inp:f32:2560x512` | 1.82, 1.82 | 2.42, 2.42 | 0.60, 0.59 |
| every label (the pass) | 48.84, 48.87 | 83.37, 83.51 | 34.53, 34.65 |

## 4c. Depth 4,096: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.64 | 11.03, 11.05 | 7.42, 7.41 |
| `FLASH_ATTN_EXT node` | 2.89, 2.89 | 10.15, 10.17 | 7.25, 7.28 |
| `GATED_DELTA_NET+CPY node` | 2.65, 2.65 | 8.99, 8.95 | 6.34, 6.30 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.01 | 7.13, 7.12 | 4.11, 4.12 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.23, 3.25 | 4.63, 4.63 | 1.41, 1.38 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.72, 1.72 | 1.16, 1.16 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.87, 2.88 | 3.97, 3.97 | 1.10, 1.09 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.41, 0.43 | 1.27, 1.28 | 0.86, 0.84 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.82, 0.81 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.97 | 2.59, 2.59 | 0.63, 0.62 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.28 | 1.90, 1.90 | 0.61, 0.61 |
| `MUL_MAT hc_attn_up:bf16:320x10240` | 1.99, 2.00 | 2.60, 2.60 | 0.61, 0.60 |
| every label (the pass) | 51.49, 51.63 | 93.34, 93.55 | 41.85, 41.92 |

## 4c. Depth 8,192: from 1 token to 4, ms a pass (loads paired in order: w1g1 → w4g1, w1g2 → w4g2; request s)

| launch group (label) | before | after | after − before |
|---|---:|---:|---:|
| `FLASH_ATTN_EXT node` | 2.92, 2.93 | 12.69, 12.68 | 9.76, 9.75 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640` | 3.61, 3.62 | 11.38, 11.37 | 7.77, 7.75 |
| `GATED_DELTA_NET+CPY node` | 2.65, 2.65 | 9.02, 8.96 | 6.37, 6.30 |
| `MUL_MAT_ID ffn_down_exps:q2_0:640x2560` | 3.01, 3.01 | 7.36, 7.35 | 4.35, 4.34 |
| `MUL_MAT attn_qkv:q3_K:2560x10240` | 3.23, 3.23 | 4.63, 4.63 | 1.40, 1.41 |
| `MUL_MAT_ID+MUL_MAT_ID+SWIGLU+MUL_MAT+MUL_MAT+SWIGLU ffn_gate_exps:q2_0:2560x640 ffn_up_exps:q2_0:2560x640 ffn_gate_shexp:q2_0:2560x640 ffn_up_shexp:q2_0:2560x640` | 0.56, 0.56 | 1.79, 1.79 | 1.23, 1.23 |
| `MUL_MAT attn_gate:q3_K:2560x6144` | 2.87, 2.88 | 3.97, 3.97 | 1.09, 1.09 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | 0.41, 0.41 | 1.26, 1.28 | 0.85, 0.87 |
| `MUL_MAT ssm_out:q4_K:6144x2560` | 0.73, 0.73 | 1.55, 1.54 | 0.81, 0.81 |
| `LIGHTNING_INDEXER indexer_score` | 0.43, 0.43 | 1.15, 1.15 | 0.72, 0.72 |
| `MUL_MAT hc_ffn_up:bf16:320x10240` | 1.95, 1.95 | 2.59, 2.58 | 0.65, 0.63 |
| `MUL_MAT ssm_out:iq4_xs:6144x2560` | 1.29, 1.29 | 1.90, 1.90 | 0.61, 0.61 |
| every label (the pass) | 51.67, 51.78 | 97.06, 97.20 | 45.39, 45.43 |

