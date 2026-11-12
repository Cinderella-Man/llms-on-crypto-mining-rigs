3115 evaluations on cards 0 to 8: the model's graph 0: 185, 1: 185, 2: 185, 3: 185, 4: 185, 5: 185, 6: 185, 7: 185, 8: 185; the draft's graphs 1450 (card 4: 725, card 8: 725), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 94, 90; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 94 | 3 | 90 | 4: 90 | 87 | 118.35 | 159.96 |
| r2 | 90 | 0 | 89 | 4: 89 | 86 | 118.22 | 155.11 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.49 | 19.45 | -0.05 |
| shared expert | 4.29 | 4.27 | -0.02 |
| hyper-connections (BF16) | 12.41 | 12.43 | +0.02 |
| router | 2.30 | 2.30 | -0.00 |
| attention and recurrent projections | 32.67 | 32.64 | -0.03 |
| Gated DeltaNet state and conv | 8.64 | 8.65 | +0.01 |
| attention (FLASH_ATTN_EXT) | 0.93 | 0.93 | -0.00 |
| other ops without weights | 6.09 | 6.11 | +0.02 |
| other weights | 0.73 | 0.73 | -0.00 |
| **sum** | **87.56** | **87.49** | **-0.06** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.26 | 0.26 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | -0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.42 | 1.42 | -0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.10 | 8.10 | -0.00 |
| other ops without weights | 1.07 | 1.08 | +0.01 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.88** | **12.88** | **-0.00** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.36 | 20.31 | -0.05 |
| shared expert | 4.55 | 4.53 | -0.02 |
| hyper-connections (BF16) | 12.98 | 12.99 | +0.01 |
| router | 2.40 | 2.40 | -0.00 |
| attention and recurrent projections | 34.09 | 34.06 | -0.03 |
| Gated DeltaNet state and conv | 8.89 | 8.90 | +0.01 |
| attention (FLASH_ATTN_EXT) | 1.02 | 1.01 | -0.00 |
| output layer | 8.10 | 8.10 | -0.00 |
| other ops without weights | 7.17 | 7.19 | +0.02 |
| other weights | 0.88 | 0.88 | -0.00 |
| **sum** | **100.44** | **100.38** | **-0.06** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.64 | 8.65 | +0.01 |
| GDN_CONV | 0.72 | 0.72 | +0.00 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.49 | 19.45 | -0.05 |
| shared expert, router, hyper-connections | 19.00 | 19.00 | -0.00 |
| everything else | 39.64 | 39.62 | -0.02 |
| **sum** | **87.56** | **87.49** | **-0.06** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.10 | 8.10 | -0.00 |
| the sampling chain after the product | 0.65 | 0.65 | +0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.92 | 0.93 | +0.00 |
| everything else | 1.96 | 1.96 | -0.00 |
| **sum** | **12.88** | **12.88** | **-0.00** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 87 of 87, 204 launch groups, verify-r2 86 of 86, 204 launch groups)

| Position | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.13 | 4.13 | +0.00 |
| the head's product | 8.10 | 8.10 | -0.00 |
| after the head's product | 0.65 | 0.65 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT_ID | 19.49 (92) | 19.45 (92) | -0.05 |
| MUL_MAT | 50.64 (737) | 50.60 (737) | -0.04 |
| RMS_NORM | 0.98 (174) | 1.01 (174) | +0.04 |
| ADD | 0.38 (119) | 0.35 (119) | -0.02 |
| SOFTPLUS | 0.17 (35) | 0.19 (35) | +0.02 |
| MUL | 0.87 (134) | 0.85 (134) | -0.01 |
| CONCAT | 0.13 (12) | 0.14 (12) | +0.01 |
| SCALE | 1.46 (197) | 1.45 (197) | -0.01 |
| SILU | 0.19 (93) | 0.20 (93) | +0.01 |
| ROPE | 0.11 (44) | 0.10 (44) | -0.01 |
| TOP_K_MASK | 0.08 (11) | 0.08 (11) | -0.01 |
| GATED_DELTA_NET | 8.64 (35) | 8.65 (35) | +0.01 |
| SIGMOID | 0.22 (93) | 0.23 (93) | +0.01 |
| TOP_K | 0.13 (11) | 0.13 (11) | -0.00 |
| the other 18 ops | 4.06 | 4.06 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 10.53 (36) | 10.52 (36) | -0.01 |
| MUL | 0.08 (5) | 0.08 (5) | -0.01 |
| SIGMOID | 0.03 (5) | 0.03 (5) | +0.01 |
| LOG | 0.00 (4) | 0.01 (4) | +0.00 |
| SOFTPLUS | 0.02 (1) | 0.03 (1) | +0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | -0.00 |
| CUMSUM | 0.01 (4) | 0.01 (4) | +0.00 |
| SUM | 0.01 (4) | 0.00 (4) | -0.00 |
| ADD | 0.04 (13) | 0.04 (13) | +0.00 |
| STEP | 0.00 (4) | 0.01 (4) | +0.00 |
| GATED_DELTA_NET | 0.25 (1) | 0.25 (1) | +0.00 |
| ARGMAX | 0.02 (8) | 0.02 (8) | -0.00 |
| SILU | 0.01 (5) | 0.01 (5) | -0.00 |
| SCALE | 0.08 (19) | 0.08 (19) | +0.00 |
| the other 19 ops | 1.50 | 1.50 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.100 |
| 118 | `PAD node_6219` | 0.216 |
| 119 | `TOP_K top_k` | 0.076 |
| 120 | `GET_ROWS top_k_rows` | 0.004 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 122 | `GET_ROWS top_p_candidates` | 0.005 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 124 | `SOFT_MAX top_p_softmax` | 0.004 |
| 125 | `CUMSUM top_p_cdf` | 0.003 |
| 126 | `SCALE top_p_cdf_scaled` | 0.003 |
| 127 | `STEP top_p_mask` | 0.003 |
| 128 | `SUM top_p_index_f32` | 0.003 |
| 129 | `CLAMP node_6238` | 0.002 |
| 130 | `SCALE top_p_ones` | 0.003 |
| 131 | `CPY _(copy)` | 0.003 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 133 | `LOG top_p_bias` | 0.002 |
| 134 | `ADD top_p_logits` | 0.003 |
| 135 | `ARGMAX temp_max_idx` | 0.003 |
| 136 | `GET_ROWS node_6248` | 0.003 |
| 137 | `GET_ROWS node_6251` | 0.002 |
| 138 | `ARGMAX greedy_argmax` | 0.002 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.002 |
| 140 | `TOP_K top_k` | 0.072 |
| 141 | `GET_ROWS top_k_rows` | 0.001 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.001 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.001 |
| 148 | `STEP top_p_mask` | 0.000 |
| 149 | `SUM top_p_index_f32` | 0.000 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 154 | `LOG top_p_bias` | 0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.001 |
| 159 | `ARGMAX greedy_argmax` | 0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.000 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 166 | `SOFT_MAX top_p_softmax` | 0.001 |
| 167 | `CUMSUM top_p_cdf` | 0.001 |
| 168 | `SCALE top_p_cdf_scaled` | 0.001 |
| 169 | `STEP top_p_mask` | 0.001 |
| 170 | `SUM top_p_index_f32` | 0.001 |
| 171 | `CLAMP node_6308` | 0.000 |
| 172 | `SCALE top_p_ones` | 0.001 |
| 173 | `CPY _(copy)` | 0.002 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 175 | `LOG top_p_bias` | 0.004 |
| 176 | `ADD top_p_logits` | 0.002 |
| 177 | `ARGMAX temp_max_idx` | 0.002 |
| 178 | `GET_ROWS node_6318` | 0.005 |
| 179 | `GET_ROWS node_6321` | 0.008 |
| 180 | `ARGMAX greedy_argmax` | 0.008 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.009 |
| 182 | `TOP_K top_k` | 0.075 |
| 183 | `GET_ROWS top_k_rows` | 0.004 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 185 | `GET_ROWS top_p_candidates` | 0.002 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.002 |
| 189 | `SCALE top_p_cdf_scaled` | 0.001 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | -0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.001 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.33 | 12.33 | -0.00 |
| 1 | 11.11 | 11.09 | -0.02 |
| 2 | 11.26 | 11.25 | -0.01 |
| 3 | 11.39 | 11.39 | +0.00 |
| 4 | 9.21 | 9.21 | -0.01 |
| 5 | 11.35 | 11.34 | -0.01 |
| 6 | 9.49 | 9.49 | -0.00 |
| 7 | 11.40 | 11.39 | -0.01 |
| 8 | 12.88 | 12.88 | -0.00 |
| **all** | **100.44** | **100.38** | **-0.06** |

