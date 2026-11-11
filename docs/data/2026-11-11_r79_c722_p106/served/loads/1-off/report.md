2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.89 | 160.59 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.85 | 155.36 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.54 | 19.52 | -0.02 |
| shared expert | 4.62 | 4.63 | +0.01 |
| hyper-connections (BF16) | 12.42 | 12.38 | -0.03 |
| router | 2.30 | 2.30 | -0.01 |
| attention and recurrent projections | 32.73 | 32.70 | -0.03 |
| Gated DeltaNet state and conv | 8.66 | 8.67 | +0.01 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.12 | 6.19 | +0.07 |
| other weights | 0.73 | 0.74 | +0.01 |
| **sum** | **88.09** | **88.07** | **-0.02** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert | 0.28 | 0.28 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.57 | +0.01 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.44 | 1.43 | -0.01 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.13 | 8.14 | +0.00 |
| other ops without weights | 1.07 | 1.09 | +0.02 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **12.94** | **12.97** | **+0.02** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.41 | 20.39 | -0.02 |
| shared expert | 4.90 | 4.91 | +0.01 |
| hyper-connections (BF16) | 12.98 | 12.95 | -0.03 |
| router | 2.40 | 2.40 | -0.01 |
| attention and recurrent projections | 34.17 | 34.13 | -0.04 |
| Gated DeltaNet state and conv | 8.92 | 8.92 | +0.01 |
| attention (FLASH_ATTN_EXT) | 1.04 | 1.03 | -0.01 |
| output layer | 8.13 | 8.14 | +0.00 |
| other ops without weights | 7.19 | 7.28 | +0.08 |
| other weights | 0.89 | 0.89 | +0.01 |
| **sum** | **101.03** | **101.03** | **+0.00** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.66 | 8.67 | +0.01 |
| GDN_CONV | 0.74 | 0.78 | +0.04 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | -0.00 |
| routed experts | 19.54 | 19.52 | -0.02 |
| shared expert, router, hyper-connections | 19.34 | 19.31 | -0.04 |
| everything else | 39.73 | 39.72 | -0.01 |
| **sum** | **88.09** | **88.07** | **-0.02** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.13 | 8.14 | +0.00 |
| the sampling chain after the product | 0.65 | 0.66 | +0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert, router, hyper-connections | 0.94 | 0.95 | +0.01 |
| everything else | 1.97 | 1.98 | +0.01 |
| **sum** | **12.94** | **12.97** | **+0.02** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 80 of 80, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.16 | 4.18 | +0.02 |
| the head's product | 8.13 | 8.14 | +0.00 |
| after the head's product | 0.65 | 0.66 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 51.03 (737) | 50.98 (737) | -0.05 |
| GDN_CONV | 0.74 (35) | 0.78 (35) | +0.04 |
| SIGMOID | 0.24 (93) | 0.28 (93) | +0.04 |
| SILU | 0.21 (93) | 0.18 (93) | -0.04 |
| MUL | 0.82 (134) | 0.79 (134) | -0.03 |
| MUL_MAT_ID | 19.54 (92) | 19.52 (92) | -0.02 |
| ARANGE | 0.00 (1) | 0.02 (1) | +0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| FILL | 0.05 (33) | 0.06 (33) | +0.01 |
| SCALE | 1.47 (197) | 1.48 (197) | +0.01 |
| REPEAT | 0.02 (2) | 0.03 (2) | +0.01 |
| DSV4_HC_PRE | 0.51 (92) | 0.50 (92) | -0.01 |
| GATED_DELTA_NET | 8.66 (35) | 8.67 (35) | +0.01 |
| ADD | 0.37 (119) | 0.36 (119) | -0.01 |
| the other 18 ops | 3.46 | 3.47 | +0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.08 (5) | 0.10 (5) | +0.02 |
| SIGMOID | 0.03 (5) | 0.03 (5) | -0.01 |
| TOP_K | 0.31 (5) | 0.31 (5) | +0.00 |
| GET_ROWS | 0.08 (27) | 0.08 (27) | -0.00 |
| SCALE | 0.09 (19) | 0.08 (19) | -0.00 |
| ARGMAX | 0.01 (8) | 0.01 (8) | +0.00 |
| STEP | 0.00 (4) | 0.01 (4) | +0.00 |
| MUL_MAT_ID | 0.86 (4) | 0.86 (4) | +0.00 |
| DSV4_HC_PRE | 0.02 (4) | 0.02 (4) | +0.00 |
| LOG | 0.01 (4) | 0.00 (4) | -0.00 |
| ARGSORT | 0.00 (4) | 0.01 (4) | +0.00 |
| SUM | 0.01 (4) | 0.01 (4) | -0.00 |
| SOFT_MAX | 0.03 (6) | 0.03 (6) | +0.00 |
| RMS_NORM | 0.05 (10) | 0.06 (10) | +0.00 |
| the other 19 ops | 11.36 | 11.35 | -0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.136 |
| 118 | `PAD node_6219` | 0.218 |
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
| 137 | `GET_ROWS node_6251` | 0.003 |
| 138 | `ARGMAX greedy_argmax` | 0.003 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 140 | `TOP_K top_k` | 0.073 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.001 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.001 |
| 148 | `STEP top_p_mask` | 0.001 |
| 149 | `SUM top_p_index_f32` | 0.000 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 154 | `LOG top_p_bias` | 0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.001 |
| 159 | `ARGMAX greedy_argmax` | 0.001 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.000 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | 0.000 |
| 169 | `STEP top_p_mask` | 0.002 |
| 170 | `SUM top_p_index_f32` | 0.002 |
| 171 | `CLAMP node_6308` | 0.000 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.001 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 175 | `LOG top_p_bias` | 0.001 |
| 176 | `ADD top_p_logits` | 0.003 |
| 177 | `ARGMAX temp_max_idx` | 0.000 |
| 178 | `GET_ROWS node_6318` | 0.004 |
| 179 | `GET_ROWS node_6321` | 0.005 |
| 180 | `ARGMAX greedy_argmax` | 0.007 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.007 |
| 182 | `TOP_K top_k` | 0.078 |
| 183 | `GET_ROWS top_k_rows` | 0.006 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.002 |
| 185 | `GET_ROWS top_p_candidates` | 0.002 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.002 |
| 189 | `SCALE top_p_cdf_scaled` | 0.000 |
| 190 | `STEP top_p_mask` | 0.001 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | 0.001 |
| 193 | `SCALE top_p_ones` | -0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | -0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.001 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.41 | 12.40 | -0.01 |
| 1 | 11.18 | 11.16 | -0.02 |
| 2 | 11.35 | 11.34 | -0.00 |
| 3 | 11.48 | 11.48 | -0.00 |
| 4 | 9.25 | 9.24 | -0.01 |
| 5 | 11.41 | 11.42 | +0.00 |
| 6 | 9.57 | 9.57 | -0.00 |
| 7 | 11.44 | 11.46 | +0.02 |
| 8 | 12.94 | 12.97 | +0.02 |
| **all** | **101.03** | **101.03** | **+0.00** |

