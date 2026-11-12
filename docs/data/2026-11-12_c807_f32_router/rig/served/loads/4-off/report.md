2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.66 | 156.18 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.71 | 157.74 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.52 | 19.51 | -0.01 |
| shared expert | 4.68 | 4.66 | -0.02 |
| hyper-connections (BF16) | 12.37 | 12.41 | +0.04 |
| router | 2.30 | 2.30 | +0.00 |
| attention and recurrent projections | 32.63 | 32.61 | -0.02 |
| Gated DeltaNet state and conv | 8.65 | 8.64 | -0.01 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.12 | 6.13 | +0.01 |
| other weights | 0.73 | 0.74 | +0.01 |
| **sum** | **87.94** | **87.93** | **-0.01** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert | 0.28 | 0.28 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | +0.00 |
| router | 0.10 | 0.10 | -0.00 |
| attention and recurrent projections | 1.42 | 1.41 | -0.01 |
| Gated DeltaNet state and conv | 0.26 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.10 | 8.10 | +0.01 |
| other ops without weights | 1.08 | 1.08 | +0.01 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.88** | **12.90** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.38 | 20.37 | -0.01 |
| shared expert | 4.96 | 4.93 | -0.02 |
| hyper-connections (BF16) | 12.93 | 12.97 | +0.04 |
| router | 2.40 | 2.40 | +0.00 |
| attention and recurrent projections | 34.05 | 34.02 | -0.03 |
| Gated DeltaNet state and conv | 8.90 | 8.90 | -0.00 |
| attention (FLASH_ATTN_EXT) | 1.03 | 1.03 | -0.01 |
| output layer | 8.10 | 8.10 | +0.01 |
| other ops without weights | 7.20 | 7.21 | +0.02 |
| other weights | 0.88 | 0.89 | +0.01 |
| **sum** | **100.83** | **100.83** | **+0.00** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.65 | 8.64 | -0.01 |
| GDN_CONV | 0.75 | 0.73 | -0.02 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.52 | 19.51 | -0.01 |
| shared expert, router, hyper-connections | 19.35 | 19.36 | +0.01 |
| everything else | 39.61 | 39.62 | +0.01 |
| **sum** | **87.94** | **87.93** | **-0.01** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.10 | 8.10 | +0.01 |
| the sampling chain after the product | 0.65 | 0.65 | -0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.26 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert, router, hyper-connections | 0.94 | 0.94 | +0.00 |
| everything else | 1.96 | 1.96 | -0.00 |
| **sum** | **12.88** | **12.90** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 80 of 80, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.13 | 4.14 | +0.01 |
| the head's product | 8.10 | 8.10 | +0.01 |
| after the head's product | 0.65 | 0.65 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| ADD | 0.34 (119) | 0.36 (119) | +0.03 |
| SCALE | 1.43 (197) | 1.45 (197) | +0.02 |
| SIGMOID | 0.26 (93) | 0.24 (93) | -0.02 |
| GDN_CONV | 0.75 (35) | 0.73 (35) | -0.02 |
| MUL | 0.85 (134) | 0.83 (134) | -0.01 |
| MUL_MAT_ID | 19.52 (92) | 19.51 (92) | -0.01 |
| MUL_MAT | 50.95 (737) | 50.94 (737) | -0.01 |
| SOFT_MAX | 0.51 (46) | 0.52 (46) | +0.01 |
| RMS_NORM | 0.97 (174) | 0.96 (174) | -0.01 |
| FILL | 0.06 (33) | 0.05 (33) | -0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| GET_ROWS | 0.18 (23) | 0.19 (23) | +0.01 |
| SET_ROWS | 0.15 (34) | 0.16 (34) | +0.01 |
| REPEAT | 0.02 (2) | 0.02 (2) | +0.01 |
| the other 18 ops | 11.02 | 11.02 | +0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.06 (5) | 0.07 (5) | +0.01 |
| SOFTPLUS | 0.02 (1) | 0.02 (1) | -0.01 |
| SCALE | 0.09 (19) | 0.08 (19) | -0.00 |
| SIGMOID | 0.05 (5) | 0.05 (5) | -0.00 |
| ARGSORT | 0.00 (4) | 0.01 (4) | +0.00 |
| ADD | 0.03 (13) | 0.04 (13) | +0.00 |
| GATED_DELTA_NET | 0.26 (1) | 0.26 (1) | +0.00 |
| MUL_MAT | 10.52 (36) | 10.53 (36) | +0.00 |
| SET_ROWS | 0.02 (8) | 0.02 (8) | -0.00 |
| RMS_NORM | 0.05 (10) | 0.06 (10) | +0.00 |
| MUL_MAT_ID | 0.86 (4) | 0.86 (4) | +0.00 |
| FILL | 0.01 (3) | 0.01 (3) | -0.00 |
| LOG | 0.00 (4) | 0.01 (4) | +0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | +0.00 |
| the other 19 ops | 0.59 | 0.59 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.104 |
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
| 131 | `CPY _(copy)` | 0.002 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 133 | `LOG top_p_bias` | 0.002 |
| 134 | `ADD top_p_logits` | 0.003 |
| 135 | `ARGMAX temp_max_idx` | 0.002 |
| 136 | `GET_ROWS node_6248` | 0.003 |
| 137 | `GET_ROWS node_6251` | 0.002 |
| 138 | `ARGMAX greedy_argmax` | 0.002 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.002 |
| 140 | `TOP_K top_k` | 0.072 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.002 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.000 |
| 148 | `STEP top_p_mask` | 0.000 |
| 149 | `SUM top_p_index_f32` | 0.001 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.000 |
| 152 | `CPY _(copy)` | 0.000 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 154 | `LOG top_p_bias` | 0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.000 |
| 159 | `ARGMAX greedy_argmax` | 0.002 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.002 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.003 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | 0.000 |
| 169 | `STEP top_p_mask` | 0.001 |
| 170 | `SUM top_p_index_f32` | 0.003 |
| 171 | `CLAMP node_6308` | 0.002 |
| 172 | `SCALE top_p_ones` | 0.001 |
| 173 | `CPY _(copy)` | 0.002 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.001 |
| 175 | `LOG top_p_bias` | 0.004 |
| 176 | `ADD top_p_logits` | 0.005 |
| 177 | `ARGMAX temp_max_idx` | 0.005 |
| 178 | `GET_ROWS node_6318` | 0.005 |
| 179 | `GET_ROWS node_6321` | 0.004 |
| 180 | `ARGMAX greedy_argmax` | 0.004 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.004 |
| 182 | `TOP_K top_k` | 0.076 |
| 183 | `GET_ROWS top_k_rows` | 0.001 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.005 |
| 185 | `GET_ROWS top_p_candidates` | 0.004 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 187 | `SOFT_MAX top_p_softmax` | 0.001 |
| 188 | `CUMSUM top_p_cdf` | 0.000 |
| 189 | `SCALE top_p_cdf_scaled` | 0.000 |
| 190 | `STEP top_p_mask` | 0.001 |
| 191 | `SUM top_p_index_f32` | 0.001 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | -0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | 0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.000 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.38 | 12.39 | +0.01 |
| 1 | 11.16 | 11.16 | -0.00 |
| 2 | 11.31 | 11.30 | -0.01 |
| 3 | 11.44 | 11.43 | -0.01 |
| 4 | 9.27 | 9.27 | -0.00 |
| 5 | 11.40 | 11.40 | +0.01 |
| 6 | 9.55 | 9.54 | -0.02 |
| 7 | 11.44 | 11.44 | +0.00 |
| 8 | 12.88 | 12.90 | +0.01 |
| **all** | **100.83** | **100.83** | **+0.00** |

