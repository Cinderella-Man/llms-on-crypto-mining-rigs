2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 87, 85; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 87 | 3 | 83 | 4: 83 | 80 | 113.37 | 149.59 |
| r2 | 85 | 0 | 84 | 4: 84 | 81 | 113.43 | 148.92 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.54 | 19.54 | +0.01 |
| shared expert | 4.62 | 4.63 | +0.01 |
| hyper-connections (BF16) | 12.43 | 12.41 | -0.02 |
| router | 2.30 | 2.31 | +0.01 |
| attention and recurrent projections | 32.33 | 32.24 | -0.09 |
| Gated DeltaNet state and conv | 8.68 | 8.68 | -0.00 |
| attention (FLASH_ATTN_EXT) | 0.94 | 0.94 | +0.00 |
| other ops without weights | 6.10 | 6.22 | +0.11 |
| other weights | 0.74 | 0.74 | +0.00 |
| **sum** | **87.68** | **87.71** | **+0.02** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.25 | 0.25 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.35 | 1.35 | -0.00 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | +0.00 |
| output layer | 3.15 | 3.15 | +0.00 |
| other ops without weights | 1.08 | 1.08 | +0.00 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.84** | **7.86** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.41 | 20.42 | +0.01 |
| shared expert | 4.87 | 4.88 | +0.01 |
| hyper-connections (BF16) | 12.99 | 12.97 | -0.01 |
| router | 2.40 | 2.41 | +0.01 |
| attention and recurrent projections | 33.68 | 33.58 | -0.10 |
| Gated DeltaNet state and conv | 8.93 | 8.93 | -0.00 |
| attention (FLASH_ATTN_EXT) | 1.03 | 1.03 | +0.00 |
| output layer | 3.15 | 3.15 | +0.00 |
| other ops without weights | 7.18 | 7.30 | +0.12 |
| other weights | 0.89 | 0.89 | +0.00 |
| **sum** | **95.53** | **95.56** | **+0.04** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.68 | 8.68 | -0.00 |
| GDN_CONV | 0.75 | 0.77 | +0.02 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.54 | 19.54 | +0.01 |
| shared expert, router, hyper-connections | 19.35 | 19.35 | +0.00 |
| everything else | 39.30 | 39.29 | -0.00 |
| **sum** | **87.68** | **87.71** | **+0.02** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.15 | 3.15 | +0.00 |
| the sampling chain after the product | 0.65 | 0.65 | -0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | +0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | -0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.91 | 0.91 | +0.00 |
| everything else | 1.89 | 1.89 | +0.01 |
| **sum** | **7.84** | **7.86** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 80 of 80, 204 launch groups, verify-r2 81 of 81, 204 launch groups)

| Position | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.04 | 4.06 | +0.01 |
| the head's product | 3.15 | 3.15 | +0.00 |
| after the head's product | 0.65 | 0.65 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 50.66 (737) | 50.56 (737) | -0.10 |
| SOFTPLUS | 0.15 (35) | 0.18 (35) | +0.03 |
| SIGMOID | 0.24 (93) | 0.26 (93) | +0.02 |
| MUL | 0.87 (134) | 0.84 (134) | -0.02 |
| GDN_CONV | 0.75 (35) | 0.77 (35) | +0.02 |
| SILU | 0.18 (93) | 0.20 (93) | +0.01 |
| DSV4_HC_PRE | 0.50 (92) | 0.51 (92) | +0.01 |
| FILL | 0.05 (33) | 0.06 (33) | +0.01 |
| ADD | 0.37 (119) | 0.38 (119) | +0.01 |
| GET_ROWS | 0.19 (23) | 0.18 (23) | -0.01 |
| RMS_NORM | 0.96 (174) | 0.95 (174) | -0.01 |
| SCALE | 1.46 (197) | 1.47 (197) | +0.01 |
| SET_ROWS | 0.15 (34) | 0.15 (34) | -0.01 |
| TOP_K | 0.13 (11) | 0.14 (11) | +0.01 |
| the other 18 ops | 31.01 | 31.05 | +0.03 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SIGMOID | 0.04 (5) | 0.03 (5) | -0.01 |
| MUL | 0.07 (5) | 0.08 (5) | +0.01 |
| GET_ROWS | 0.08 (27) | 0.07 (27) | -0.01 |
| SOFTPLUS | 0.03 (1) | 0.04 (1) | +0.01 |
| CPY | 0.03 (5) | 0.03 (5) | -0.00 |
| ADD | 0.03 (13) | 0.04 (13) | +0.00 |
| MUL_MAT | 5.48 (36) | 5.48 (36) | +0.00 |
| ARGSORT | 0.00 (4) | 0.01 (4) | +0.00 |
| MUL_MAT_ID | 0.87 (4) | 0.87 (4) | +0.00 |
| LOG | 0.00 (4) | 0.00 (4) | +0.00 |
| ARGMAX | 0.02 (8) | 0.01 (8) | -0.00 |
| STEP | 0.00 (4) | 0.00 (4) | +0.00 |
| SCALE | 0.08 (19) | 0.08 (19) | -0.00 |
| CLAMP | 0.00 (4) | 0.00 (4) | +0.00 |
| the other 19 ops | 1.10 | 1.10 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 3.151 |
| 118 | `PAD node_6219` | 0.217 |
| 119 | `TOP_K top_k` | 0.077 |
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
| 136 | `GET_ROWS node_6248` | 0.002 |
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
| 151 | `SCALE top_p_ones` | 0.001 |
| 152 | `CPY _(copy)` | 0.001 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 154 | `LOG top_p_bias` | -0.000 |
| 155 | `ADD top_p_logits` | 0.000 |
| 156 | `ARGMAX temp_max_idx` | 0.000 |
| 157 | `GET_ROWS node_6283` | 0.001 |
| 158 | `GET_ROWS node_6286` | 0.001 |
| 159 | `ARGMAX greedy_argmax` | 0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 161 | `TOP_K top_k` | 0.072 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.002 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.003 |
| 167 | `CUMSUM top_p_cdf` | 0.001 |
| 168 | `SCALE top_p_cdf_scaled` | 0.000 |
| 169 | `STEP top_p_mask` | 0.002 |
| 170 | `SUM top_p_index_f32` | 0.002 |
| 171 | `CLAMP node_6308` | 0.002 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.000 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.005 |
| 175 | `LOG top_p_bias` | 0.002 |
| 176 | `ADD top_p_logits` | 0.007 |
| 177 | `ARGMAX temp_max_idx` | 0.004 |
| 178 | `GET_ROWS node_6318` | 0.004 |
| 179 | `GET_ROWS node_6321` | 0.003 |
| 180 | `ARGMAX greedy_argmax` | 0.005 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.006 |
| 182 | `TOP_K top_k` | 0.075 |
| 183 | `GET_ROWS top_k_rows` | 0.004 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 185 | `GET_ROWS top_p_candidates` | 0.001 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 187 | `SOFT_MAX top_p_softmax` | 0.001 |
| 188 | `CUMSUM top_p_cdf` | 0.001 |
| 189 | `SCALE top_p_cdf_scaled` | 0.000 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | -0.000 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | 0.001 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (80 passes) | verify-r2 (81 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.34 | 12.33 | -0.00 |
| 1 | 11.16 | 11.16 | +0.00 |
| 2 | 11.24 | 11.25 | +0.00 |
| 3 | 11.48 | 11.50 | +0.02 |
| 4 | 9.18 | 9.17 | -0.00 |
| 5 | 11.36 | 11.38 | +0.01 |
| 6 | 9.45 | 9.44 | -0.00 |
| 7 | 11.48 | 11.47 | -0.00 |
| 8 | 7.84 | 7.86 | +0.01 |
| **all** | **95.53** | **95.56** | **+0.04** |

