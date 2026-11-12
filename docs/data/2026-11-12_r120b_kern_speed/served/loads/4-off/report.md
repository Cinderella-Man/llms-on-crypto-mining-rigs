2848 evaluations on cards 0 to 8: the model's graph 0: 170, 1: 170, 2: 170, 3: 170, 4: 170, 5: 170, 6: 170, 7: 170, 8: 170; the draft's graphs 1318 (card 4: 659, card 8: 659), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 85, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 85 | 3 | 81 | 1: 1, 4: 80 | 77 | 105.61 | 143.52 |
| r2 | 84 | 0 | 83 | 1: 1, 4: 82 | 79 | 105.74 | 143.33 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.62 | 19.66 | +0.04 |
| shared expert | 4.34 | 4.35 | +0.01 |
| hyper-connections (BF16) | 12.42 | 12.39 | -0.02 |
| router | 2.30 | 2.31 | +0.01 |
| attention and recurrent projections | 25.24 | 25.25 | +0.00 |
| Gated DeltaNet state and conv | 8.69 | 8.70 | +0.01 |
| attention (FLASH_ATTN_EXT) | 0.97 | 0.96 | -0.00 |
| other ops without weights | 6.08 | 6.12 | +0.04 |
| other weights | 0.75 | 0.75 | -0.00 |
| **sum** | **80.42** | **80.49** | **+0.08** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.88 | 0.88 | +0.00 |
| shared expert | 0.24 | 0.24 | +0.00 |
| hyper-connections (BF16) | 0.55 | 0.55 | -0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 0.81 | 0.81 | -0.00 |
| Gated DeltaNet state and conv | 0.26 | 0.26 | +0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 3.15 | 3.15 | +0.00 |
| other ops without weights | 1.11 | 1.11 | +0.00 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **7.34** | **7.35** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.50 | 20.54 | +0.04 |
| shared expert | 4.58 | 4.59 | +0.01 |
| hyper-connections (BF16) | 12.97 | 12.95 | -0.02 |
| router | 2.40 | 2.41 | +0.01 |
| attention and recurrent projections | 26.05 | 26.06 | +0.00 |
| Gated DeltaNet state and conv | 8.95 | 8.96 | +0.01 |
| attention (FLASH_ATTN_EXT) | 1.06 | 1.05 | -0.00 |
| output layer | 3.15 | 3.15 | +0.00 |
| other ops without weights | 7.19 | 7.23 | +0.04 |
| other weights | 0.90 | 0.90 | -0.00 |
| **sum** | **87.76** | **87.84** | **+0.08** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.69 | 8.70 | +0.01 |
| GDN_CONV | 0.74 | 0.75 | +0.01 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.62 | 19.66 | +0.04 |
| shared expert, router, hyper-connections | 19.06 | 19.06 | -0.00 |
| everything else | 32.24 | 32.26 | +0.02 |
| **sum** | **80.42** | **80.49** | **+0.08** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 3.15 | 3.15 | +0.00 |
| the sampling chain after the product | 0.73 | 0.72 | -0.01 |
| the output's hyper-connection products | 0.10 | 0.10 | +0.00 |
| GATED_DELTA_NET | 0.26 | 0.26 | +0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.88 | 0.88 | +0.00 |
| shared expert, router, hyper-connections | 0.89 | 0.89 | -0.00 |
| everything else | 1.31 | 1.31 | +0.01 |
| **sum** | **7.34** | **7.35** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 77 of 77, 204 launch groups, verify-r2 79 of 79, 204 launch groups)

| Position | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 3.46 | 3.48 | +0.01 |
| the head's product | 3.15 | 3.15 | +0.00 |
| after the head's product | 0.73 | 0.72 | -0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.83 (134) | 0.78 (134) | -0.04 |
| MUL_MAT_ID | 19.62 (92) | 19.66 (92) | +0.04 |
| SOFTPLUS | 0.15 (35) | 0.18 (35) | +0.03 |
| SIGMOID | 0.24 (93) | 0.26 (93) | +0.02 |
| REPEAT | 0.03 (2) | 0.05 (2) | +0.01 |
| GDN_CONV | 0.74 (35) | 0.75 (35) | +0.01 |
| ADD | 0.38 (119) | 0.39 (119) | +0.01 |
| RMS_NORM | 0.95 (174) | 0.96 (174) | +0.01 |
| SCALE | 1.47 (197) | 1.48 (197) | +0.01 |
| MUL_MAT | 43.26 (737) | 43.27 (737) | +0.01 |
| CONCAT | 0.14 (12) | 0.13 (12) | -0.01 |
| GATED_DELTA_NET | 8.69 (35) | 8.70 (35) | +0.01 |
| SOFT_MAX | 0.51 (46) | 0.50 (46) | -0.00 |
| CONT | 0.53 (31) | 0.53 (31) | -0.00 |
| the other 18 ops | 2.89 | 2.86 | -0.02 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SCALE | 0.09 (19) | 0.08 (19) | -0.01 |
| SOFTPLUS | 0.02 (1) | 0.01 (1) | -0.01 |
| SIGMOID | 0.04 (5) | 0.05 (5) | +0.01 |
| CPY | 0.04 (5) | 0.03 (5) | -0.01 |
| ARGMAX | 0.02 (8) | 0.03 (8) | +0.01 |
| LOG | 0.01 (4) | 0.02 (4) | +0.01 |
| GET_ROWS | 0.08 (27) | 0.09 (27) | +0.01 |
| SUM | 0.01 (4) | 0.01 (4) | -0.01 |
| MUL | 0.06 (5) | 0.07 (5) | +0.00 |
| ADD | 0.03 (13) | 0.03 (13) | +0.00 |
| CLAMP | 0.01 (4) | 0.01 (4) | -0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | -0.00 |
| GATED_DELTA_NET | 0.26 (1) | 0.26 (1) | +0.00 |
| SWIGLU | 0.00 (2) | 0.01 (2) | +0.00 |
| the other 19 ops | 6.35 | 6.35 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.000 |
| 115 | `ADD node_6216` | 0.000 |
| 116 | `SCALE result_norm` | 0.000 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 3.152 |
| 118 | `PAD node_6219` | 0.216 |
| 119 | `TOP_K top_k` | 0.074 |
| 120 | `GET_ROWS top_k_rows` | 0.004 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 122 | `GET_ROWS top_p_candidates` | 0.003 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 124 | `SOFT_MAX top_p_softmax` | 0.004 |
| 125 | `CUMSUM top_p_cdf` | 0.005 |
| 126 | `SCALE top_p_cdf_scaled` | 0.002 |
| 127 | `STEP top_p_mask` | 0.004 |
| 128 | `SUM top_p_index_f32` | 0.005 |
| 129 | `CLAMP node_6238` | 0.002 |
| 130 | `SCALE top_p_ones` | 0.009 |
| 131 | `CPY _(copy)` | 0.007 |
| 132 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.009 |
| 133 | `LOG top_p_bias` | 0.014 |
| 134 | `ADD top_p_logits` | 0.011 |
| 135 | `ARGMAX temp_max_idx` | 0.016 |
| 136 | `GET_ROWS node_6248` | 0.013 |
| 137 | `GET_ROWS node_6251` | 0.010 |
| 138 | `ARGMAX greedy_argmax` | 0.005 |
| 139 | `GET_ROWS greedy_sampled_token` | 0.011 |
| 140 | `TOP_K top_k` | 0.078 |
| 141 | `GET_ROWS top_k_rows` | 0.007 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 143 | `GET_ROWS top_p_candidates` | 0.003 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.003 |
| 145 | `SOFT_MAX top_p_softmax` | 0.003 |
| 146 | `CUMSUM top_p_cdf` | 0.003 |
| 147 | `SCALE top_p_cdf_scaled` | 0.003 |
| 148 | `STEP top_p_mask` | 0.003 |
| 149 | `SUM top_p_index_f32` | 0.003 |
| 150 | `CLAMP node_6273` | 0.003 |
| 151 | `SCALE top_p_ones` | 0.002 |
| 152 | `CPY _(copy)` | 0.003 |
| 153 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 154 | `LOG top_p_bias` | 0.003 |
| 155 | `ADD top_p_logits` | 0.002 |
| 156 | `ARGMAX temp_max_idx` | 0.003 |
| 157 | `GET_ROWS node_6283` | 0.003 |
| 158 | `GET_ROWS node_6286` | 0.003 |
| 159 | `ARGMAX greedy_argmax` | 0.002 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.003 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | 0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.001 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.000 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | -0.000 |
| 169 | `STEP top_p_mask` | 0.000 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6308` | 0.000 |
| 172 | `SCALE top_p_ones` | -0.000 |
| 173 | `CPY _(copy)` | 0.000 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 175 | `LOG top_p_bias` | 0.000 |
| 176 | `ADD top_p_logits` | 0.000 |
| 177 | `ARGMAX temp_max_idx` | 0.000 |
| 178 | `GET_ROWS node_6318` | 0.001 |
| 179 | `GET_ROWS node_6321` | 0.001 |
| 180 | `ARGMAX greedy_argmax` | -0.000 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 182 | `TOP_K top_k` | 0.071 |
| 183 | `GET_ROWS top_k_rows` | 0.000 |
| 184 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 185 | `GET_ROWS top_p_candidates` | 0.001 |
| 186 | `GET_ROWS top_p_sorted_logits` | -0.000 |
| 187 | `SOFT_MAX top_p_softmax` | 0.000 |
| 188 | `CUMSUM top_p_cdf` | 0.000 |
| 189 | `SCALE top_p_cdf_scaled` | -0.000 |
| 190 | `STEP top_p_mask` | -0.000 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | 0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 196 | `LOG top_p_bias` | 0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | -0.000 |
| 199 | `GET_ROWS node_6353` | 0.001 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | -0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (77 passes) | verify-r2 (79 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 10.86 | 10.93 | +0.06 |
| 1 | 10.81 | 10.81 | +0.00 |
| 2 | 10.68 | 10.70 | +0.01 |
| 3 | 10.74 | 10.73 | -0.01 |
| 4 | 8.47 | 8.47 | +0.00 |
| 5 | 10.18 | 10.18 | +0.00 |
| 6 | 8.68 | 8.68 | +0.01 |
| 7 | 9.99 | 9.99 | -0.01 |
| 8 | 7.34 | 7.35 | +0.01 |
| **all** | **87.76** | **87.84** | **+0.08** |

