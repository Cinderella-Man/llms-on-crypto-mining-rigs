3115 evaluations on cards 0 to 8: the model's graph 0: 185, 1: 185, 2: 185, 3: 185, 4: 185, 5: 185, 6: 185, 7: 185, 8: 185; the draft's graphs 1450 (card 4: 725, card 8: 725), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 94, 90; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 94 | 3 | 90 | 4: 90 | 87 | 118.37 | 159.21 |
| r2 | 90 | 0 | 89 | 4: 89 | 86 | 118.21 | 155.36 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.48 | 19.43 | -0.05 |
| shared expert | 4.30 | 4.30 | -0.00 |
| hyper-connections (BF16) | 12.41 | 12.40 | -0.01 |
| router | 2.30 | 2.29 | -0.01 |
| attention and recurrent projections | 32.64 | 32.65 | +0.01 |
| Gated DeltaNet state and conv | 8.64 | 8.67 | +0.02 |
| attention (FLASH_ATTN_EXT) | 0.93 | 0.93 | -0.00 |
| other ops without weights | 6.12 | 6.09 | -0.03 |
| other weights | 0.73 | 0.73 | -0.00 |
| **sum** | **87.56** | **87.48** | **-0.07** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert | 0.26 | 0.26 | -0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | -0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.42 | 1.42 | -0.00 |
| Gated DeltaNet state and conv | 0.25 | 0.25 | -0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.09 | 8.10 | +0.01 |
| other ops without weights | 1.07 | 1.08 | +0.01 |
| other weights | 0.15 | 0.15 | +0.00 |
| **sum** | **12.86** | **12.87** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.35 | 20.30 | -0.05 |
| shared expert | 4.56 | 4.55 | -0.00 |
| hyper-connections (BF16) | 12.97 | 12.96 | -0.01 |
| router | 2.40 | 2.39 | -0.01 |
| attention and recurrent projections | 34.06 | 34.07 | +0.01 |
| Gated DeltaNet state and conv | 8.90 | 8.92 | +0.02 |
| attention (FLASH_ATTN_EXT) | 1.02 | 1.02 | -0.00 |
| output layer | 8.09 | 8.10 | +0.01 |
| other ops without weights | 7.19 | 7.17 | -0.02 |
| other weights | 0.88 | 0.88 | -0.00 |
| **sum** | **100.42** | **100.36** | **-0.06** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.64 | 8.67 | +0.02 |
| GDN_CONV | 0.72 | 0.73 | +0.00 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.48 | 19.43 | -0.05 |
| shared expert, router, hyper-connections | 19.00 | 18.98 | -0.02 |
| everything else | 39.63 | 39.61 | -0.03 |
| **sum** | **87.56** | **87.48** | **-0.07** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.09 | 8.10 | +0.01 |
| the sampling chain after the product | 0.65 | 0.65 | +0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | -0.00 |
| GATED_DELTA_NET | 0.25 | 0.25 | -0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.87 | 0.87 | +0.00 |
| shared expert, router, hyper-connections | 0.93 | 0.92 | -0.00 |
| everything else | 1.96 | 1.96 | +0.00 |
| **sum** | **12.86** | **12.87** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 87 of 87, 204 launch groups, verify-r2 86 of 86, 204 launch groups)

| Position | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.13 | 4.13 | -0.00 |
| the head's product | 8.09 | 8.10 | +0.01 |
| after the head's product | 0.65 | 0.65 | +0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.88 (134) | 0.80 (134) | -0.08 |
| SIGMOID | 0.24 (93) | 0.29 (93) | +0.05 |
| MUL_MAT_ID | 19.48 (92) | 19.43 (92) | -0.05 |
| MUL_MAT | 50.62 (737) | 50.59 (737) | -0.03 |
| DSV4_HC_PRE | 0.49 (92) | 0.52 (92) | +0.03 |
| GATED_DELTA_NET | 8.64 (35) | 8.67 (35) | +0.02 |
| SCALE | 1.46 (197) | 1.45 (197) | -0.02 |
| SOFTPLUS | 0.18 (35) | 0.16 (35) | -0.02 |
| SILU | 0.19 (93) | 0.21 (93) | +0.02 |
| SOFT_MAX | 0.51 (46) | 0.51 (46) | -0.01 |
| SET_ROWS | 0.15 (34) | 0.16 (34) | +0.01 |
| TOP_K_MASK | 0.09 (11) | 0.08 (11) | -0.01 |
| SWIGLU | 0.07 (40) | 0.07 (40) | -0.01 |
| GET_ROWS | 0.20 (23) | 0.19 (23) | -0.01 |
| the other 18 ops | 4.34 | 4.36 | +0.02 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| SOFTPLUS | 0.02 (1) | 0.03 (1) | +0.01 |
| MUL_MAT | 10.51 (36) | 10.51 (36) | +0.01 |
| ADD | 0.04 (13) | 0.03 (13) | -0.01 |
| MUL | 0.08 (5) | 0.08 (5) | -0.00 |
| SIGMOID | 0.03 (5) | 0.03 (5) | -0.00 |
| SOFT_MAX | 0.03 (6) | 0.03 (6) | +0.00 |
| LOG | 0.01 (4) | 0.00 (4) | -0.00 |
| RMS_NORM | 0.05 (10) | 0.05 (10) | +0.00 |
| GET_ROWS | 0.08 (27) | 0.07 (27) | -0.00 |
| TOP_K | 0.31 (5) | 0.31 (5) | +0.00 |
| ARGSORT | 0.01 (4) | 0.00 (4) | -0.00 |
| CLAMP | 0.00 (4) | 0.00 (4) | +0.00 |
| DSV4_HC_PRE | 0.02 (4) | 0.02 (4) | -0.00 |
| GDN_CONV | 0.02 (1) | 0.02 (1) | +0.00 |
| the other 19 ops | 1.66 | 1.67 | +0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.097 |
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
| 143 | `GET_ROWS top_p_candidates` | 0.001 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.000 |
| 147 | `SCALE top_p_cdf_scaled` | 0.001 |
| 148 | `STEP top_p_mask` | 0.002 |
| 149 | `SUM top_p_index_f32` | 0.000 |
| 150 | `CLAMP node_6273` | 0.000 |
| 151 | `SCALE top_p_ones` | 0.001 |
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
| 162 | `GET_ROWS top_k_rows` | 0.001 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.001 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 166 | `SOFT_MAX top_p_softmax` | 0.002 |
| 167 | `CUMSUM top_p_cdf` | 0.001 |
| 168 | `SCALE top_p_cdf_scaled` | 0.003 |
| 169 | `STEP top_p_mask` | 0.001 |
| 170 | `SUM top_p_index_f32` | 0.004 |
| 171 | `CLAMP node_6308` | 0.001 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.002 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.003 |
| 175 | `LOG top_p_bias` | 0.002 |
| 176 | `ADD top_p_logits` | 0.002 |
| 177 | `ARGMAX temp_max_idx` | 0.003 |
| 178 | `GET_ROWS node_6318` | 0.005 |
| 179 | `GET_ROWS node_6321` | 0.004 |
| 180 | `ARGMAX greedy_argmax` | 0.005 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.006 |
| 182 | `TOP_K top_k` | 0.077 |
| 183 | `GET_ROWS top_k_rows` | 0.002 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 185 | `GET_ROWS top_p_candidates` | 0.002 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.001 |
| 189 | `SCALE top_p_cdf_scaled` | 0.001 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | 0.001 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | -0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | -0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | 0.000 |
| 199 | `GET_ROWS node_6353` | 0.000 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (87 passes) | verify-r2 (86 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.36 | 12.35 | -0.01 |
| 1 | 11.10 | 11.08 | -0.02 |
| 2 | 11.27 | 11.25 | -0.01 |
| 3 | 11.39 | 11.40 | +0.00 |
| 4 | 9.22 | 9.22 | -0.01 |
| 5 | 11.33 | 11.32 | -0.01 |
| 6 | 9.49 | 9.48 | -0.01 |
| 7 | 11.39 | 11.38 | -0.01 |
| 8 | 12.86 | 12.87 | +0.01 |
| **all** | **100.42** | **100.36** | **-0.06** |

