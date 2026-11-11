2911 evaluations on cards 0 to 8: the model's graph 0: 173, 1: 173, 2: 173, 3: 173, 4: 173, 5: 173, 6: 173, 7: 173, 8: 173; the draft's graphs 1354 (card 4: 677, card 8: 677), its last card 8
main passes per piece (cut at pauses over 3000 ms): 1, 88, 84; passes whose cards are out of order: 0

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps | median raw GPU time of a typical verify pass, ms | from pass to pass under the profiler, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 88 | 3 | 84 | 4: 84 | 81 | 118.99 | 158.46 |
| r2 | 84 | 0 | 83 | 4: 83 | 80 | 118.98 | 158.16 |

**GPU time of a pass by part, ms: cards 0 to 7**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 19.56 | 19.55 | -0.00 |
| shared expert | 4.63 | 4.63 | -0.01 |
| hyper-connections (BF16) | 12.42 | 12.37 | -0.05 |
| router | 2.30 | 2.30 | -0.00 |
| attention and recurrent projections | 32.74 | 32.76 | +0.02 |
| Gated DeltaNet state and conv | 8.68 | 8.69 | +0.02 |
| attention (FLASH_ATTN_EXT) | 0.95 | 0.94 | -0.01 |
| other ops without weights | 6.16 | 6.13 | -0.03 |
| other weights | 0.74 | 0.75 | +0.01 |
| **sum** | **88.18** | **88.13** | **-0.05** |

**GPU time of a pass by part, ms: card 8**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert | 0.28 | 0.28 | +0.00 |
| hyper-connections (BF16) | 0.56 | 0.56 | +0.00 |
| router | 0.10 | 0.10 | +0.00 |
| attention and recurrent projections | 1.42 | 1.43 | +0.01 |
| Gated DeltaNet state and conv | 0.26 | 0.25 | -0.00 |
| attention (FLASH_ATTN_EXT) | 0.09 | 0.09 | -0.00 |
| output layer | 8.11 | 8.12 | +0.01 |
| other ops without weights | 1.08 | 1.08 | +0.00 |
| other weights | 0.15 | 0.15 | -0.00 |
| **sum** | **12.90** | **12.92** | **+0.01** |

**GPU time of a pass by part, ms: all cards**

| Part | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| routed experts | 20.42 | 20.41 | -0.00 |
| shared expert | 4.91 | 4.90 | -0.01 |
| hyper-connections (BF16) | 12.98 | 12.93 | -0.05 |
| router | 2.41 | 2.40 | -0.00 |
| attention and recurrent projections | 34.16 | 34.19 | +0.03 |
| Gated DeltaNet state and conv | 8.93 | 8.95 | +0.02 |
| attention (FLASH_ATTN_EXT) | 1.04 | 1.03 | -0.01 |
| output layer | 8.11 | 8.12 | +0.01 |
| other ops without weights | 7.24 | 7.21 | -0.03 |
| other weights | 0.89 | 0.90 | +0.01 |
| **sum** | **101.09** | **101.05** | **-0.04** |

**Where the difference goes, ms a pass: cards 0 to 7**

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| GATED_DELTA_NET | 8.68 | 8.69 | +0.02 |
| GDN_CONV | 0.73 | 0.74 | +0.01 |
| copies of the PLE conv history (cache_ple_r) | 0.07 | 0.07 | +0.00 |
| routed experts | 19.56 | 19.55 | -0.00 |
| shared expert, router, hyper-connections | 19.36 | 19.30 | -0.06 |
| everything else | 39.79 | 39.78 | -0.01 |
| **sum** | **88.18** | **88.13** | **-0.05** |

**Where the difference goes, ms a pass: card 8** (its commonest graph, launch groups in order)

| Where | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| the head's product | 8.11 | 8.12 | +0.01 |
| the sampling chain after the product | 0.65 | 0.65 | -0.00 |
| the output's hyper-connection products | 0.11 | 0.11 | -0.00 |
| GATED_DELTA_NET | 0.26 | 0.25 | -0.00 |
| GDN_CONV | 0.02 | 0.02 | +0.00 |
| routed experts | 0.86 | 0.86 | +0.00 |
| shared expert, router, hyper-connections | 0.94 | 0.94 | +0.00 |
| everything else | 1.96 | 1.97 | +0.01 |
| **sum** | **12.90** | **12.92** | **+0.01** |

**The last card by position in its graph, ms** (the evaluations with the commonest graph: verify-r1 81 of 81, 204 launch groups, verify-r2 80 of 80, 204 launch groups)

| Position | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| before the head's product | 4.14 | 4.15 | +0.01 |
| the head's product | 8.11 | 8.12 | +0.01 |
| after the head's product | 0.65 | 0.65 | -0.00 |

**By the launch groups' first op, ms a pass (launch groups a pass): cards 0 to 7**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL | 0.87 (134) | 0.79 (134) | -0.08 |
| SIGMOID | 0.25 (93) | 0.28 (93) | +0.03 |
| SCALE | 1.48 (197) | 1.46 (197) | -0.02 |
| TOP_K | 0.15 (11) | 0.13 (11) | -0.02 |
| GATED_DELTA_NET | 8.68 (35) | 8.69 (35) | +0.02 |
| SILU | 0.18 (93) | 0.19 (93) | +0.01 |
| GET_ROWS | 0.20 (23) | 0.19 (23) | -0.01 |
| FLASH_ATTN_EXT | 0.95 (11) | 0.94 (11) | -0.01 |
| CONT | 0.53 (31) | 0.52 (31) | -0.01 |
| CONCAT | 0.13 (12) | 0.14 (12) | +0.01 |
| RMS_NORM | 0.97 (174) | 0.98 (174) | +0.01 |
| GDN_CONV | 0.73 (35) | 0.74 (35) | +0.01 |
| DSV4_HC_PRE | 0.50 (92) | 0.50 (92) | -0.01 |
| SOFTPLUS | 0.17 (35) | 0.17 (35) | +0.01 |
| the other 18 ops | 72.40 | 72.41 | +0.01 |

**By the launch groups' first op, ms a pass (launch groups a pass): card 8**

| First op | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| MUL_MAT | 10.55 (36) | 10.56 (36) | +0.01 |
| ARGSORT | 0.00 (4) | 0.01 (4) | +0.01 |
| SET_ROWS | 0.02 (8) | 0.02 (8) | +0.00 |
| GET_ROWS | 0.08 (27) | 0.08 (27) | -0.00 |
| RMS_NORM | 0.05 (10) | 0.05 (10) | +0.00 |
| LOG | 0.01 (4) | 0.00 (4) | -0.00 |
| SIGMOID | 0.05 (5) | 0.04 (5) | -0.00 |
| SUM | 0.01 (4) | 0.00 (4) | -0.00 |
| SCALE | 0.08 (19) | 0.09 (19) | +0.00 |
| STEP | 0.01 (4) | 0.00 (4) | -0.00 |
| MUL_MAT_ID | 0.86 (4) | 0.86 (4) | +0.00 |
| CPY | 0.03 (5) | 0.03 (5) | -0.00 |
| ADD | 0.03 (13) | 0.04 (13) | +0.00 |
| CUMSUM | 0.01 (4) | 0.01 (4) | -0.00 |
| the other 19 ops | 1.13 | 1.13 | -0.00 |

**The last card's launch groups from three before the head's product on, verify-r2, ms**

| Group | Label | ms |
|---|---:|---:|
| 114 | `ADD node_6214` | 0.003 |
| 115 | `ADD node_6216` | 0.003 |
| 116 | `SCALE result_norm` | 0.003 |
| 117 | `MUL_MAT output:q5_K:2560x248320` | 8.120 |
| 118 | `PAD node_6219` | 0.217 |
| 119 | `TOP_K top_k` | 0.076 |
| 120 | `GET_ROWS top_k_rows` | 0.004 |
| 121 | `ARGSORT top_p_sorted_idx` | 0.003 |
| 122 | `GET_ROWS top_p_candidates` | 0.005 |
| 123 | `GET_ROWS top_p_sorted_logits` | 0.004 |
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
| 140 | `TOP_K top_k` | 0.073 |
| 141 | `GET_ROWS top_k_rows` | 0.002 |
| 142 | `ARGSORT top_p_sorted_idx` | 0.001 |
| 143 | `GET_ROWS top_p_candidates` | 0.002 |
| 144 | `GET_ROWS top_p_sorted_logits` | 0.001 |
| 145 | `SOFT_MAX top_p_softmax` | 0.001 |
| 146 | `CUMSUM top_p_cdf` | 0.001 |
| 147 | `SCALE top_p_cdf_scaled` | 0.000 |
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
| 159 | `ARGMAX greedy_argmax` | -0.000 |
| 160 | `GET_ROWS greedy_sampled_token` | 0.000 |
| 161 | `TOP_K top_k` | 0.071 |
| 162 | `GET_ROWS top_k_rows` | 0.000 |
| 163 | `ARGSORT top_p_sorted_idx` | -0.000 |
| 164 | `GET_ROWS top_p_candidates` | 0.001 |
| 165 | `GET_ROWS top_p_sorted_logits` | 0.000 |
| 166 | `SOFT_MAX top_p_softmax` | 0.002 |
| 167 | `CUMSUM top_p_cdf` | 0.000 |
| 168 | `SCALE top_p_cdf_scaled` | 0.001 |
| 169 | `STEP top_p_mask` | -0.000 |
| 170 | `SUM top_p_index_f32` | 0.000 |
| 171 | `CLAMP node_6308` | -0.000 |
| 172 | `SCALE top_p_ones` | 0.000 |
| 173 | `CPY _(copy)` | 0.001 |
| 174 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.004 |
| 175 | `LOG top_p_bias` | 0.001 |
| 176 | `ADD top_p_logits` | 0.006 |
| 177 | `ARGMAX temp_max_idx` | 0.003 |
| 178 | `GET_ROWS node_6318` | 0.005 |
| 179 | `GET_ROWS node_6321` | 0.003 |
| 180 | `ARGMAX greedy_argmax` | 0.003 |
| 181 | `GET_ROWS greedy_sampled_token` | 0.007 |
| 182 | `TOP_K top_k` | 0.076 |
| 183 | `GET_ROWS top_k_rows` | 0.003 |
| 184 | `ARGSORT top_p_sorted_idx` | 0.007 |
| 185 | `GET_ROWS top_p_candidates` | 0.003 |
| 186 | `GET_ROWS top_p_sorted_logits` | 0.002 |
| 187 | `SOFT_MAX top_p_softmax` | 0.002 |
| 188 | `CUMSUM top_p_cdf` | 0.001 |
| 189 | `SCALE top_p_cdf_scaled` | -0.000 |
| 190 | `STEP top_p_mask` | 0.000 |
| 191 | `SUM top_p_index_f32` | 0.000 |
| 192 | `CLAMP node_6343` | 0.000 |
| 193 | `SCALE top_p_ones` | 0.000 |
| 194 | `CPY _(copy)` | -0.000 |
| 195 | `SET_ROWS top_p_mask_(reshaped)_(view)` | 0.000 |
| 196 | `LOG top_p_bias` | -0.000 |
| 197 | `ADD top_p_logits` | 0.000 |
| 198 | `ARGMAX temp_max_idx` | 0.000 |
| 199 | `GET_ROWS node_6353` | 0.000 |
| 200 | `GET_ROWS node_6356` | 0.000 |
| 201 | `ARGMAX greedy_argmax` | 0.000 |
| 202 | `GET_ROWS greedy_sampled_token` | -0.000 |
| 203 | `SET_ROWS h_nextn_dev_(view)` | 0.007 |

**Per card, ms a pass (the profiler's cost subtracted per span)**

| Card | verify-r1 (81 passes) | verify-r2 (80 passes) | verify-r2 − verify-r1 |
|---|---:|---:|---:|
| 0 | 12.41 | 12.41 | +0.00 |
| 1 | 11.21 | 11.18 | -0.02 |
| 2 | 11.35 | 11.35 | -0.00 |
| 3 | 11.51 | 11.49 | -0.01 |
| 4 | 9.26 | 9.26 | +0.00 |
| 5 | 11.42 | 11.42 | +0.00 |
| 6 | 9.57 | 9.57 | -0.00 |
| 7 | 11.46 | 11.44 | -0.02 |
| 8 | 12.90 | 12.92 | +0.01 |
| **all** | **101.09** | **101.05** | **-0.04** |

