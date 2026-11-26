# Prompt reads by part, product and weight type: `pmix` against `gmix`

Kept requests: prime. The pieces of each profiler file, matched to the load's requests (main evaluations on card 0 per piece):

- group `pmix` (gp-pg.csv): 14 pieces, 12 requests; warmup d0: 14; decode d0 p1: 90; decode d0 p2: 88; sample d0 ps: 85; prime d8192: 17; decode d8192 p1: 88; decode d8192 p2: 84; sample d8192 ps: 88; prime d8704: 17; decode d8704 p1: 86; decode d8704 p2: 85; sample d8704 ps: 85
- group `gmix` (gp-gg.csv): 14 pieces, 12 requests; warmup d0: 16; decode d0 p1: 106; decode d0 p2: 104; sample d0 ps: 119; prime d8192: 17; decode d8192 p1: 98; decode d8192 p2: 97; sample d8192 ps: 86; prime d8704: 17; decode d8704 p1: 92; decode d8704 p2: 91; sample d8704 ps: 84
- host `pmix` (hp-ph.csv): 14 pieces, 12 requests; warmup d0: 14; decode d0 p1: 90; decode d0 p2: 88; sample d0 ps: 85; prime d8192: 17; decode d8192 p1: 88; decode d8192 p2: 84; sample d8192 ps: 88; prime d8704: 17; decode d8704 p1: 86; decode d8704 p2: 85; sample d8704 ps: 85
- host `gmix` (hp-gh.csv): 14 pieces, 12 requests; warmup d0: 16; decode d0 p1: 106; decode d0 p2: 104; sample d0 ps: 119; prime d8192: 17; decode d8192 p1: 98; decode d8192 p2: 97; sample d8192 ps: 86; prime d8704: 17; decode d8704 p1: 92; decode d8704 p2: 91; sample d8704 ps: 84

## Request prime d8192

| File | load | prompt tokens read | prompt ms | tok/s |
|---|---|---:|---:|---:|
| `pmix` | plain (pu2) | 8,128 | 36,533.6 | 222.5 |
| `gmix` | plain (gu2) | 8,128 | 56,250.7 | 144.5 |
| `pmix` | group (pg2) | 8,128 | 100,333.9 | 81.0 |
| `gmix` | group (gg2) | 8,128 | 105,865.0 | 76.8 |
| `pmix` | host (ph3) | 8,128 | 58,665.9 | 138.5 |
| `gmix` | host (gh3) | 8,128 | 65,474.7 | 124.1 |

### Group profile: GPU time by part (ms for the request, all cards)

Main evaluations on card 0: `pmix` 17, `gmix` 17.

| Part | `pmix` | `gmix` | difference | share of the difference |
|---|---:|---:|---:|---:|
| routed experts | 39,734.8 | 45,937.8 | +6,203.0 | 87.6 % |
| shared expert | 1,291.3 | 1,334.8 | +43.5 | 0.6 % |
| hyper-connections (BF16) | 11,239.0 | 11,276.1 | +37.1 | 0.5 % |
| router | 707.2 | 713.2 | +5.9 | 0.1 % |
| attention and recurrent projections | 12,013.8 | 12,729.1 | +715.3 | 10.1 % |
| Gated DeltaNet state and conv | 4,785.3 | 4,784.7 | -0.6 | -0.0 % |
| attention (FLASH_ATTN_EXT) | 12,790.6 | 12,788.3 | -2.4 | -0.0 % |
| output layer | 2.9 | 2.9 | +0.0 | 0.0 % |
| other ops without weights | 3,945.1 | 4,022.1 | +76.9 | 1.1 % |
| other weights | 399.2 | 403.1 | +3.8 | 0.1 % |
| MTP draft (its catch-up over the prompt) | 1,222.7 | 1,223.3 | +0.7 | 0.0 % |
| **all** | **88,132.1** | **95,215.3** | **+7,083.3** | 100 % |

### Group profile: GPU time by card (ms; main model's evaluations + the draft's)

| Card | `pmix` main | `pmix` draft | `gmix` main | `gmix` draft | difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 10,946.1 (17) | 0.0 (0) | 11,296.8 (17) | 0.0 (0) | +350.8 |
| 1 | 11,586.1 (17) | 0.0 (0) | 14,512.8 (17) | 0.0 (0) | +2,926.8 |
| 2 | 10,034.8 (17) | 0.0 (0) | 13,339.4 (17) | 0.0 (0) | +3,304.6 |
| 3 | 11,295.5 (17) | 0.0 (0) | 11,427.8 (17) | 0.0 (0) | +132.3 |
| 4 | 8,700.9 (17) | 0.0 (0) | 8,996.0 (17) | 0.0 (0) | +295.1 |
| 5 | 10,067.4 (17) | 0.0 (0) | 10,200.6 (17) | 0.0 (0) | +133.2 |
| 6 | 9,693.6 (17) | 0.0 (0) | 9,678.4 (17) | 0.0 (0) | -15.2 |
| 7 | 10,357.7 (17) | 0.0 (0) | 10,280.5 (17) | 0.0 (0) | -77.3 |
| 8 | 4,227.2 (17) | 1,222.7 (17) | 4,259.6 (17) | 1,223.3 (17) | +33.0 |

(evaluations in parentheses)

### Routed experts by projection and weight type (plain launch groups: one layer, one micro-batch)

| Projection | type | file | launches | ms | ms a launch | x Q2_0 (same file) | excess over Q2_0, ms |
|---|---|---|---:|---:|---:|---:|---:|
| gate | q2_0 | `pmix` | 768 | 12,257.4 | 15.960 | 1.00 | +0.0 |
| up | q2_0 | `pmix` | 768 | 12,248.7 | 15.949 | 1.00 | +0.0 |
| down | q2_0 | `pmix` | 816 | 15,214.1 | 18.645 | 1.00 | +0.0 |
| gate | iq1_m | `gmix` | 32 | 3,304.1 | 103.252 | 6.50 | +2,795.9 |
| gate | iq2_s | `gmix` | 112 | 1,880.1 | 16.787 | 1.06 | +101.7 |
| gate | iq2_xs | `gmix` | 64 | 1,216.8 | 19.013 | 1.20 | +200.6 |
| gate | iq2_xxs | `gmix` | 176 | 2,901.1 | 16.484 | 1.04 | +106.5 |
| gate | q2_0 | `gmix` | 384 | 6,097.4 | 15.879 | 1.00 | +0.0 |
| up | iq1_m | `gmix` | 32 | 3,202.8 | 100.087 | 6.31 | +2,695.0 |
| up | iq2_s | `gmix` | 112 | 1,880.1 | 16.787 | 1.06 | +103.0 |
| up | iq2_xs | `gmix` | 64 | 1,215.5 | 18.993 | 1.20 | +200.0 |
| up | iq2_xxs | `gmix` | 176 | 2,897.3 | 16.462 | 1.04 | +104.6 |
| up | q2_0 | `gmix` | 384 | 6,093.2 | 15.868 | 1.00 | +0.0 |
| down | q2_0 | `gmix` | 816 | 15,229.2 | 18.663 | 1.00 | +0.0 |

### Products by difference (the label without types and shapes, all cards)

| Product | `pmix` types | `gmix` types | `pmix` launches | `gmix` launches | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID ffn_gate_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 12,257.4 | 15,399.5 | +3,142.2 |
| `MUL_MAT_ID ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 12,248.7 | 15,289.0 | +3,040.3 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 3,921.1 | 4,301.6 | +380.5 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 2,733.5 | 2,862.3 | +128.8 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 204 | 204 | 1,519.1 | 1,618.5 | +99.4 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 612 | 612 | 2,360.6 | 2,434.4 | +73.8 |
| `FLASH_ATTN_EXT node_1583` | - | - | 17 | 17 | 1,060.0 | 1,098.9 | +38.9 |
| `FLASH_ATTN_EXT node_2639` | - | - | 17 | 17 | 1,104.2 | 1,082.1 | -22.1 |
| `FLASH_ATTN_EXT node_4729` | - | - | 17 | 17 | 1,057.3 | 1,035.4 | -22.0 |
| `MUL_MAT ffn_up_shexp` | iq4_xs, q2_0, q4_K, q5_K, q6_K | iq3_s, iq4_xs, q2_0, q4_K, q5_K, q6_K | 770 | 773 | 411.0 | 431.8 | +20.9 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 204 | 204 | 1,097.9 | 1,118.1 | +20.2 |
| `MUL_MAT ffn_gate_shexp` | iq4_xs, q2_0, q4_K, q5_K, q6_K | iq3_s, iq4_xs, q2_0, q4_K, q5_K, q6_K | 770 | 773 | 417.0 | 436.9 | +19.9 |
| `FLASH_ATTN_EXT node_5763` | - | - | 17 | 17 | 1,052.5 | 1,032.6 | -19.9 |
| `FLASH_ATTN_EXT node_2111` | - | - | 17 | 17 | 1,042.3 | 1,061.3 | +19.0 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 816 | 816 | 15,214.1 | 15,229.2 | +15.1 |
| `FLASH_ATTN_EXT node_549` | - | - | 17 | 17 | 1,054.1 | 1,040.8 | -13.3 |
| the other 724 products (each under 10.0 ms apart) | | | | | 28,358.6 | 28,519.7 | +161.0 |
| **the main model's evaluations** | | | | | **86,909.4** | **93,992.0** | **+7,082.6** |

### Pricing (est.)

- Plain prompt time: `pmix` 36,533.6 ms, `gmix` 56,250.7 ms, gap +19,717.1 ms (+54.0 %).
- f = 36,533.6 / 88,132.1 = 0.4145 (`pmix`'s plain time over its summed GPU time).
- `gmix` at f: 39,469.9 ms; its GPU-time difference priced at f: +2,936.2 ms; the rest of its plain time, the pipeline loss: +16,780.9 ms.
- routed gate/up in iq1_m: +5,490.9 ms GPU over Q2_0's rate, est. +2,276.2 ms of plain time.
- routed gate/up in iq2_xs: +400.6 ms GPU over Q2_0's rate, est. +166.1 ms of plain time.
- routed gate/up in iq2_xxs: +211.1 ms GPU over Q2_0's rate, est. +87.5 ms of plain time.
- routed gate/up in iq2_s: +204.7 ms GPU over Q2_0's rate, est. +84.8 ms of plain time.

### Host profile: per card, the host time inside its evaluations and their GPU time (ms)

| Card | `pmix` evals | `pmix` host | `pmix` GPU | `gmix` evals | `gmix` host | `gmix` GPU | host difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 428.2 | 11,154.3 | 17 | 1,418.8 | 12,498.6 | +990.6 |
| 1 | 17 | 1,139.1 | 11,744.6 | 17 | 18,443.3 | 15,501.8 | +17,304.2 |
| 2 | 17 | 1,444.3 | 10,226.4 | 17 | 13,826.8 | 14,274.4 | +12,382.5 |
| 3 | 17 | 1,399.7 | 11,383.3 | 17 | 1,023.6 | 11,532.5 | -376.0 |
| 4 | 17 | 1,442.7 | 8,808.7 | 17 | 1,484.7 | 9,036.8 | +42.0 |
| 5 | 17 | 1,224.0 | 10,221.3 | 17 | 1,253.8 | 10,302.8 | +29.8 |
| 6 | 17 | 1,369.1 | 9,835.1 | 17 | 1,408.1 | 9,743.1 | +39.0 |
| 7 | 17 | 1,266.0 | 10,430.9 | 17 | 1,272.7 | 10,365.7 | +6.8 |
| 8 | 34 | 731.3 | 5,441.2 | 34 | 722.6 | 5,349.6 | -8.8 |
| all | | 10,444.5 | 89,245.8 | | 40,854.5 | 98,605.4 | +30,410.0 |

The piece's span: `pmix` 58,227.7 ms, `gmix` 65,059.2 ms.

## Request prime d8704

| File | load | prompt tokens read | prompt ms | tok/s |
|---|---|---:|---:|---:|
| `pmix` | plain (pu2) | 1,028 | 6,651.8 | 154.5 |
| `gmix` | plain (gu2) | 1,028 | 10,895.6 | 94.3 |
| `pmix` | group (pg2) | 1,028 | 24,875.9 | 41.3 |
| `gmix` | group (gg2) | 1,028 | 26,654.0 | 38.6 |
| `pmix` | host (ph3) | 1,028 | 13,539.6 | 75.9 |
| `gmix` | host (gh3) | 1,028 | 14,276.1 | 72.0 |

### Group profile: GPU time by part (ms for the request, all cards)

Main evaluations on card 0: `pmix` 17, `gmix` 17.

| Part | `pmix` | `gmix` | difference | share of the difference |
|---|---:|---:|---:|---:|
| routed experts | 14,042.8 | 15,163.4 | +1,120.6 | 91.1 % |
| shared expert | 196.8 | 199.6 | +2.8 | 0.2 % |
| hyper-connections (BF16) | 3,571.1 | 3,583.5 | +12.4 | 1.0 % |
| router | 201.1 | 200.3 | -0.8 | -0.1 % |
| attention and recurrent projections | 1,705.0 | 1,792.1 | +87.1 | 7.1 % |
| Gated DeltaNet state and conv | 685.5 | 684.9 | -0.6 | -0.0 % |
| attention (FLASH_ATTN_EXT) | 1,814.2 | 1,818.5 | +4.3 | 0.3 % |
| output layer | 2.9 | 2.9 | -0.0 | -0.0 % |
| other ops without weights | 554.5 | 559.0 | +4.6 | 0.4 % |
| other weights | 119.7 | 119.4 | -0.3 | -0.0 % |
| MTP draft (its catch-up over the prompt) | 155.3 | 155.1 | -0.2 | -0.0 % |
| **all** | **23,048.9** | **24,278.6** | **+1,229.7** | 100 % |

### Group profile: GPU time by card (ms; main model's evaluations + the draft's)

| Card | `pmix` main | `pmix` draft | `gmix` main | `gmix` draft | difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,102.5 (17) | 0.0 (0) | 3,284.8 (17) | 0.0 (0) | +182.4 |
| 1 | 2,993.6 (17) | 0.0 (0) | 3,406.1 (17) | 0.0 (0) | +412.5 |
| 2 | 2,685.6 (17) | 0.0 (0) | 3,083.0 (17) | 0.0 (0) | +397.4 |
| 3 | 2,933.4 (17) | 0.0 (0) | 2,978.0 (17) | 0.0 (0) | +44.7 |
| 4 | 2,333.2 (17) | 0.0 (0) | 2,444.6 (17) | 0.0 (0) | +111.3 |
| 5 | 2,677.9 (17) | 0.0 (0) | 2,750.9 (17) | 0.0 (0) | +73.0 |
| 6 | 2,471.7 (17) | 0.0 (0) | 2,476.1 (17) | 0.0 (0) | +4.4 |
| 7 | 2,708.0 (17) | 0.0 (0) | 2,715.5 (17) | 0.0 (0) | +7.5 |
| 8 | 987.6 (17) | 155.3 (3) | 984.4 (17) | 155.1 (3) | -3.4 |

(evaluations in parentheses)

### Routed experts by projection and weight type (plain launch groups: one layer, one micro-batch)

| Projection | type | file | launches | ms | ms a launch | x Q2_0 (same file) | excess over Q2_0, ms |
|---|---|---|---:|---:|---:|---:|---:|
| gate | q2_0 | `pmix` | 768 | 4,156.5 | 5.412 | 1.00 | +0.0 |
| up | q2_0 | `pmix` | 768 | 4,151.0 | 5.405 | 1.00 | +0.0 |
| down | q2_0 | `pmix` | 816 | 5,721.6 | 7.012 | 1.00 | +0.0 |
| gate | iq1_m | `gmix` | 32 | 419.2 | 13.101 | 2.47 | +249.8 |
| gate | iq2_s | `gmix` | 112 | 663.4 | 5.923 | 1.12 | +70.3 |
| gate | iq2_xs | `gmix` | 64 | 461.0 | 7.203 | 1.36 | +122.0 |
| gate | iq2_xxs | `gmix` | 176 | 1,123.1 | 6.381 | 1.20 | +191.0 |
| gate | q2_0 | `gmix` | 384 | 2,033.5 | 5.296 | 1.00 | +0.0 |
| up | iq1_m | `gmix` | 32 | 418.5 | 13.077 | 2.47 | +249.2 |
| up | iq2_s | `gmix` | 112 | 662.2 | 5.912 | 1.12 | +69.8 |
| up | iq2_xs | `gmix` | 64 | 460.5 | 7.196 | 1.36 | +122.0 |
| up | iq2_xxs | `gmix` | 176 | 1,121.6 | 6.373 | 1.20 | +190.7 |
| up | q2_0 | `gmix` | 384 | 2,031.0 | 5.289 | 1.00 | +0.0 |
| down | q2_0 | `gmix` | 816 | 5,752.4 | 7.049 | 1.00 | +0.0 |

### Products by difference (the label without types and shapes, all cards)

| Product | `pmix` types | `gmix` types | `pmix` launches | `gmix` launches | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID ffn_gate_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 4,156.5 | 4,700.2 | +543.7 |
| `MUL_MAT_ID ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 4,151.0 | 4,693.7 | +542.7 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 472.2 | 519.5 | +47.2 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 816 | 816 | 5,721.6 | 5,752.4 | +30.8 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 329.1 | 344.7 | +15.5 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 204 | 204 | 186.1 | 196.6 | +10.5 |
| the other 661 products (each under 10.0 ms apart) | | | | | 7,877.0 | 7,916.5 | +39.5 |
| **the main model's evaluations** | | | | | **22,893.6** | **24,123.5** | **+1,229.9** |

### Pricing (est.)

- Plain prompt time: `pmix` 6,651.8 ms, `gmix` 10,895.6 ms, gap +4,243.8 ms (+63.8 %).
- f = 6,651.8 / 23,048.9 = 0.2886 (`pmix`'s plain time over its summed GPU time).
- `gmix` at f: 7,006.7 ms; its GPU-time difference priced at f: +354.9 ms; the rest of its plain time, the pipeline loss: +3,888.9 ms.
- routed gate/up in iq1_m: +499.0 ms GPU over Q2_0's rate, est. +144.0 ms of plain time.
- routed gate/up in iq2_xxs: +381.7 ms GPU over Q2_0's rate, est. +110.2 ms of plain time.
- routed gate/up in iq2_xs: +244.1 ms GPU over Q2_0's rate, est. +70.4 ms of plain time.
- routed gate/up in iq2_s: +140.1 ms GPU over Q2_0's rate, est. +40.4 ms of plain time.

### Host profile: per card, the host time inside its evaluations and their GPU time (ms)

| Card | `pmix` evals | `pmix` host | `pmix` GPU | `gmix` evals | `gmix` host | `gmix` GPU | host difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 63.8 | 3,321.2 | 17 | 83.1 | 3,417.1 | +19.3 |
| 1 | 17 | 60.3 | 3,175.4 | 17 | 5,019.9 | 3,712.3 | +4,959.6 |
| 2 | 17 | 56.8 | 2,859.4 | 17 | 3,055.5 | 3,406.5 | +2,998.7 |
| 3 | 17 | 60.5 | 3,132.8 | 17 | 80.2 | 3,085.8 | +19.7 |
| 4 | 17 | 49.5 | 2,457.1 | 17 | 50.6 | 2,510.6 | +1.1 |
| 5 | 17 | 56.1 | 2,852.2 | 17 | 58.9 | 2,898.6 | +2.8 |
| 6 | 17 | 53.3 | 2,611.1 | 17 | 55.8 | 2,619.2 | +2.5 |
| 7 | 17 | 58.3 | 2,840.0 | 17 | 62.0 | 2,887.8 | +3.7 |
| 8 | 20 | 28.4 | 1,168.4 | 20 | 28.6 | 1,197.0 | +0.1 |
| all | | 487.1 | 24,417.7 | | 8,494.5 | 25,734.8 | +8,007.5 |

The piece's span: `pmix` 13,380.8 ms, `gmix` 14,170.5 ms.

