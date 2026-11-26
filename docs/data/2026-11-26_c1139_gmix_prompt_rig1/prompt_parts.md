# Prompt reads by part, product and weight type: `pmix` against `gmix`

Kept requests: prime. The pieces of each profiler file, matched to the load's requests (main evaluations on card 0 per piece):

- group `pmix` (gp-pg.csv): 14 pieces, 12 requests; warmup d0: 14; decode d0 p1: 90; decode d0 p2: 88; sample d0 ps: 85; prime d8192: 17; decode d8192 p1: 88; decode d8192 p2: 84; sample d8192 ps: 88; prime d9216: 4; decode d9216 p1: 87; decode d9216 p2: 86; sample d9216 ps: 84
- group `gmix` (gp-gg.csv): 14 pieces, 12 requests; warmup d0: 16; decode d0 p1: 106; decode d0 p2: 104; sample d0 ps: 119; prime d8192: 17; decode d8192 p1: 98; decode d8192 p2: 97; sample d8192 ps: 86; prime d9216: 4; decode d9216 p1: 75; decode d9216 p2: 74; sample d9216 ps: 88
- host `pmix` (hp-ph.csv): 14 pieces, 12 requests; warmup d0: 14; decode d0 p1: 90; decode d0 p2: 88; sample d0 ps: 85; prime d8192: 17; decode d8192 p1: 88; decode d8192 p2: 84; sample d8192 ps: 88; prime d9216: 4; decode d9216 p1: 87; decode d9216 p2: 86; sample d9216 ps: 84
- host `gmix` (hp-gh.csv): 14 pieces, 12 requests; warmup d0: 16; decode d0 p1: 106; decode d0 p2: 104; sample d0 ps: 119; prime d8192: 17; decode d8192 p1: 98; decode d8192 p2: 97; sample d8192 ps: 86; prime d9216: 4; decode d9216 p1: 75; decode d9216 p2: 74; sample d9216 ps: 88

## Request prime d8192

| File | load | prompt tokens read | prompt ms | tok/s |
|---|---|---:|---:|---:|
| `pmix` | plain (pu1) | 8,128 | 36,786.5 | 221.0 |
| `gmix` | plain (gu1) | 8,128 | 56,033.2 | 145.1 |
| `pmix` | group (pg) | 8,128 | 101,447.9 | 80.1 |
| `gmix` | group (gg) | 8,128 | 107,365.6 | 75.7 |
| `pmix` | host (ph) | 8,128 | 57,082.5 | 142.4 |
| `gmix` | host (gh) | 8,128 | 65,384.3 | 124.3 |

### Group profile: GPU time by part (ms for the request, all cards)

Main evaluations on card 0: `pmix` 17, `gmix` 17.

| Part | `pmix` | `gmix` | difference | share of the difference |
|---|---:|---:|---:|---:|
| routed experts | 39,497.5 | 45,919.3 | +6,421.8 | 85.6 % |
| shared expert | 1,283.2 | 1,333.3 | +50.2 | 0.7 % |
| hyper-connections (BF16) | 11,181.4 | 11,269.9 | +88.5 | 1.2 % |
| router | 706.6 | 712.1 | +5.5 | 0.1 % |
| attention and recurrent projections | 11,942.1 | 12,726.2 | +784.1 | 10.4 % |
| Gated DeltaNet state and conv | 4,766.2 | 4,782.7 | +16.5 | 0.2 % |
| attention (FLASH_ATTN_EXT) | 12,678.3 | 12,752.3 | +74.0 | 1.0 % |
| output layer | 2.9 | 2.9 | +0.0 | 0.0 % |
| other ops without weights | 3,967.5 | 4,022.6 | +55.1 | 0.7 % |
| other weights | 399.0 | 405.7 | +6.7 | 0.1 % |
| MTP draft (its catch-up over the prompt) | 1,219.7 | 1,220.7 | +1.1 | 0.0 % |
| **all** | **87,644.3** | **95,147.8** | **+7,503.5** | 100 % |

### Group profile: GPU time by card (ms; main model's evaluations + the draft's)

| Card | `pmix` main | `pmix` draft | `gmix` main | `gmix` draft | difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 10,889.2 (17) | 0.0 (0) | 11,312.2 (17) | 0.0 (0) | +423.1 |
| 1 | 11,468.9 (17) | 0.0 (0) | 14,502.4 (17) | 0.0 (0) | +3,033.5 |
| 2 | 10,042.8 (17) | 0.0 (0) | 13,338.3 (17) | 0.0 (0) | +3,295.5 |
| 3 | 11,225.5 (17) | 0.0 (0) | 11,457.5 (17) | 0.0 (0) | +232.0 |
| 4 | 8,684.9 (17) | 0.0 (0) | 8,944.6 (17) | 0.0 (0) | +259.7 |
| 5 | 10,020.0 (17) | 0.0 (0) | 10,201.9 (17) | 0.0 (0) | +181.9 |
| 6 | 9,645.4 (17) | 0.0 (0) | 9,639.8 (17) | 0.0 (0) | -5.6 |
| 7 | 10,264.1 (17) | 0.0 (0) | 10,271.2 (17) | 0.0 (0) | +7.1 |
| 8 | 4,183.9 (17) | 1,219.7 (17) | 4,259.1 (17) | 1,220.7 (17) | +76.2 |

(evaluations in parentheses)

### Routed experts by projection and weight type (plain launch groups: one layer, one micro-batch)

| Projection | type | file | launches | ms | ms a launch | x Q2_0 (same file) | excess over Q2_0, ms |
|---|---|---|---:|---:|---:|---:|---:|
| gate | q2_0 | `pmix` | 768 | 12,178.7 | 15.858 | 1.00 | +0.0 |
| up | q2_0 | `pmix` | 768 | 12,173.0 | 15.850 | 1.00 | +0.0 |
| down | q2_0 | `pmix` | 816 | 15,129.2 | 18.541 | 1.00 | +0.0 |
| gate | iq1_m | `gmix` | 32 | 3,312.0 | 103.499 | 6.53 | +2,805.0 |
| gate | iq2_s | `gmix` | 112 | 1,878.4 | 16.771 | 1.06 | +104.0 |
| gate | iq2_xs | `gmix` | 64 | 1,213.1 | 18.955 | 1.20 | +199.2 |
| gate | iq2_xxs | `gmix` | 176 | 2,907.4 | 16.520 | 1.04 | +119.1 |
| gate | q2_0 | `gmix` | 384 | 6,083.6 | 15.843 | 1.00 | +0.0 |
| up | iq1_m | `gmix` | 32 | 3,207.1 | 100.221 | 6.33 | +2,700.6 |
| up | iq2_s | `gmix` | 112 | 1,877.4 | 16.762 | 1.06 | +104.6 |
| up | iq2_xs | `gmix` | 64 | 1,213.8 | 18.965 | 1.20 | +200.7 |
| up | iq2_xxs | `gmix` | 176 | 2,902.1 | 16.489 | 1.04 | +116.3 |
| up | q2_0 | `gmix` | 384 | 6,078.2 | 15.829 | 1.00 | +0.0 |
| down | q2_0 | `gmix` | 816 | 15,225.9 | 18.659 | 1.00 | +0.0 |

### Products by difference (the label without types and shapes, all cards)

| Product | `pmix` types | `gmix` types | `pmix` launches | `gmix` launches | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID ffn_gate_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 12,178.7 | 15,394.5 | +3,215.8 |
| `MUL_MAT_ID ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 768 | 768 | 12,173.0 | 15,278.6 | +3,105.6 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 3,890.9 | 4,297.9 | +407.0 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 612 | 612 | 2,716.3 | 2,860.4 | +144.1 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 204 | 204 | 1,511.5 | 1,624.0 | +112.5 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 816 | 816 | 15,129.2 | 15,225.9 | +96.8 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 612 | 612 | 2,345.1 | 2,432.5 | +87.4 |
| `FLASH_ATTN_EXT node_1583` | - | - | 17 | 17 | 1,062.1 | 1,092.1 | +30.0 |
| `MUL_MAT attn_output` | q4_K, q5_K, q6_K | q4_K, q5_K, q6_K | 204 | 204 | 1,092.9 | 1,117.5 | +24.6 |
| `MUL_MAT ffn_gate_shexp` | iq4_xs, q2_0, q4_K, q5_K, q6_K | iq3_s, iq4_xs, q2_0, q4_K, q5_K, q6_K | 770 | 773 | 415.3 | 437.1 | +21.8 |
| `MUL_MAT hc_ffn_down` | bf16 | bf16 | 816 | 816 | 2,354.0 | 2,374.0 | +20.0 |
| `MUL_MAT ffn_up_shexp` | iq4_xs, q2_0, q4_K, q5_K, q6_K | iq3_s, iq4_xs, q2_0, q4_K, q5_K, q6_K | 770 | 773 | 410.8 | 430.3 | +19.6 |
| `FLASH_ATTN_EXT node_2111` | - | - | 17 | 17 | 1,035.9 | 1,054.8 | +19.0 |
| `FLASH_ATTN_EXT node_4729` | - | - | 17 | 17 | 1,040.0 | 1,021.3 | -18.7 |
| `FLASH_ATTN_EXT node_6291` | - | - | 17 | 17 | 1,076.5 | 1,095.0 | +18.5 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | - | - | 1632 | 1632 | 2,417.7 | 2,436.1 | +18.4 |
| `MUL_MAT hc_attn_down` | bf16 | bf16 | 816 | 816 | 2,359.8 | 2,376.4 | +16.6 |
| `FLASH_ATTN_EXT node_1077` | - | - | 17 | 17 | 1,134.6 | 1,148.7 | +14.1 |
| `MUL_MAT hc_ffn_up` | bf16 | bf16 | 816 | 816 | 1,496.5 | 1,506.7 | +10.2 |
| the other 721 products (each under 10.0 ms apart) | | | | | 20,584.0 | 20,723.2 | +139.2 |
| **the main model's evaluations** | | | | | **86,424.6** | **93,927.0** | **+7,502.4** |

### Pricing (est.)

- Plain prompt time: `pmix` 36,786.5 ms, `gmix` 56,033.2 ms, gap +19,246.7 ms (+52.3 %).
- f = 36,786.5 / 87,644.3 = 0.4197 (`pmix`'s plain time over its summed GPU time).
- `gmix` at f: 39,935.9 ms; its GPU-time difference priced at f: +3,149.4 ms; the rest of its plain time, the pipeline loss: +16,097.3 ms.
- routed gate/up in iq1_m: +5,505.6 ms GPU over Q2_0's rate, est. +2,310.8 ms of plain time.
- routed gate/up in iq2_xs: +399.9 ms GPU over Q2_0's rate, est. +167.8 ms of plain time.
- routed gate/up in iq2_xxs: +235.4 ms GPU over Q2_0's rate, est. +98.8 ms of plain time.
- routed gate/up in iq2_s: +208.6 ms GPU over Q2_0's rate, est. +87.5 ms of plain time.

### Host profile: per card, the host time inside its evaluations and their GPU time (ms)

| Card | `pmix` evals | `pmix` host | `pmix` GPU | `gmix` evals | `gmix` host | `gmix` GPU | host difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 421.0 | 11,152.4 | 17 | 440.6 | 11,480.4 | +19.6 |
| 1 | 17 | 1,125.5 | 11,729.3 | 17 | 18,348.4 | 15,441.6 | +17,222.9 |
| 2 | 17 | 1,474.2 | 10,167.8 | 17 | 13,800.1 | 14,227.6 | +12,325.8 |
| 3 | 17 | 1,379.8 | 11,351.3 | 17 | 1,036.9 | 11,508.0 | -342.9 |
| 4 | 17 | 1,435.4 | 8,742.6 | 17 | 1,457.4 | 9,007.5 | +22.0 |
| 5 | 17 | 1,202.0 | 10,251.9 | 17 | 1,254.7 | 10,283.1 | +52.6 |
| 6 | 17 | 1,397.6 | 9,772.6 | 17 | 1,409.0 | 9,722.9 | +11.4 |
| 7 | 17 | 1,270.3 | 10,351.3 | 17 | 1,258.7 | 10,320.8 | -11.6 |
| 8 | 34 | 729.8 | 5,424.4 | 34 | 731.7 | 5,331.4 | +2.0 |
| all | | 10,435.8 | 88,943.7 | | 39,737.3 | 97,323.4 | +29,301.6 |

The piece's span: `pmix` 56,617.7 ms, `gmix` 64,897.5 ms.

## Request prime d9216

| File | load | prompt tokens read | prompt ms | tok/s |
|---|---|---:|---:|---:|
| `pmix` | plain (pu1) | 1,540 | 8,996.6 | 171.2 |
| `gmix` | plain (gu1) | 1,540 | 13,421.4 | 114.7 |
| `pmix` | group (pg) | 1,540 | 21,087.2 | 73.0 |
| `gmix` | group (gg) | 1,540 | 23,029.0 | 66.9 |
| `pmix` | host (ph) | 1,540 | 15,560.9 | 99.0 |
| `gmix` | host (gh) | 1,540 | 16,332.6 | 94.3 |

### Group profile: GPU time by part (ms for the request, all cards)

Main evaluations on card 0: `pmix` 4, `gmix` 4.

| Part | `pmix` | `gmix` | difference | share of the difference |
|---|---:|---:|---:|---:|
| routed experts | 7,519.3 | 8,934.3 | +1,415.1 | 75.5 % |
| shared expert | 242.3 | 255.6 | +13.4 | 0.7 % |
| hyper-connections (BF16) | 2,092.2 | 2,155.0 | +62.8 | 3.4 % |
| router | 131.7 | 135.3 | +3.5 | 0.2 % |
| attention and recurrent projections | 2,247.2 | 2,463.4 | +216.2 | 11.5 % |
| Gated DeltaNet state and conv | 902.5 | 923.8 | +21.4 | 1.1 % |
| attention (FLASH_ATTN_EXT) | 4,672.9 | 4,786.0 | +113.1 | 6.0 % |
| output layer | 2.9 | 2.9 | +0.0 | 0.0 % |
| other ops without weights | 784.0 | 808.6 | +24.6 | 1.3 % |
| other weights | 73.9 | 77.8 | +3.9 | 0.2 % |
| MTP draft (its catch-up over the prompt) | 231.1 | 231.0 | -0.1 | -0.0 % |
| **all** | **18,899.8** | **20,773.7** | **+1,873.9** | 100 % |

### Group profile: GPU time by card (ms; main model's evaluations + the draft's)

| Card | `pmix` main | `pmix` draft | `gmix` main | `gmix` draft | difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,243.4 (4) | 0.0 (0) | 2,368.7 (4) | 0.0 (0) | +125.3 |
| 1 | 2,585.9 (4) | 0.0 (0) | 3,286.5 (4) | 0.0 (0) | +700.7 |
| 2 | 2,115.9 (4) | 0.0 (0) | 2,769.9 (4) | 0.0 (0) | +654.0 |
| 3 | 2,476.9 (4) | 0.0 (0) | 2,614.0 (4) | 0.0 (0) | +137.0 |
| 4 | 1,829.0 (4) | 0.0 (0) | 1,941.6 (4) | 0.0 (0) | +112.6 |
| 5 | 2,097.8 (4) | 0.0 (0) | 2,186.1 (4) | 0.0 (0) | +88.3 |
| 6 | 2,186.3 (4) | 0.0 (0) | 2,259.8 (4) | 0.0 (0) | +73.4 |
| 7 | 2,105.5 (4) | 0.0 (0) | 2,163.6 (4) | 0.0 (0) | +58.1 |
| 8 | 1,028.1 (4) | 231.1 (4) | 952.5 (4) | 231.0 (4) | -75.7 |

(evaluations in parentheses)

### Routed experts by projection and weight type (plain launch groups: one layer, one micro-batch)

| Projection | type | file | launches | ms | ms a launch | x Q2_0 (same file) | excess over Q2_0, ms |
|---|---|---|---:|---:|---:|---:|---:|
| gate | q2_0 | `pmix` | 144 | 2,311.7 | 16.053 | 1.00 | +0.0 |
| up | q2_0 | `pmix` | 144 | 2,311.0 | 16.048 | 1.00 | +0.0 |
| down | q2_0 | `pmix` | 192 | 2,882.2 | 15.012 | 1.00 | +0.0 |
| gate | iq1_m | `gmix` | 6 | 654.3 | 109.047 | 6.74 | +557.3 |
| gate | iq2_s | `gmix` | 21 | 374.7 | 17.842 | 1.10 | +35.1 |
| gate | iq2_xs | `gmix` | 12 | 235.8 | 19.647 | 1.21 | +41.7 |
| gate | iq2_xxs | `gmix` | 33 | 564.7 | 17.112 | 1.06 | +31.1 |
| gate | q2_0 | `gmix` | 72 | 1,164.2 | 16.170 | 1.00 | +0.0 |
| up | iq1_m | `gmix` | 6 | 626.3 | 104.377 | 6.46 | +529.3 |
| up | iq2_s | `gmix` | 21 | 374.9 | 17.850 | 1.10 | +35.4 |
| up | iq2_xs | `gmix` | 12 | 236.2 | 19.682 | 1.22 | +42.2 |
| up | iq2_xxs | `gmix` | 33 | 564.8 | 17.116 | 1.06 | +31.3 |
| up | q2_0 | `gmix` | 72 | 1,164.0 | 16.166 | 1.00 | +0.0 |
| down | q2_0 | `gmix` | 192 | 2,957.0 | 15.401 | 1.00 | +0.0 |

### Products by difference (the label without types and shapes, all cards)

| Product | `pmix` types | `gmix` types | `pmix` launches | `gmix` launches | `pmix` ms | `gmix` ms | difference |
|---|---|---|---:|---:|---:|---:|---:|
| `MUL_MAT_ID ffn_gate_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 144 | 144 | 2,311.7 | 2,993.7 | +682.0 |
| `MUL_MAT_ID ffn_up_exps` | q2_0 | iq1_m, iq2_s, iq2_xs, iq2_xxs, q2_0 | 144 | 144 | 2,311.0 | 2,966.1 | +655.2 |
| `MUL_MAT attn_qkv` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q5_K, q6_K | 144 | 144 | 731.6 | 835.4 | +103.9 |
| `MUL_MAT_ID ffn_down_exps` | q2_0 | q2_0 | 192 | 192 | 2,882.2 | 2,957.0 | +74.8 |
| `MUL_MAT ssm_out` | iq4_xs, q4_K, q5_K, q6_K | iq4_xs, q4_K, q5_K, q6_K | 144 | 144 | 513.2 | 557.6 | +44.4 |
| `FLASH_ATTN_EXT node_4729` | - | - | 4 | 4 | 370.5 | 411.9 | +41.4 |
| `FLASH_ATTN_EXT node_6291` | - | - | 4 | 4 | 420.9 | 391.2 | -29.7 |
| `MUL_MAT attn_gate` | iq4_xs, q2_0, q4_K | iq4_xs, q4_K, q6_K | 144 | 144 | 443.0 | 472.5 | +29.5 |
| `MUL_MAT attn_q` | iq4_xs, q2_0 | iq4_xs, q2_0, q4_K | 48 | 48 | 284.9 | 313.8 | +28.8 |
| `FLASH_ATTN_EXT node_2639` | - | - | 4 | 4 | 393.8 | 420.8 | +27.0 |
| `FLASH_ATTN_EXT node_1077` | - | - | 4 | 4 | 411.0 | 437.2 | +26.2 |
| `MUL_MAT hc_attn_down` | bf16 | bf16 | 192 | 192 | 438.6 | 456.3 | +17.7 |
| `FLASH_ATTN_EXT node_4201` | - | - | 4 | 4 | 389.8 | 404.4 | +14.5 |
| `SCALE+SIGMOID+SCALE+DSV4_HC_POST hc_inject` | - | - | 384 | 384 | 453.6 | 465.6 | +11.9 |
| `FLASH_ATTN_EXT node_1583` | - | - | 4 | 4 | 396.9 | 407.5 | +10.7 |
| `MUL_MAT hc_ffn_down` | bf16 | bf16 | 192 | 192 | 441.0 | 451.1 | +10.0 |
| the other 651 products (each under 10.0 ms apart) | | | | | 5,474.9 | 5,600.5 | +125.6 |
| **the main model's evaluations** | | | | | **18,668.7** | **20,542.7** | **+1,874.0** |

### Pricing (est.)

- Plain prompt time: `pmix` 8,996.6 ms, `gmix` 13,421.4 ms, gap +4,424.8 ms (+49.2 %).
- f = 8,996.6 / 18,899.8 = 0.4760 (`pmix`'s plain time over its summed GPU time).
- `gmix` at f: 9,888.6 ms; its GPU-time difference priced at f: +892.0 ms; the rest of its plain time, the pipeline loss: +3,532.8 ms.
- routed gate/up in iq1_m: +1,086.5 ms GPU over Q2_0's rate, est. +517.2 ms of plain time.
- routed gate/up in iq2_xs: +83.9 ms GPU over Q2_0's rate, est. +39.9 ms of plain time.
- routed gate/up in iq2_s: +70.5 ms GPU over Q2_0's rate, est. +33.5 ms of plain time.
- routed gate/up in iq2_xxs: +62.4 ms GPU over Q2_0's rate, est. +29.7 ms of plain time.

### Host profile: per card, the host time inside its evaluations and their GPU time (ms)

| Card | `pmix` evals | `pmix` host | `pmix` GPU | `gmix` evals | `gmix` host | `gmix` GPU | host difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 14.4 | 2,396.2 | 4 | 52.3 | 2,439.0 | +37.9 |
| 1 | 4 | 13.4 | 2,704.1 | 4 | 4,067.2 | 3,490.3 | +4,053.8 |
| 2 | 4 | 12.3 | 2,191.8 | 4 | 2,942.6 | 3,074.1 | +2,930.3 |
| 3 | 4 | 13.4 | 2,606.4 | 4 | 22.1 | 2,585.2 | +8.7 |
| 4 | 4 | 12.5 | 1,914.6 | 4 | 15.9 | 1,920.6 | +3.4 |
| 5 | 4 | 13.6 | 2,233.3 | 4 | 18.6 | 2,212.1 | +5.1 |
| 6 | 4 | 14.5 | 2,316.7 | 4 | 14.1 | 2,276.6 | -0.4 |
| 7 | 4 | 15.2 | 2,205.2 | 4 | 14.0 | 2,197.5 | -1.2 |
| 8 | 8 | 8.7 | 1,217.0 | 8 | 9.2 | 1,198.2 | +0.5 |
| all | | 118.0 | 19,785.2 | | 7,156.0 | 21,393.4 | +7,038.0 |

The piece's span: `pmix` 14,199.4 ms, `gmix` 15,378.4 ms.

