24 chunks, 98280 scored tokens; reference for differences: ud; dump reference: Q8_0 (its log p agrees across the dumps to 0.0e+00 nats)

| Positions | All | produced by g3 | produced by ud-q4kxl |
|---|---|---|---|
| stops (</think> closing a non-empty block) | 230 in 24 chunks | 112 | 118 |
| empty blocks (left out) | 0 in 0 chunks | 0 | 0 |
| hesitation markers in reasoning | 842 in 21 chunks | 456 | 386 |

Stops: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - Q8_0 |
|---|---|---|---|---|---|---|
| ud | -0.1628 | 0.8891 |  |  |  | +0.0014 (-0.0159 to +0.0186) |
| g3 | -0.1672 | 0.8876 | -0.0044 (-0.0350 to +0.0174) | -0.0217 (-0.0719 to +0.0128) | +0.0120 (-0.0080 to +0.0333) | -0.0030 (-0.0178 to +0.0080) |
| g3h | -0.1573 | 0.8929 | +0.0054 (-0.0210 to +0.0255) | -0.0116 (-0.0576 to +0.0210) | +0.0215 (+0.0032 to +0.0409) | +0.0068 (-0.0059 to +0.0179) |
| k2h | -0.1561 | 0.8937 | +0.0066 (-0.0222 to +0.0280) | -0.0112 (-0.0643 to +0.0249) | +0.0235 (+0.0064 to +0.0442) | +0.0080 (-0.0079 to +0.0218) |
| k1h | -0.1547 | 0.8933 | +0.0080 (-0.0154 to +0.0274) | -0.0086 (-0.0479 to +0.0235) | +0.0238 (+0.0059 to +0.0426) | +0.0094 (-0.0014 to +0.0204) |
| g3g | -0.1699 | 0.8861 | -0.0071 (-0.0346 to +0.0137) | -0.0250 (-0.0742 to +0.0087) | +0.0098 (-0.0088 to +0.0284) | -0.0057 (-0.0202 to +0.0066) |
| k2hg | -0.1615 | 0.8902 | +0.0013 (-0.0255 to +0.0219) | -0.0191 (-0.0608 to +0.0134) | +0.0206 (+0.0024 to +0.0406) | +0.0027 (-0.0086 to +0.0123) |

Hesitation markers: log p of the actual token

| File | mean log p | mean p | log p - ud (95 % bootstrap over chunks) | produced by g3 | produced by ud-q4kxl | log p - Q8_0 |
|---|---|---|---|---|---|---|
| ud | -0.6957 | 0.6397 |  |  |  | -0.0039 (-0.0148 to +0.0048) |
| g3 | -0.6907 | 0.6419 | +0.0050 (-0.0092 to +0.0221) | +0.0225 (+0.0079 to +0.0353) | -0.0157 (-0.0221 to +0.0005) | +0.0011 (-0.0111 to +0.0170) |
| g3h | -0.7059 | 0.6372 | -0.0102 (-0.0273 to +0.0105) | +0.0107 (-0.0060 to +0.0259) | -0.0348 (-0.0458 to -0.0131) | -0.0141 (-0.0299 to +0.0060) |
| k2h | -0.7016 | 0.6380 | -0.0059 (-0.0221 to +0.0128) | +0.0166 (+0.0045 to +0.0350) | -0.0325 (-0.0468 to -0.0187) | -0.0098 (-0.0226 to +0.0065) |
| k1h | -0.7046 | 0.6367 | -0.0089 (-0.0249 to +0.0093) | +0.0126 (-0.0015 to +0.0315) | -0.0342 (-0.0531 to -0.0178) | -0.0127 (-0.0261 to +0.0024) |
| g3g | -0.6935 | 0.6408 | +0.0023 (-0.0100 to +0.0168) | +0.0184 (+0.0052 to +0.0298) | -0.0167 (-0.0299 to -0.0065) | -0.0016 (-0.0119 to +0.0111) |
| k2hg | -0.7063 | 0.6369 | -0.0106 (-0.0221 to +0.0049) | +0.0069 (-0.0057 to +0.0258) | -0.0313 (-0.0495 to -0.0215) | -0.0145 (-0.0239 to -0.0024) |

Kill rule: |log p - ud| < 0.05 nats with an interval covering 0, at stops and markers, for every file

- g3: stops -0.0044 (-0.0350 to +0.0174); hesitation markers +0.0050 (-0.0092 to +0.0221) -> within
- g3h: stops +0.0054 (-0.0210 to +0.0255); hesitation markers -0.0102 (-0.0273 to +0.0105) -> within
- k2h: stops +0.0066 (-0.0222 to +0.0280); hesitation markers -0.0059 (-0.0221 to +0.0128) -> within
- k1h: stops +0.0080 (-0.0154 to +0.0274); hesitation markers -0.0089 (-0.0249 to +0.0093) -> within
- g3g: stops -0.0071 (-0.0346 to +0.0137); hesitation markers +0.0023 (-0.0100 to +0.0168) -> within
- k2hg: stops +0.0013 (-0.0255 to +0.0219); hesitation markers -0.0106 (-0.0221 to +0.0049) -> within

Quant choice as a reasoning-length lever: closed
