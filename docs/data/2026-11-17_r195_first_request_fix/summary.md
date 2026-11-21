# R195: the first request at a new depth against its repeat, per keep-alive arm

e1: GGML_CUDA_KEEPALIVE_EAGER=1 (the v4 preset as released); k1: =3 (C866). One load per arm, every value one measured request.

## Generation

| arm | depth | first, ms | repeat, ms | first − repeat | steps first / repeat | paired steps: first − repeat | of which steps 0 to 2 | steps 3 on | block stops first / repeat | same reply |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| e1 | 4,096 | 8,965.94 | 8,635.74 | +330.21 | 77 / 77 | +330.26 | +138.81 | +191.45 | 30 / 0 | yes |
| e1 | 6,144 | 8,461.49 | 8,018.89 | +442.60 | 69 / 68 | +238.82 | +143.41 | +95.41 | 31 / 0 | yes |
| e1 | 8,192 | 8,255.25 | 7,948.18 | +307.08 | 69 / 68 | +189.68 | +132.54 | +57.15 | 21 / 0 | yes |
| k1 | 4,096 | 8,868.06 | 8,640.83 | +227.23 | 77 / 77 | +228.78 | +29.56 | +199.22 | 22 / 0 | yes |
| k1 | 6,144 | 8,354.34 | 8,053.28 | +301.06 | 69 / 68 | +96.44 | +33.79 | +62.65 | 13 / 0 | yes |
| k1 | 8,192 | 8,263.42 | 7,956.58 | +306.84 | 69 / 68 | +188.42 | +135.73 | +52.69 | 11 / 0 | yes |

## The passes that run kernel by kernel or are captured (GPU ms summed over the nine cards)

How on cards 0 to 8: E kernel by kernel, C captured, R replayed.

| arm | depth | prompt's last 4 tokens, first | repeat | first verify, first | repeat | second verify, first | repeat | the three, first − repeat | first's last pass (2 tokens at 4,096, 1 at 6,144) | first's 60-token micro-batch | slowest draft evaluation, first, ms | largest one-pass difference, ms (pass) |
|---|---:|---|---|---|---|---|---|---:|---|---|---:|---|
| e1 | 4,096 | EEEEEEEEE 237.57 | EEEEEEEEE 97.75 | EEEEEEEEE 221.89 | EEEEEEEEE 101.53 | CCCCCCCCC 99.00 | CCCCCCCCC 93.65 | +265.53 | EEEEEEEEE 186.94 | CCCCCCCCC 1,962.81 | 10.01 | +139.81 (0) |
| e1 | 6,144 | EEEEEEEEE 220.00 | EEEEEEEEE 103.57 | EEEEEEEEE 224.90 | EEEEEEEEE 107.78 | CCCCCCCCC 104.99 | CCCCCCCCC 99.91 | +238.63 | EEEEEEEEE 186.17 | CCCCCCCCC 1,176.78 | 10.10 | +117.12 (1) |
| e1 | 8,192 | EEEEEEEEE 223.17 | EEEEEEEEE 106.71 | EEEEEEEEE 227.96 | EEEEEEEEE 105.30 | CCCCCCCCC 102.53 | CCCCCCCCC 97.11 | +244.54 | not written | not written | 9.48 | +122.66 (1) |
| k1 | 4,096 | EEEEEEEEE 240.17 | EEEEEEEEE 97.97 | EEEEEEEEE 114.64 | EEEEEEEEE 101.64 | CCCCCCCCC 99.40 | CCCCCCCCC 93.64 | +160.96 | EEEEEEEEE 187.51 | CCCCCCCCC 1,637.05 | 10.06 | +142.21 (0) |
| k1 | 6,144 | EEEEEEEEE 107.68 | EEEEEEEEE 104.97 | EEEEEEEEE 117.33 | EEEEEEEEE 108.54 | CCCCCCCCC 105.35 | CCCCCCCCC 101.62 | +15.23 | EEEEEEEEE 187.11 | CCCCCCCCC 1,167.62 | 10.05 | +8.79 (1) |
| k1 | 8,192 | EEEEEEEEE 113.22 | EEEEEEEEE 106.77 | EEEEEEEEE 227.31 | EEEEEEEEE 105.30 | CCCCCCCCC 102.77 | CCCCCCCCC 97.17 | +134.06 | not written | not written | 7.65 | +122.01 (1) |
