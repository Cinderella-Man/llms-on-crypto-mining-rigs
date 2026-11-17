# Graph decisions and keep-alive counts per request (<OUT>/server-e1.log)

| task | request | gen tokens | gen ms | graphs reused (llama) | debug: reused | warm-up done (captured) | warm-up reset (kernel by kernel) | update failed | disabled | kernel by kernel | at new shapes | block stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (load) |  |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |
| 0 | warmup 0 at 0 | 32 | 2844.69 | 13 | 159 | 13 | 1 | 0 | 0 | 72 | 64 | 2 |
| 19 | prime 0 at 4096 | 1 | 0.0 | 13 | 0 | 0 | 0 | 72 | 0 | 9 | 9 | 9 |
| 21 | decode 1 at 4096 | 256 | 8965.94 | 88 | 901 | 12 | 3 | 15 | 0 | 31 | 30 | 30 |
| 100 | decode 2 at 4096 | 256 | 8635.74 | 163 | 903 | 10 | 9 | 0 | 0 | 27 | 0 | 0 |
| 179 | prime 0 at 6144 | 1 | 0.0 | 163 | 0 | 0 | 1 | 72 | 0 | 10 | 10 | 10 |
| 182 | decode 1 at 6144 | 256 | 8461.49 | 230 | 805 | 13 | 2 | 13 | 0 | 31 | 31 | 31 |
| 253 | decode 2 at 6144 | 256 | 8018.89 | 296 | 794 | 10 | 9 | 0 | 0 | 27 | 0 | 0 |
| 323 | prime 0 at 8192 | 1 | 0.0 | 296 | 0 | 0 | 0 | 72 | 0 | 9 | 9 | 9 |
| 326 | decode 1 at 8192 | 256 | 8255.25 | 364 | 816 | 12 | 2 | 13 | 0 | 21 | 21 | 21 |
| 397 | decode 2 at 8192 | 256 | 7948.18 | 431 | 806 | 9 | 9 | 0 | 0 | 18 | 0 | 0 |
