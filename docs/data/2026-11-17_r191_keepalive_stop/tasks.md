# Graph decisions and keep-alive counts per request (<OUT>/server-k1.log)

| task | request | gen tokens | gen ms | graphs reused (llama) | debug: reused | warm-up done (captured) | warm-up reset (kernel by kernel) | update failed | disabled | kernel by kernel | at new shapes | block stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (load) |  |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |
| 0 | warmup 0 at 0 | 32 | 4485.55 | 13 | 159 | 13 | 1 | 0 | 0 | 72 | 0 | 0 |
| 19 | prime 0 at 4096 | 1 | 0.0 | 13 | 0 | 0 | 0 | 72 | 0 | 9 | 0 | 0 |
| 21 | decode 1 at 4096 | 256 | 9685.7 | 88 | 901 | 12 | 3 | 15 | 0 | 31 | 0 | 0 |
| 100 | decode 2 at 4096 | 256 | 8632.35 | 163 | 903 | 10 | 9 | 0 | 0 | 27 | 0 | 0 |
