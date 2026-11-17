# Graph decisions and keep-alive counts per request (<OUT>/server-g1.log)

| task | request | gen tokens | gen ms | graphs reused (llama) | debug: reused | warm-up done (captured) | warm-up reset (kernel by kernel) | update failed | disabled | kernel by kernel | at new shapes | block stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (load) |  |  |  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |
| 0 | warmup 0 at 0 | 32 | 2827.71 | 13 | 159 | 13 | 1 | 0 | 0 | 72 | 64 | 2 |
| 19 | prime 0 at 4096 | 1 | 0.0 | 13 | 0 | 0 | 0 | 72 | 0 | 9 | 9 | 9 |
| 21 | decode 1 at 4096 | 256 | 8947.48 | 88 | 901 | 12 | 3 | 15 | 0 | 31 | 30 | 30 |
| 100 | decode 2 at 4096 | 256 | 8656.51 | 163 | 902 | 9 | 9 | 0 | 0 | 28 | 0 | 0 |
