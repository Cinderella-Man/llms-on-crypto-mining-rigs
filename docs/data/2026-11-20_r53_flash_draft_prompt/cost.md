## Each prime (server prompt time; wall time as the client saw it)

| Depth | Load | Arm | Read on cached | Prompt tok/s | ms a prompt token | Wall tok/s | Wall − prompt, ms |
|---:|---|---|---|---:|---:|---:|---:|
| 8,192 | d1 | draft | 8,068 on 60 | 193.5 | 5.168 | 193.4 | 32 |
| 8,192 | d2 | draft | 8,068 on 60 | 193.6 | 5.166 | 193.5 | 11 |
| 8,192 | n1 | nodraft | 8,068 on 60 | 226.5 | 4.414 | 226.5 | 11 |
| 8,192 | n2 | nodraft | 8,068 on 60 | 222.8 | 4.488 | 222.7 | 11 |
| 16,384 | d1 | draft | 16,260 on 60 | 319.5 | 3.130 | 319.1 | 53 |
| 16,384 | d2 | draft | 16,260 on 60 | 309.9 | 3.227 | 309.8 | 19 |
| 16,384 | n1 | nodraft | 16,260 on 60 | 327.8 | 3.051 | 327.7 | 18 |
| 16,384 | n2 | nodraft | 16,260 on 60 | 326.6 | 3.062 | 326.4 | 18 |
| 32,768 | d1 | draft | 16,900 on 15,804 | 225.9 | 4.426 | 225.7 | 74 |
| 32,768 | d2 | draft | 16,900 on 15,804 | 221.1 | 4.523 | 221.0 | 33 |
| 32,768 | n1 | nodraft | 16,900 on 15,804 | 232.6 | 4.298 | 232.5 | 34 |
| 32,768 | n2 | nodraft | 16,900 on 15,804 | 232.5 | 4.301 | 232.4 | 34 |

## The draft's cost per depth (mean of each arm's loads)

| Depth | Read on cached | ms a prompt token, draft | without | the draft adds, ms | Prompt tok/s, draft vs without | Wall ms a prompt token, draft | without | adds, ms | Wall tok/s vs without |
|---:|---|---|---|---:|---:|---|---|---:|---:|
| 8,192 | 8,068 on 60 | 5.168, 5.166 | 4.414, 4.488 | 0.716 | -13.9 % | 5.172, 5.168 | 4.416, 4.490 | 0.717 | -13.9 % |
| 16,384 | 16,260 on 60 | 3.130, 3.227 | 3.051, 3.062 | 0.122 | -3.8 % | 3.133, 3.228 | 3.052, 3.063 | 0.123 | -3.9 % |
| 32,768 | 16,900 on 15,804 | 4.426, 4.523 | 4.298, 4.301 | 0.175 | -3.9 % | 4.431, 4.525 | 4.300, 4.303 | 0.176 | -3.9 % |

## The whole load (every chunk's tokens over their summed time)

| Load | Arm | Tokens | Prompt s | Prompt tok/s | Wall s | Wall tok/s |
|---|---|---:|---:|---:|---:|---:|
| d1 | draft | 41,228 | 167.39 | 246.3 | 167.55 | 246.1 |
| d2 | draft | 41,228 | 170.60 | 241.7 | 170.66 | 241.6 |
| n1 | nodraft | 41,228 | 157.86 | 261.2 | 157.93 | 261.1 |
| n2 | nodraft | 41,228 | 158.68 | 259.8 | 158.75 | 259.7 |

The draft arm against the arm without it over the whole load: -6.3 % on the server's prompt time, -6.4 % on wall time.
