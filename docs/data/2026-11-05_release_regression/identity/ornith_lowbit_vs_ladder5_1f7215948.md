Reference: `v3-lowbit`

| prompt | mode | build | prompt tokens | template = ref | tokens | = ref | first difference | ref margin there | max abs d logprob (shared prefix) |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| py-duration | pipe | v3-b12076-ladder5 | 68 | yes | 256 | NO | 24 | 0.1885 | 0.3236 |
| py-duration | probs | v3-b12076-ladder5 | 68 | yes | 256 | NO | 24 | 0.1885 | 0.3236 |
| js-fib | pipe | v3-b12076-ladder5 | 109 | yes | 256 | NO | 4 | 0.1601 | 0.0545 |
| js-fib | probs | v3-b12076-ladder5 | 109 | yes | 256 | NO | 4 | 0.1601 | 0.0545 |
| sh-review | pipe | v3-b12076-ladder5 | 3656 | yes | 256 | NO | 70 | 0.8327 | 0.4939 |
| sh-review | probs | v3-b12076-ladder5 | 3656 | yes | 256 | NO | 70 | 0.8327 | 0.4939 |
| prose | pipe | v3-b12076-ladder5 | 58 | yes | 256 | NO | 31 | 0.6293 | 0.1749 |
| prose | probs | v3-b12076-ladder5 | 58 | yes | 256 | NO | 31 | 0.6293 | 0.1749 |

| build | prompt | pipe tokens = probs tokens | first difference |
|---|---|---|---:|
| v3-lowbit | py-duration | yes |  |
| v3-lowbit | js-fib | yes |  |
| v3-lowbit | sh-review | yes |  |
| v3-lowbit | prose | yes |  |
| v3-b12076-ladder5 | py-duration | yes |  |
| v3-b12076-ladder5 | js-fib | yes |  |
| v3-b12076-ladder5 | sh-review | yes |  |
| v3-b12076-ladder5 | prose | yes |  |

verdict: DIFFERS (8 requests, 1 builds against v3-lowbit)
