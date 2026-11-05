Reference: `v2-rc4`

| prompt | mode | build | prompt tokens | template = ref | tokens | = ref | first difference | ref margin there | max abs d logprob (shared prefix) |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| py-duration | pipe | v3-b12076-ladder5-1551f5fe1 | 58 | yes | 256 | NO | 123 | 0.0112 | 0.2117 |
| py-duration | probs | v3-b12076-ladder5-1551f5fe1 | 58 | yes | 256 | NO | 123 | 0.0112 | 0.2117 |
| js-fib | pipe | v3-b12076-ladder5-1551f5fe1 | 96 | yes | 256 | NO | 60 | 0.0239 | 0.2072 |
| js-fib | probs | v3-b12076-ladder5-1551f5fe1 | 96 | yes | 256 | NO | 60 | 0.0239 | 0.2072 |
| sh-review | pipe | v3-b12076-ladder5-1551f5fe1 | 3406 | yes | 256 | NO | 99 | 0.0555 | 0.1477 |
| sh-review | probs | v3-b12076-ladder5-1551f5fe1 | 3406 | yes | 256 | NO | 99 | 0.0555 | 0.1477 |
| prose | pipe | v3-b12076-ladder5-1551f5fe1 | 45 | yes | 256 | NO | 133 | 0.3433 | 0.4381 |
| prose | probs | v3-b12076-ladder5-1551f5fe1 | 45 | yes | 256 | NO | 133 | 0.3433 | 0.4381 |

| build | prompt | pipe tokens = probs tokens | first difference |
|---|---|---|---:|
| v2-rc4 | py-duration | yes |  |
| v2-rc4 | js-fib | yes |  |
| v2-rc4 | sh-review | yes |  |
| v2-rc4 | prose | yes |  |
| v3-b12076-ladder5-1551f5fe1 | py-duration | yes |  |
| v3-b12076-ladder5-1551f5fe1 | js-fib | yes |  |
| v3-b12076-ladder5-1551f5fe1 | sh-review | yes |  |
| v3-b12076-ladder5-1551f5fe1 | prose | yes |  |

verdict: DIFFERS (8 requests, 1 builds against v2-rc4)
