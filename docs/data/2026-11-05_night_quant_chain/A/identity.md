Reference: `lowbit`

| prompt | mode | build | prompt tokens | template = ref | tokens | = ref | first difference | ref margin there | max abs d logprob (shared prefix) |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| py-duration | pipe | ladder5 | 68 | yes | 256 | NO | 74 | 1.1332 | 0.5117 |
| py-duration | probs | ladder5 | 68 | yes | 256 | NO | 74 | 1.1332 | 0.5117 |
| js-fib | pipe | ladder5 | 109 | yes | 256 | NO | 7 | 0.1654 | 0.0459 |
| js-fib | probs | ladder5 | 109 | yes | 256 | NO | 7 | 0.1654 | 0.0459 |
| sh-review | pipe | ladder5 | 3656 | yes | 256 | NO | 7 | 1.1965 | 0.1529 |
| sh-review | probs | ladder5 | 3656 | yes | 256 | NO | 7 | 1.1965 | 0.1529 |
| prose | pipe | ladder5 | 58 | yes | 256 | NO | 25 | 0.6856 | 0.4457 |
| prose | probs | ladder5 | 58 | yes | 256 | NO | 25 | 0.6856 | 0.4457 |

| build | prompt | pipe tokens = probs tokens | first difference |
|---|---|---|---:|
| lowbit | py-duration | yes |  |
| lowbit | js-fib | yes |  |
| lowbit | sh-review | yes |  |
| lowbit | prose | yes |  |
| ladder5 | py-duration | yes |  |
| ladder5 | js-fib | yes |  |
| ladder5 | sh-review | yes |  |
| ladder5 | prose | yes |  |

verdict: DIFFERS (8 requests, 1 builds against lowbit)
