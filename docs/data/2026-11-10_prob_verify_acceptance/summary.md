# Probabilistic verification and kept draws: the counters (66 requests in 12 sessions, 6 loads)

## Per load

| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step (pooled) | Tokens a step (mean of requests) |
|---|---|---:|---|---:|---:|---:|---:|
| pv0 | pv0-1 | 17,410 | 11,688 of 16,958 | 0.6892 | 5,722 | 3.043 | 3.127 |
| pv0 | pv0-2 | 17,359 | 11,708 of 16,743 | 0.6993 | 5,651 | 3.072 | 3.143 |
| kd1 | kd1-1 | 16,762 | 11,270 of 16,266 | 0.6929 | 5,492 | 3.052 | 3.116 |
| kd1 | kd1-2 | 17,140 | 11,542 of 16,576 | 0.6963 | 5,598 | 3.062 | 3.134 |
| pv1 | pv1-1 | 18,788 | 13,066 of 17,050 | 0.7663 | 5,722 | 3.283 | 3.344 |
| pv1 | pv1-2 | 18,784 | 13,099 of 16,944 | 0.7731 | 5,685 | 3.304 | 3.369 |
| recorded | the recorded session | 27,067 | 18,447 of 25,704 | 0.7177 | 8,620 | 3.140 | 3.191 |

## Each arm against `pv0`

| Metric | A | B | A, each load | B, each load | A − B | % | Spread of A | Spread of B | Class (B's spread) | Class (the larger spread) | 95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| tokens a step | kd1 | pv0 | 3.052, 3.062 | 3.043, 3.072 | -0.000 | -0.01 | 0.010 | 0.029 | negligible | negligible | -0.029 to +0.036 | -0.029 to +0.035 | -0.010 | -0.040 to +0.019 | 17 / 16 / 33 |
| share accepted | kd1 | pv0 | 0.6929, 0.6963 | 0.6892, 0.6993 | +0.0003 | +0.05 | 0.0035 | 0.0100 | negligible | negligible | -0.0096 to +0.0132 | -0.0096 to +0.0128 | -0.0029 | -0.0133 to +0.0072 | 18 / 16 / 32 |
| tokens a step | pv1 | pv0 | 3.283, 3.304 | 3.043, 3.072 | +0.237 | +7.74 | 0.021 | 0.029 | gain | gain | +0.164 to +0.292 | +0.160 to +0.288 | +0.222 | +0.176 to +0.267 | 58 / 6 / 2 |
| share accepted | pv1 | pv0 | 0.7663, 0.7731 | 0.6892, 0.6993 | +0.0755 | +10.87 | 0.0067 | 0.0100 | gain | gain | +0.0512 to +0.0936 | +0.0502 to +0.0920 | +0.0594 | +0.0455 to +0.0732 | 58 / 7 / 1 |

## One arm, its second load against its first

| Metric | A | B | A, each load | B, each load | A − B | % | 95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |
|---|---|---|---|---|---:|---:|---|---|---:|---|---|
| tokens a step | pv0-2 | pv0-1 | 3.072 | 3.043 | +0.029 | +0.96 | -0.043 to +0.103 | -0.046 to +0.112 | +0.016 | -0.027 to +0.062 | 13 / 15 / 38 |
| share accepted | pv0-2 | pv0-1 | 0.6993 | 0.6892 | +0.0100 | +1.46 | -0.0143 to +0.0349 | -0.0150 to +0.0378 | +0.0064 | -0.0080 to +0.0220 | 14 / 15 / 37 |
| tokens a step | kd1-2 | kd1-1 | 3.062 | 3.052 | +0.010 | +0.32 | -0.056 to +0.089 | -0.063 to +0.096 | +0.018 | -0.027 to +0.064 | 17 / 14 / 35 |
| share accepted | kd1-2 | kd1-1 | 0.6963 | 0.6929 | +0.0035 | +0.50 | -0.0189 to +0.0302 | -0.0216 to +0.0326 | +0.0070 | -0.0082 to +0.0226 | 18 / 14 / 34 |
| tokens a step | pv1-2 | pv1-1 | 3.304 | 3.283 | +0.021 | +0.63 | -0.015 to +0.062 | -0.030 to +0.065 | +0.025 | -0.028 to +0.079 | 18 / 12 / 36 |
| share accepted | pv1-2 | pv1-1 | 0.7731 | 0.7663 | +0.0067 | +0.88 | -0.0054 to +0.0211 | -0.0103 to +0.0223 | +0.0060 | -0.0110 to +0.0237 | 18 / 12 / 36 |

## The same reply in two loads

| Loads | Kind | Same reply | Same counters | Below 8,192 prompt tokens | From 8,192 |
|---|---|---|---|---|---|
| pv0-1, pv0-2 | one arm, two loads | 37 of 66 | 36 of 66 | 31 of 60 | 6 of 6 |
| pv0-1, kd1-1 | arm against the control | 46 of 66 | 45 of 66 | 40 of 60 | 6 of 6 |
| pv0-1, kd1-2 | arm against the control | 35 of 66 | 34 of 66 | 31 of 60 | 4 of 6 |
| pv0-1, pv1-1 | arm against the control | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| pv0-1, pv1-2 | arm against the control | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| pv0-2, kd1-1 | arm against the control | 38 of 66 | 39 of 66 | 32 of 60 | 6 of 6 |
| pv0-2, kd1-2 | arm against the control | 36 of 66 | 33 of 66 | 32 of 60 | 4 of 6 |
| pv0-2, pv1-1 | arm against the control | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| pv0-2, pv1-2 | arm against the control | 2 of 66 | 2 of 66 | 2 of 60 | 0 of 6 |
| kd1-1, kd1-2 | one arm, two loads | 35 of 66 | 34 of 66 | 31 of 60 | 4 of 6 |
| kd1-1, pv1-1 | two arms | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| kd1-1, pv1-2 | two arms | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| kd1-2, pv1-1 | two arms | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| kd1-2, pv1-2 | two arms | 2 of 66 | 1 of 66 | 2 of 60 | 0 of 6 |
| pv1-1, pv1-2 | one arm, two loads | 35 of 66 | 36 of 66 | 33 of 60 | 2 of 6 |

## The same reply, by session

| Loads | Sessions the same in every request | Sessions that differ from their first request | Sessions that differ from a later request | Requests the same again after a session's first differing one |
|---|---:|---:|---:|---:|
| pv0-1, pv0-2 | 7 of 12 | 4 | 1 | 0 of 24 |
| pv0-1, kd1-1 | 8 of 12 | 3 | 1 | 0 of 16 |
| pv0-1, kd1-2 | 6 of 12 | 3 | 3 | 1 of 26 |
| pv0-1, pv1-1 | 0 of 12 | 10 | 2 | 0 of 52 |
| pv0-1, pv1-2 | 0 of 12 | 10 | 2 | 0 of 52 |
| pv0-2, kd1-1 | 7 of 12 | 3 | 2 | 0 of 23 |
| pv0-2, kd1-2 | 6 of 12 | 1 | 5 | 0 of 24 |
| pv0-2, pv1-1 | 0 of 12 | 10 | 2 | 0 of 52 |
| pv0-2, pv1-2 | 0 of 12 | 10 | 2 | 0 of 52 |
| kd1-1, kd1-2 | 6 of 12 | 2 | 4 | 0 of 25 |
| kd1-1, pv1-1 | 0 of 12 | 10 | 2 | 0 of 52 |
| kd1-1, pv1-2 | 0 of 12 | 10 | 2 | 0 of 52 |
| kd1-2, pv1-1 | 0 of 12 | 10 | 2 | 0 of 52 |
| kd1-2, pv1-2 | 0 of 12 | 10 | 2 | 0 of 52 |
| pv1-1, pv1-2 | 6 of 12 | 3 | 3 | 1 of 26 |

## The server logs

| Load | Switches on its line | Generated | Requests with a grammar | Their tokens: sampled, drawn on the device | Rejected (after a device draw) | Verify steps | Restores inside a step | Accepted of drafted: step lines, request lines, timings | Checkpoint restores at a prompt | Draws kept | Error lines |
|---|---|---:|---:|---|---|---:|---:|---|---:|---|---:|
| pv0-1 | LLAMA_TRACE=1 | 17,410 | 66 | 17,420, 17,420 | 1 (1) | 5,661 | 0 | 11,688 of 16,958; 11,688 of 16,958; 11,688 of 16,958 | 65 | off | 0 |
| pv0-2 | LLAMA_TRACE=1 | 17,359 | 66 | 17,368, 17,368 | 1 (1) | 5,590 | 0 | 11,708 of 16,743; 11,708 of 16,743; 11,708 of 16,743 | 65 | off | 0 |
| kd1-1 | LLAMA_SPEC_KEEP_DRAWS=1 LLAMA_TRACE=1 | 16,762 | 66 | 16,772, 16,772 | 1 (1) | 5,431 | 0 | 11,270 of 16,266; 11,270 of 16,266; 11,270 of 16,266 | 65 | on | 0 |
| kd1-2 | LLAMA_SPEC_KEEP_DRAWS=1 LLAMA_TRACE=1 | 17,140 | 66 | 17,151, 17,151 | 1 (1) | 5,536 | 0 | 11,542 of 16,576; 11,542 of 16,576; 11,542 of 16,576 | 65 | on | 0 |
| pv1-1 | --spec-draft-sampling probabilistic --spec-draft-p-min 0 LLAMA_TRACE=1 | 18,788 | 66 | 18,833, 18,833 | 14 (14) | 5,691 | 0 | 13,066 of 17,050; 13,066 of 17,050; 13,066 of 17,050 | 64 | off | 0 |
| pv1-2 | --spec-draft-sampling probabilistic --spec-draft-p-min 0 LLAMA_TRACE=1 | 18,784 | 66 | 18,828, 18,828 | 14 (14) | 5,655 | 0 | 13,099 of 16,944; 13,099 of 16,944; 13,099 of 16,944 | 64 | off | 0 |
