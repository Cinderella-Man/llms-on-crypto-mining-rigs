48 scored attempts, arms gains, v4c2, v4c22, v4c2g; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | gains: result, min, turns, tool calls | v4c2: result, min, turns, tool calls | v4c22: result, min, turns, tool calls | v4c2g: result, min, turns, tool calls | time gains / v4c2 | time gains / v4c22 | time gains / v4c2g |
|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 4.8, 9, 8 / 1 / 0 | pass, 2.5, 6, 6 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | pass, 2.4, 6, 6 / 0 / 0 | 1.94× | 1.98× | 1.99× |
| bugfix-javascript-space-age | pass, 1.0, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | pass, 0.8, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | 1.33× | 1.24× | 1.16× |
| bugfix-python-phone-number | pass, 2.0, 6, 6 / 0 / 0 | pass, 1.1, 6, 5 / 0 / 0 | pass, 1.3, 5, 5 / 0 / 0 | pass, 1.2, 5, 5 / 0 / 0 | 1.89× | 1.55× | 1.68× |
| bugfix-javascript-triangle | pass, 1.1, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | 1.57× | 1.60× | 1.27× |
| bugfix-python-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.1, 5, 6 / 1 / 0 | pass, 2.4, 5, 5 / 0 / 0 | 1.50× | 2.27× | 1.10× |
| bugfix-javascript-phone-number | pass, 1.2, 5, 4 / 0 / 0 | pass, 0.9, 5, 5 / 0 / 0 | pass, 1.1, 5, 4 / 0 / 0 | pass, 1.8, 7, 7 / 1 / 0 | 1.22× | 1.08× | 0.65× |
| bugfix-python-list-ops | pass, 1.9, 5, 5 / 0 / 0 | pass, 2.3, 5, 5 / 0 / 0 | pass, 1.3, 5, 5 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | 0.81× | 1.46× | 1.40× |
| bugfix-javascript-grade-school | pass, 1.3, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 0.7, 5, 4 / 0 / 0 | pass, 1.4, 6, 6 / 0 / 0 | 1.73× | 1.72× | 0.89× |
| bugfix-python-scale-generator | pass, 3.1, 5, 5 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | pass, 5.4, 6, 6 / 1 / 0 | 1.52× | 1.61× | 0.57× |
| bugfix-javascript-complex-numbers | pass, 2.0, 5, 5 / 0 / 0 | runaway (time), 12.0, 35, 34 / 2 / 0 | pass, 1.0, 5, 5 / 0 / 0 | pass, 1.5, 5, 5 / 0 / 0 | 0.16× | 1.93× | 1.31× |
| bugfix-python-pig-latin | pass, 2.5, 7, 6 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | pass, 2.3, 9, 8 / 1 / 0 | pass, 1.9, 5, 5 / 0 / 0 | 1.75× | 1.08× | 1.30× |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 | pass, 5.0, 5, 5 / 0 / 0 | pass, 5.1, 6, 6 / 0 / 0 | pass, 1.8, 5, 5 / 0 / 0 | 0.52× | 0.52× | 1.42× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| gains | 12 / 12 | 26.1 | 2.0 | 4.23, 24.54 | 73 | 24.8 | 63 / 1 / 0 | 0 |
| v4c2 | 11 / 12 | 31.3 | 1.6 | 2.42, 9.29 | 136 | 26.1 | 87 / 2 / 0 | 1 |
| v4c22 | 12 / 12 | 19.9 | 1.2 | 2.81, 8.58 | 148 | 27.5 | 61 / 2 / 0 | 0 |
| v4c2g | 12 / 12 | 23.0 | 1.6 | 4.42, 10.84 | 106 | 24.6 | 65 / 2 / 0 | 0 |

Tasks scored on every arm: 12.
- v4c2 against gains: 31.3 against 26.1 minutes in all (0.83× faster), per task 1.13× (geometric mean, range 0.16–1.94); passed 11 against 12
- v4c22 against gains: 19.9 against 26.1 minutes in all (1.31× faster), per task 1.42× (geometric mean, range 0.52–2.27); passed 12 against 12
- v4c2g against gains: 23.0 against 26.1 minutes in all (1.13× faster), per task 1.16× (geometric mean, range 0.57–1.99); passed 12 against 12

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- gains: graph reallocations 69 of 69 requests (329 same size, 137 changed): **reallocates** where v4c2, v4c22, v4c2g do not
- v4c2: graph reallocations 0 of 91 requests (0 same size, 0 changed)
- v4c22: graph reallocations 0 of 65 requests (0 same size, 0 changed)
- v4c2g: graph reallocations 0 of 65 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | gains: tasks, TTFT s, new prompt tokens | v4c2: tasks, TTFT s, new prompt tokens | v4c22: tasks, TTFT s, new prompt tokens | v4c2g: tasks, TTFT s, new prompt tokens |
|---|---|---|---|---|
| 1 | 12, 10.75, 789 | 12, 6.65, 809 | 12, 6.67, 789 | 12, 8.03, 789 |
| 2 | 12, 21.02, 1668 | 12, 7.97, 1550 | 12, 8.17, 1608 | 12, 10.14, 1308 |
| 3 | 12, 5.84, 404 | 12, 3.25, 410 | 12, 3.37, 422 | 12, 4.52, 422 |
| 4 | 12, 1.21, 30 | 12, 1.05, 31 | 12, 1.03, 30 | 12, 1.20, 32 |
| 5 | 12, 2.07, 84 | 12, 1.72, 109 | 12, 1.71, 100 | 12, 1.15, 36 |
| 6 | 5, 3.00, 209 | 3, 1.25, 46 | 2, 1.88, 160 | 4, 1.73, 90 |
| 7 | 2, 1.73, 92 | 1, 1.82, 179 | 1, 1.07, 32 | 1, 4.24, 376 |
| 8 | 1, 1.81, 67 | 1, 2.44, 324 | 1, 1.72, 110 | 0, —, — |
| 9 | 1, 5.59, 344 | 1, 0.90, 22 | 1, 1.65, 106 | 0, —, — |
| 10 | 0, —, — | 1, 1.16, 37 | 0, —, — | 0, —, — |
| 11 | 0, —, — | 1, 1.67, 79 | 0, —, — | 0, —, — |
| 12 | 0, —, — | 1, 2.40, 308 | 0, —, — | 0, —, — |
| 13 | 0, —, — | 1, 2.35, 287 | 0, —, — | 0, —, — |
| 14 | 0, —, — | 1, 1.31, 46 | 0, —, — | 0, —, — |
| 15 | 0, —, — | 1, 2.47, 309 | 0, —, — | 0, —, — |
| 16 | 0, —, — | 1, 9.49, 1082 | 0, —, — | 0, —, — |
| 17 | 0, —, — | 1, 1.01, 24 | 0, —, — | 0, —, — |
| 18 | 0, —, — | 1, 3.93, 570 | 0, —, — | 0, —, — |
| 19 | 0, —, — | 1, 1.07, 25 | 0, —, — | 0, —, — |
| 20 | 0, —, — | 1, 9.04, 680 | 0, —, — | 0, —, — |
| 21 | 0, —, — | 1, 1.41, 36 | 0, —, — | 0, —, — |
| 22 | 0, —, — | 1, 2.13, 206 | 0, —, — | 0, —, — |
| 23 | 0, —, — | 1, 2.28, 239 | 0, —, — | 0, —, — |
| 24 | 0, —, — | 1, 0.92, 19 | 0, —, — | 0, —, — |
| 25 | 0, —, — | 1, 1.51, 60 | 0, —, — | 0, —, — |
| 26 | 0, —, — | 1, 1.13, 29 | 0, —, — | 0, —, — |
| 27 | 0, —, — | 1, 3.87, 548 | 0, —, — | 0, —, — |
| 28 | 0, —, — | 1, 9.79, 851 | 0, —, — | 0, —, — |
| 29 | 0, —, — | 1, 0.93, 19 | 0, —, — | 0, —, — |
| 30 | 0, —, — | 1, 1.88, 130 | 0, —, — | 0, —, — |
| 31 | 0, —, — | 1, 11.48, 1258 | 0, —, — | 0, —, — |
| 32 | 0, —, — | 1, 10.17, 727 | 0, —, — | 0, —, — |
| 33 | 0, —, — | 1, 19.98, 2920 | 0, —, — | 0, —, — |
| 34 | 0, —, — | 1, 1.14, 25 | 0, —, — | 0, —, — |
| 35 | 0, —, — | 1, —, — | 0, —, — | 0, —, — |
