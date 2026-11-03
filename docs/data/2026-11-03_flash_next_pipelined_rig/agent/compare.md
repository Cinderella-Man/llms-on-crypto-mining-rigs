16 scored attempts, arms pipe0, pipe3b; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | pipe0: result, min, turns, tool calls | pipe3b: result, min, turns, tool calls | time pipe0 / pipe3b |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 10.2, 7, 7 / 1 / 0 | pass, 11.5, 5, 5 / 0 / 0 | 0.89× |
| bugfix-javascript-space-age | pass, 2.8, 4, 3 / 0 / 0 | pass, 3.7, 5, 4 / 0 / 0 | 0.76× |
| bugfix-python-phone-number | pass, 3.5, 6, 6 / 0 / 0 | pass, 3.4, 5, 5 / 0 / 0 | 1.04× |
| bugfix-javascript-triangle | pass, 2.0, 5, 4 / 0 / 0 | pass, 2.8, 5, 4 / 0 / 0 | 0.73× |
| bugfix-python-variable-length-quantity | pass, 7.0, 5, 5 / 0 / 0 | pass, 5.6, 5, 5 / 0 / 0 | 1.24× |
| bugfix-javascript-phone-number | pass, 2.3, 5, 4 / 0 / 0 | pass, 3.6, 5, 5 / 0 / 0 | 0.63× |
| bugfix-python-list-ops | runaway (time), 12.0, 3, 2 / 0 / 0 | pass, 8.6, 5, 5 / 0 / 0 | 1.39× |
| bugfix-javascript-grade-school | pass, 2.5, 6, 5 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | 1.29× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| pipe0 | 7 / 8 | 42.3 | 3.2 | 7.59, 18.17 | 61 | 7.9 | 36 / 1 / 0 | 1 |
| pipe3b | 8 / 8 | 41.2 | 3.7 | 8.75, 19.34 | 60 | 4.9 | 37 / 0 / 0 | 0 |

Tasks scored on every arm: 8.
- pipe3b against pipe0: 41.2 against 42.3 minutes in all (1.03× faster), per task 0.96× (geometric mean, range 0.63–1.39); passed 8 against 7

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | pipe0: tasks, TTFT s, new prompt tokens | pipe3b: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 8, 14.20, 706 | 8, 14.49, 706 |
| 2 | 8, 17.74, 1494 | 8, 17.83, 1591 |
| 3 | 8, 7.81, 311 | 8, 8.25, 362 |
| 4 | 7, 3.01, 31 | 8, 3.01, 31 |
| 5 | 6, 4.31, 102 | 8, 4.87, 148 |
| 6 | 3, 4.47, 92 | 0, —, — |
| 7 | 1, 4.58, 110 | 0, —, — |
