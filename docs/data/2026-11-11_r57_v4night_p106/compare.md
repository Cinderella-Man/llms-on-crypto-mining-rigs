24 scored attempts, arms v3rel, v4night; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v3rel: result, min, turns, tool calls | v4night: result, min, turns, tool calls | time v3rel / v4night |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 3.5, 5, 5 / 0 / 0 | pass, 2.9, 7, 7 / 0 / 0 | 1.18× |
| bugfix-javascript-space-age | pass, 1.4, 5, 4 / 0 / 0 | pass, 1.4, 5, 4 / 0 / 0 | 0.99× |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 | pass, 1.8, 6, 6 / 0 / 0 | 1.00× |
| bugfix-javascript-triangle | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.0, 4, 3 / 0 / 0 | 1.13× |
| bugfix-python-variable-length-quantity | pass, 2.2, 5, 5 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | 1.07× |
| bugfix-javascript-phone-number | pass, 1.3, 5, 4 / 0 / 0 | pass, 4.1, 6, 5 / 0 / 0 | 0.31× |
| bugfix-python-list-ops | pass, 2.1, 5, 4 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | 0.96× |
| bugfix-javascript-grade-school | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | 1.02× |
| bugfix-python-scale-generator | pass, 2.1, 5, 5 / 0 / 0 | pass, 2.5, 5, 5 / 0 / 0 | 0.86× |
| bugfix-javascript-complex-numbers | pass, 2.0, 7, 6 / 0 / 0 | pass, 1.5, 6, 5 / 0 / 0 | 1.29× |
| bugfix-python-pig-latin | pass, 2.4, 6, 5 / 0 / 0 | pass, 2.9, 6, 6 / 0 / 0 | 0.82× |
| bugfix-javascript-variable-length-quantity | runaway (time), 12.0, 2, 2 / 0 / 0 | runaway (time), 12.0, 6, 6 / 0 / 0 | 1.00× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v3rel | 11 / 12 | 33.0 | 2.0 | 7.50, 16.39 | 69 | 21.0 | 53 / 0 / 0 | 1 |
| v4night | 12 / 12 | 35.6 | 2.1 | 6.94, 15.23 | 70 | 20.6 | 61 / 0 / 0 | 1 |

Tasks scored on every arm: 12.
- v4night against v3rel: 35.6 against 33.0 minutes in all (0.93× faster), per task 0.92× (geometric mean, range 0.31–1.29); passed 12 against 11

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v3rel: tasks, TTFT s, new prompt tokens | v4night: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 12, 13.63, 789 | 12, 13.33, 789 |
| 2 | 12, 15.80, 1528 | 12, 15.04, 1608 |
| 3 | 11, 7.23, 382 | 12, 6.94, 348 |
| 4 | 11, 2.54, 30 | 12, 2.53, 31 |
| 5 | 11, 3.09, 53 | 11, 3.69, 91 |
| 6 | 2, 3.36, 88 | 6, 4.05, 122 |
| 7 | 1, 6.79, 329 | 1, 3.92, 115 |
