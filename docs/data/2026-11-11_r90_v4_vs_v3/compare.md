24 scored attempts, arms v3, gains; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v3: result, min, turns, tool calls | gains: result, min, turns, tool calls | time v3 / gains |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 4.1, 8, 8 / 0 / 0 | pass, 3.2, 8, 7 / 2 / 0 | 1.28× |
| bugfix-javascript-space-age | pass, 1.4, 6, 6 / 0 / 0 | pass, 0.9, 5, 4 / 0 / 0 | 1.53× |
| bugfix-python-phone-number | pass, 2.4, 6, 6 / 0 / 0 | pass, 1.5, 5, 5 / 0 / 0 | 1.61× |
| bugfix-javascript-triangle | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.4, 6, 5 / 0 / 0 | 0.89× |
| bugfix-python-variable-length-quantity | pass, 1.8, 6, 6 / 0 / 0 | pass, 3.9, 5, 6 / 0 / 0 | 0.47× |
| bugfix-javascript-phone-number | pass, 1.4, 5, 4 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | 1.01× |
| bugfix-python-list-ops | pass, 2.7, 5, 5 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | 1.62× |
| bugfix-javascript-grade-school | pass, 1.4, 6, 6 / 0 / 0 | pass, 2.0, 6, 5 / 0 / 0 | 0.72× |
| bugfix-python-scale-generator | pass, 3.0, 5, 5 / 0 / 0 | pass, 2.7, 6, 6 / 0 / 0 | 1.11× |
| bugfix-javascript-complex-numbers | pass, 2.6, 7, 6 / 0 / 0 | pass, 2.7, 5, 5 / 0 / 0 | 0.95× |
| bugfix-python-pig-latin | pass, 2.9, 5, 4 / 1 / 0 | pass, 2.7, 5, 5 / 0 / 0 | 1.07× |
| bugfix-javascript-variable-length-quantity | runaway (time), 12.0, 2, 2 / 0 / 0 | pass, 4.6, 5, 5 / 0 / 0 | 2.60× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v3 | 11 / 12 | 37.0 | 2.5 | 6.52, 16.17 | 65 | 20.7 | 62 / 1 / 0 | 1 |
| gains | 12 / 12 | 28.7 | 2.3 | 4.55, 24.48 | 74 | 24.9 | 63 / 2 / 0 | 0 |

Tasks scored on every arm: 12.
- gains against v3: 28.7 against 37.0 minutes in all (1.29× faster), per task 1.14× (geometric mean, range 0.47–2.60); passed 12 against 11

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v3: tasks, TTFT s, new prompt tokens | gains: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 12, 13.54, 789 | 12, 10.68, 789 |
| 2 | 12, 15.15, 1212 | 12, 20.58, 1670 |
| 3 | 11, 7.23, 425 | 12, 5.36, 346 |
| 4 | 11, 2.57, 31 | 12, 1.21, 30 |
| 5 | 11, 2.89, 52 | 12, 1.99, 128 |
| 6 | 6, 3.05, 64 | 4, 4.42, 274 |
| 7 | 2, 3.83, 117 | 1, 4.38, 362 |
| 8 | 1, 3.09, 48 | 1, 1.26, 33 |
