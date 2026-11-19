12 scored attempts, arms o5t0b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5t0b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 9, 9 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.5, 7, 7 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.8, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.7, 8, 8 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5t0b | 12 / 12 | 20.3 | 1.4 | 5.64, 9.62 | 101 | 19.5 | 66 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5t0b: graph reallocations 0 of 66 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5t0b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.03, 789 |
| 2 | 12, 9.58, 1452 |
| 3 | 12, 6.29, 320 |
| 4 | 12, 1.22, 32 |
| 5 | 8, 1.30, 36 |
| 6 | 4, 2.15, 40 |
| 7 | 3, 1.34, 35 |
| 8 | 2, 2.17, 78 |
| 9 | 1, 1.29, 35 |
