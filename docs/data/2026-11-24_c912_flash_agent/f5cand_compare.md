12 scored attempts, arms f5cand; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f5cand: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 5.2, 5, 5 / 1 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 7, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.8, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f5cand | 12 / 12 | 23.2 | 1.2 | 2.51, 8.25 | 145 | 28.6 | 58 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f5cand: graph reallocations 0 of 62 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f5cand: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 5.11, 789 |
| 2 | 12, 8.20, 1574 |
| 3 | 12, 2.40, 318 |
| 4 | 12, 1.08, 31 |
| 5 | 11, 1.74, 91 |
| 6 | 2, 1.92, 192 |
| 7 | 1, 1.79, 130 |
