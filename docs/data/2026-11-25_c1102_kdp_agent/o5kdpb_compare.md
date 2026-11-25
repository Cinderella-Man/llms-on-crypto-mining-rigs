12 scored attempts, arms o5kdpb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5kdpb: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.7, 12, 12 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.0, 9, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.5, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5kdpb | 12 / 12 | 22.7 | 1.5 | 4.40, 9.15 | 107 | 20.4 | 66 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5kdpb: graph reallocations 0 of 66 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5kdpb: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.63, 789 |
| 2 | 12, 9.11, 1452 |
| 3 | 12, 1.28, 32 |
| 4 | 12, 2.89, 130 |
| 5 | 6, 1.31, 36 |
| 6 | 3, 2.35, 85 |
| 7 | 2, 1.93, 61 |
| 8 | 2, 1.48, 40 |
| 9 | 2, 1.52, 42 |
| 10 | 1, 1.23, 32 |
| 11 | 1, 1.67, 52 |
| 12 | 1, 1.50, 41 |
