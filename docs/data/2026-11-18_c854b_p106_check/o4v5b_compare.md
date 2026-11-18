12 scored attempts, arms o4v5b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4v5b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.0, 8, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.9, 8, 8 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.7, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.6, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4v5b | 12 / 12 | 28.1 | 2.0 | 7.23, 11.72 | 100 | 16.4 | 60 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4v5b: graph reallocations 0 of 60 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4v5b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.49, 789 |
| 2 | 12, 10.77, 1452 |
| 3 | 12, 4.77, 240 |
| 4 | 12, 3.44, 134 |
| 5 | 6, 1.41, 36 |
| 6 | 2, 1.91, 56 |
| 7 | 2, 1.44, 32 |
| 8 | 2, 1.41, 36 |
