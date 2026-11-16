12 scored attempts, arms o4snb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4snb: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.3, 14, 14 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.8, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 10.8, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 5.7, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.3, 11, 11 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4snb | 12 / 12 | 36.8 | 2.3 | 6.06, 11.32 | 106 | 16.1 | 73 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4snb: graph reallocations 0 of 73 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4snb: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.45, 789 |
| 2 | 12, 10.16, 1452 |
| 3 | 12, 5.42, 262 |
| 4 | 12, 4.90, 218 |
| 5 | 7, 1.43, 37 |
| 6 | 3, 2.31, 82 |
| 7 | 3, 1.40, 33 |
| 8 | 3, 1.74, 44 |
| 9 | 2, 3.81, 152 |
| 10 | 2, 1.61, 41 |
| 11 | 2, 1.58, 42 |
| 12 | 1, 1.40, 32 |
| 13 | 1, 1.43, 35 |
| 14 | 1, 1.69, 41 |
