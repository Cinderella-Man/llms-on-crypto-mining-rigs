12 scored attempts, arms o5oc1; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5oc1: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.0, 7, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.8, 9, 9 / 1 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.5, 10, 10 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.3, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5oc1 | 12 / 12 | 24.1 | 1.3 | 5.37, 10.04 | 102 | 19.3 | 67 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5oc1: graph reallocations 0 of 67 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5oc1: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.94, 789 |
| 2 | 12, 9.58, 1452 |
| 3 | 12, 6.38, 360 |
| 4 | 12, 1.77, 57 |
| 5 | 8, 1.33, 36 |
| 6 | 3, 1.19, 32 |
| 7 | 3, 1.25, 35 |
| 8 | 2, 1.26, 32 |
| 9 | 2, 3.30, 135 |
| 10 | 1, 4.58, 211 |
