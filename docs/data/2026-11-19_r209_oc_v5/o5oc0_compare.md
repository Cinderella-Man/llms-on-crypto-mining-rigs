12 scored attempts, arms o5oc0; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5oc0: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.8, 8, 8 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.3, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.5, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.4, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5oc0 | 12 / 12 | 20.0 | 1.4 | 6.78, 11.30 | 102 | 17.9 | 60 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5oc0: graph reallocations 0 of 60 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5oc0: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.43, 789 |
| 2 | 12, 10.17, 1452 |
| 3 | 12, 5.40, 259 |
| 4 | 12, 1.52, 40 |
| 5 | 7, 2.31, 75 |
| 6 | 3, 1.40, 36 |
| 7 | 1, 3.10, 129 |
| 8 | 1, 1.38, 35 |
