12 scored attempts, arms o5q8b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5q8b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.4, 10, 10 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 4.7, 10, 10 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 4.2, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.3, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5q8b | 12 / 12 | 26.6 | 1.8 | 5.71, 9.93 | 99 | 20.5 | 67 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5q8b: graph reallocations 0 of 67 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5q8b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.12, 789 |
| 2 | 12, 9.85, 1452 |
| 3 | 12, 3.95, 197 |
| 4 | 12, 1.54, 42 |
| 5 | 8, 1.33, 36 |
| 6 | 3, 3.25, 121 |
| 7 | 2, 1.49, 38 |
| 8 | 2, 1.36, 34 |
| 9 | 2, 1.84, 52 |
| 10 | 2, 2.20, 69 |
