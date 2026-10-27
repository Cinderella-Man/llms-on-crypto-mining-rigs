7 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 7.1, 8, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 2.0, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 5.4, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.9, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 3.8, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | runaway (time), 12.0, 2, 2 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 6 / 7 | 35.5 | 3.8 | 10.16, 20.07 | 52 | 5.4 | 31 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 7, 18.29, 803 |
| 2 | 7, 19.25, 1406 |
| 3 | 6, 3.80, 31 |
| 4 | 6, 5.81, 119 |
| 5 | 2, 6.53, 160 |
| 6 | 1, 3.33, 28 |
| 7 | 1, 3.96, 35 |
| 8 | 1, 9.41, 344 |
