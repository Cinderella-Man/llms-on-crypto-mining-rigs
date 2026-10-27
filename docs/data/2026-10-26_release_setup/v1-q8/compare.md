12 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.0, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.5, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.6, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 0.7, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.6, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.8, 6, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.8, 7, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 11 / 12 | 30.3 | 1.6 | 3.96, 7.67 | 146 | 25.7 | 65 / 0 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 6.56, 694 |
| 2 | 12, 7.28, 1268 |
| 3 | 12, 3.90, 434 |
| 4 | 12, 2.96, 294 |
| 5 | 11, 1.17, 20 |
| 6 | 7, 1.07, 20 |
| 7 | 2, 3.22, 99 |
