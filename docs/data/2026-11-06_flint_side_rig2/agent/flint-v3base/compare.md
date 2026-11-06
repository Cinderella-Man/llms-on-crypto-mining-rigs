12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.5, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 7, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.1, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 6.8, 7, 6 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 9.3, 10, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.3, 5, 4 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.4, 8, 8 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 31.2 | 1.2 | 2.64, 7.71 | 180 | 45.5 | 68 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.24, 693 |
| 2 | 12, 4.66, 1016 |
| 3 | 12, 3.00, 536 |
| 4 | 12, 2.20, 313 |
| 5 | 11, 1.47, 76 |
| 6 | 8, 1.14, 20 |
| 7 | 4, 1.48, 79 |
| 8 | 2, 1.68, 74 |
| 9 | 1, 1.32, 20 |
| 10 | 1, 3.74, 347 |
