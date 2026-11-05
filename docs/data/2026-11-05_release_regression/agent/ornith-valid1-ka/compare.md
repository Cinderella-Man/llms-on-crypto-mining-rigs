12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.0, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.8, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.4, 10, 10 / 1 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.1, 7, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.7, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 32.0 | 2.1 | 8.39, 19.38 | 58 | 16.3 | 63 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.61, 789 |
| 2 | 12, 19.02, 1452 |
| 3 | 12, 3.17, 32 |
| 4 | 12, 5.08, 122 |
| 5 | 6, 3.22, 32 |
| 6 | 4, 6.91, 238 |
| 7 | 2, 3.43, 38 |
| 8 | 1, 9.44, 363 |
| 9 | 1, 3.16, 28 |
| 10 | 1, 3.36, 36 |
