12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.3, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 7, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.8, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.1, 6, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.3, 8, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.1, 8, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 22.7 | 1.2 | 2.30, 6.52 | 182 | 45.0 | 66 / 0 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.21, 693 |
| 2 | 12, 2.38, 408 |
| 3 | 12, 3.20, 544 |
| 4 | 12, 2.17, 376 |
| 5 | 11, 1.12, 20 |
| 6 | 9, 1.23, 20 |
| 7 | 3, 1.64, 21 |
| 8 | 2, 1.51, 61 |
