12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.3, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.2, 8, 9 / 1 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.3, 9, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.3, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.0, 11, 11 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 21.6 | 1.2 | 2.45, 6.89 | 168 | 44.7 | 73 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.40, 693 |
| 2 | 12, 5.64, 1122 |
| 3 | 12, 3.50, 490 |
| 4 | 12, 1.44, 106 |
| 5 | 11, 0.56, 20 |
| 6 | 6, 0.77, 53 |
| 7 | 3, 1.91, 97 |
| 8 | 3, 2.41, 134 |
| 9 | 2, 1.30, 82 |
| 10 | 1, 0.60, 18 |
| 11 | 1, 1.03, 66 |
