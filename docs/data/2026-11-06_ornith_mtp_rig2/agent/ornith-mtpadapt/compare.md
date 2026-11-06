11 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 5.2, 8, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.7, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 9.1, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.4, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.1, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 11 / 11 | 38.0 | 3.5 | 11.43, 31.43 | 40 | 12.8 | 57 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 11, 19.73, 803 |
| 2 | 11, 31.43, 1447 |
| 3 | 11, 6.77, 223 |
| 4 | 11, 3.48, 31 |
| 5 | 6, 4.21, 70 |
| 6 | 3, 3.42, 35 |
| 7 | 2, 3.45, 33 |
| 8 | 2, 7.32, 201 |
