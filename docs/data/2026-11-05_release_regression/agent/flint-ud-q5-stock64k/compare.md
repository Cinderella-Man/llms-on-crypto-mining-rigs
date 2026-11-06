12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 10.4, 8, 8 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 3.0, 6, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 6.7, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.8, 8, 7 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 2 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 3.0, 6, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 10.9, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 7.7, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 11 / 12 | 62.7 | 3.0 | 4.25, 8.36 | 141 | 13.7 | 62 / 0 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 6.67, 694 |
| 2 | 12, 7.87, 1256 |
| 3 | 12, 4.45, 383 |
| 4 | 11, 3.94, 283 |
| 5 | 10, 1.16, 20 |
| 6 | 7, 2.08, 95 |
| 7 | 2, 1.12, 18 |
| 8 | 2, 3.33, 200 |
