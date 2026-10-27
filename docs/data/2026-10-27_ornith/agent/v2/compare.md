9 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.6, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.2, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.3, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.9, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | runaway (time), 12.0, 2, 2 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.7, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 4, 4 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 7 / 9 | 42.6 | 2.9 | 8.65, 16.82 | 67 | 8.8 | 36 / 0 / 0 | 2 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 14.32, 803 |
| 2 | 9, 15.82, 1452 |
| 3 | 8, 3.26, 32 |
| 4 | 8, 4.82, 228 |
| 5 | 2, 5.95, 227 |
| 6 | 1, 3.09, 36 |
