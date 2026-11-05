8 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | runaway (time), 12.0, 4, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.1, 9, 8 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.2, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.1, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.1, 7, 6 / 1 / 0 |
| bugfix-python-list-ops | pass, 2.0, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.9, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 7 / 8 | 27.9 | 2.1 | 3.69, 7.80 | 140 | 14.6 | 39 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 8, 5.53, 612 |
| 2 | 8, 7.67, 1291 |
| 3 | 8, 3.93, 400 |
| 4 | 8, 1.24, 20 |
| 5 | 6, 2.20, 128 |
| 6 | 2, 2.07, 112 |
| 7 | 2, 2.42, 161 |
| 8 | 1, 1.16, 19 |
| 9 | 1, 2.57, 132 |
