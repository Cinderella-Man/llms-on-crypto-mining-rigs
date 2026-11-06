9 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.1, 7, 7 / 2 / 0 |
| bugfix-javascript-space-age | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.7, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.3, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 10.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.3, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 8 / 9 | 39.5 | 2.9 | 12.61, 38.44 | 39 | 13.0 | 44 / 2 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 20.63, 803 |
| 2 | 9, 32.38, 1448 |
| 3 | 9, 12.34, 408 |
| 4 | 9, 3.77, 44 |
| 5 | 6, 3.46, 35 |
| 6 | 1, 3.64, 43 |
| 7 | 1, 3.33, 35 |
