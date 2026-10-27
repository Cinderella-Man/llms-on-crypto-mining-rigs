9 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.9, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.4, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 4.1, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.5, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.3, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 5.6, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 3 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 8 / 9 | 35.5 | 2.5 | 11.43, 16.52 | 64 | 7.9 | 40 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 13.93, 803 |
| 2 | 9, 14.39, 1448 |
| 3 | 9, 3.47, 31 |
| 4 | 8, 5.10, 134 |
| 5 | 3, 3.37, 36 |
| 6 | 1, 3.48, 28 |
| 7 | 1, 3.37, 35 |
