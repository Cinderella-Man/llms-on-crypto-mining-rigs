9 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 7.5, 11, 11 / 2 / 0 |
| bugfix-javascript-space-age | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 6.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.3, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 3.1, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 8.2, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.1, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 9.9, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 9 / 9 | 43.8 | 3.1 | 11.70, 20.64 | 52 | 6.6 | 45 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 17.24, 803 |
| 2 | 9, 18.70, 1448 |
| 3 | 9, 3.86, 30 |
| 4 | 9, 8.15, 220 |
| 5 | 3, 4.04, 36 |
| 6 | 1, 3.94, 36 |
| 7 | 1, 4.63, 44 |
| 8 | 1, 3.85, 32 |
| 9 | 1, 5.42, 54 |
| 10 | 1, 5.29, 74 |
| 11 | 1, 4.11, 35 |
