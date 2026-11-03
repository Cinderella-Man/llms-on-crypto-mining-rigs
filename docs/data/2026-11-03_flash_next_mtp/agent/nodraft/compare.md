9 scored attempts, arms nodraft; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | nodraft: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 8.2, 8, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 4.1, 6, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 2.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 6.9, 7, 7 / 1 / 0 |
| bugfix-javascript-phone-number | pass, 2.5, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 4.3, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.8, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.7, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | nodraft: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 9, 15.76, 803 |
| 2 | 9, 18.44, 1566 |
| 3 | 9, 10.15, 437 |
| 4 | 9, 3.60, 31 |
| 5 | 9, 4.34, 91 |
| 6 | 3, 4.59, 104 |
| 7 | 2, 9.10, 340 |
| 8 | 1, 3.51, 42 |
