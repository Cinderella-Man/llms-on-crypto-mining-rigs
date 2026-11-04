12 scored attempts, arms s35stack; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | s35stack: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.8, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.9, 5, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.6, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 7.5, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.7, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.2, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 4.1, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.4, 6, 6 / 1 / 0 |
| bugfix-python-pig-latin | pass, 4.7, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.5, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| s35stack | 12 / 12 | 34.8 | 2.2 | 7.42, 17.34 | 68 | 18.8 | 62 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | s35stack: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 13.73, 789 |
| 2 | 12, 15.61, 1578 |
| 3 | 12, 7.12, 319 |
| 4 | 12, 2.71, 30 |
| 5 | 12, 3.43, 90 |
| 6 | 4, 4.89, 180 |
