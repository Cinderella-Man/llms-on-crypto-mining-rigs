12 scored attempts, arms f5ut; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f5ut: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.6, 8, 8 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 6, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 7, 7 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 7, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.3, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.6, 11, 11 / 2 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.1, 5, 4 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | prompt tokens read (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f5ut | 12 / 12 | 18.0 | 0.9 | 1.64, 5.39 | 213 | 48.6 | 71 / 2 / 0 | 0 | 34502 (2875) | 0 (0; 0.0 of 162.1) | 12.53 (bugfix-javascript-complex-numbers, 2) |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f5ut: graph reallocations 75 of 76 requests (200 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f5ut: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.29, 693 |
| 2 | 12, 4.14, 950 |
| 3 | 12, 2.32, 542 |
| 4 | 12, 0.79, 52 |
| 5 | 11, 0.79, 32 |
| 6 | 7, 0.77, 20 |
| 7 | 5, 0.94, 79 |
| 8 | 2, 1.01, 40 |
| 9 | 1, 1.05, 54 |
| 10 | 1, 0.68, 19 |
| 11 | 1, 0.74, 20 |
