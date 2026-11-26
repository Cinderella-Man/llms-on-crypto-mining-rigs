12 scored attempts, arms pre; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | pre: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.1, 7, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.9, 6, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways | prompt tokens read (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) | card energy kJ (J a task) | J a generated token | W generating / reading the prompt / idle (all cards) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre | 12 / 12 | 15.2 | 1.1 | 2.42, 8.31 | 154 | 28.2 | 59 / 0 / 0 | 0 | 38127 (3177) | 0 (0; 0.0 of 248.0) | 19.86 (bugfix-python-affine-cipher, 1) | 321.9 (26827) | 17.93 | 335.0 / 393.0 / 309.5 |

Card energy (FN101): all cards' power from the rig's card log, over each arm's session window (its first request to its last request's end, as the proxy logged them); J a task and J a generated token divide that energy; the watts are the cards' sum while generating (first token to end), reading the prompt (start to first token) and idle between turns (the rest of the window).

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- pre: graph reallocations 0 of 65 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | pre: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.99, 789 |
| 2 | 12, 8.31, 1573 |
| 3 | 12, 2.99, 356 |
| 4 | 12, 1.08, 31 |
| 5 | 12, 1.73, 88 |
| 6 | 4, 1.59, 82 |
| 7 | 1, 1.74, 95 |
