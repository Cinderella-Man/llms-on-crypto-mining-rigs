29 scored attempts, arms rel, pipe0, hcq8; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls | pipe0: result, min, turns, tool calls | hcq8: result, min, turns, tool calls | time rel / pipe0 | time rel / hcq8 |
|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 5.8, 6, 6 / 0 / 0 | pass, 10.2, 7, 7 / 1 / 0 | pass, 4.6, 5, 4 / 0 / 0 | 0.57× | 1.26× |
| bugfix-javascript-space-age | pass, 2.4, 5, 4 / 0 / 0 | pass, 2.8, 4, 3 / 0 / 0 | pass, 1.8, 5, 4 / 0 / 0 | 0.84× | 1.33× |
| bugfix-python-phone-number | pass, 2.5, 6, 5 / 0 / 0 | pass, 3.5, 6, 6 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | 0.72× | 1.28× |
| bugfix-javascript-triangle | pass, 1.9, 5, 4 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | pass, 1.8, 5, 5 / 0 / 0 | 0.94× | 1.03× |
| bugfix-python-variable-length-quantity | runaway (time), 15.6, 5, 5 / 0 / 0 | pass, 7.0, 5, 5 / 0 / 0 | pass, 5.4, 6, 6 / 0 / 0 | 2.24× | 2.87× |
| bugfix-javascript-phone-number | pass, 2.8, 5, 6 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | pass, 2.3, 5, 4 / 0 / 0 | 1.24× | 1.21× |
| bugfix-python-list-ops | pass, 4.5, 5, 5 / 0 / 0 | runaway (time), 12.0, 3, 2 / 0 / 0 | pass, 6.1, 6, 6 / 0 / 0 | 0.38× | 0.74× |
| bugfix-javascript-grade-school | pass, 3.0, 7, 6 / 0 / 0 | pass, 2.5, 6, 5 / 0 / 0 | pass, 1.5, 5, 4 / 0 / 0 | 1.19× | 1.99× |
| bugfix-python-scale-generator | pass, 5.6, 6, 6 / 0 / 0 | — | pass, 4.6, 5, 5 / 0 / 0 | — | 1.24× |
| bugfix-javascript-complex-numbers | — | — | pass, 2.8, 5, 5 / 0 / 0 | — | — |
| bugfix-python-pig-latin | — | — | pass, 4.9, 5, 5 / 0 / 0 | — | — |
| bugfix-javascript-variable-length-quantity | — | — | pass, 4.8, 5, 5 / 0 / 0 | — | — |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 9 / 9 | 44.2 | 3.0 | 7.65, 19.80 | 61 | 7.8 | 47 / 0 / 0 | 1 |
| pipe0 | 7 / 8 | 42.3 | 3.2 | 7.59, 18.17 | 61 | 7.9 | 36 / 1 / 0 | 1 |
| hcq8 | 12 / 12 | 42.6 | 3.7 | 8.31, 18.19 | 64 | 9.0 | 57 / 0 / 0 | 0 |

Tasks scored on every arm: 8.
- pipe0 against rel: 42.3 against 38.6 minutes in all (0.91× faster), per task 0.89× (geometric mean, range 0.38–2.24); passed 7 against 8
- hcq8 against rel: 25.6 against 38.6 minutes in all (1.51× faster), per task 1.35× (geometric mean, range 0.74–2.87); passed 8 against 8

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens | pipe0: tasks, TTFT s, new prompt tokens | hcq8: tasks, TTFT s, new prompt tokens |
|---|---|---|---|
| 1 | 9, 15.75, 803 | 8, 14.20, 706 | 12, 15.07, 789 |
| 2 | 9, 17.73, 1533 | 8, 17.74, 1494 | 12, 17.41, 1550 |
| 3 | 9, 9.70, 411 | 8, 7.81, 311 | 12, 8.38, 399 |
| 4 | 9, 3.94, 31 | 7, 3.01, 31 | 12, 2.68, 31 |
| 5 | 9, 5.07, 84 | 6, 4.31, 102 | 12, 3.80, 94 |
| 6 | 4, 5.19, 84 | 3, 4.47, 92 | 2, 3.62, 64 |
| 7 | 1, 5.28, 186 | 1, 4.58, 110 | 0, —, — |
