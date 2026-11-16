72 scored attempts, arms o4a, o4sa, o3a, o4b, o4sb, o3b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4a: result, min, turns, tool calls | o4sa: result, min, turns, tool calls | o3a: result, min, turns, tool calls | o4b: result, min, turns, tool calls | o4sb: result, min, turns, tool calls | o3b: result, min, turns, tool calls | time o4a / o4sa | time o4a / o3a | time o4a / o4b | time o4a / o4sb | time o4a / o3b |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 3.3, 7, 7 / 0 / 0 | pass, 3.7, 9, 9 / 0 / 0 | pass, 4.7, 13, 13 / 1 / 0 | pass, 2.5, 6, 6 / 0 / 0 | pass, 3.4, 11, 11 / 1 / 0 | pass, 3.3, 8, 8 / 0 / 0 | 0.91× | 0.71× | 1.31× | 0.98× | 1.00× |
| bugfix-javascript-space-age | pass, 1.1, 4, 4 / 0 / 0 | pass, 0.8, 4, 4 / 0 / 0 | pass, 1.3, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 0.8, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | 1.41× | 0.88× | 1.01× | 1.39× | 1.01× |
| bugfix-python-phone-number | pass, 2.5, 5, 5 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | pass, 2.5, 5, 5 / 0 / 0 | pass, 2.8, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | 1.25× | 1.01× | 0.88× | 1.31× | 1.20× |
| bugfix-javascript-triangle | pass, 1.2, 4, 4 / 0 / 0 | pass, 0.9, 4, 4 / 0 / 0 | pass, 1.2, 4, 4 / 0 / 0 | pass, 1.2, 4, 4 / 0 / 0 | pass, 0.9, 4, 4 / 0 / 0 | pass, 1.3, 4, 4 / 0 / 0 | 1.42× | 1.01× | 1.03× | 1.46× | 0.97× |
| bugfix-python-variable-length-quantity | pass, 2.8, 6, 6 / 0 / 0 | pass, 2.0, 5, 5 / 0 / 0 | pass, 3.2, 7, 7 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | pass, 1.9, 5, 5 / 0 / 0 | 1.37× | 0.86× | 1.23× | 1.17× | 1.49× |
| bugfix-javascript-phone-number | pass, 1.7, 5, 5 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 1.5, 4, 4 / 0 / 0 | pass, 1.5, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 1.6, 4, 4 / 0 / 0 | 1.49× | 1.10× | 1.07× | 1.51× | 1.03× |
| bugfix-python-list-ops | pass, 1.6, 4, 4 / 0 / 0 | pass, 1.9, 4, 4 / 0 / 0 | pass, 2.5, 5, 5 / 0 / 0 | pass, 1.7, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 1.6, 4, 4 / 0 / 0 | 0.84× | 0.63× | 0.94× | 1.46× | 0.98× |
| bugfix-javascript-grade-school | pass, 1.2, 4, 4 / 0 / 0 | pass, 1.3, 6, 7 / 0 / 0 | pass, 2.0, 7, 7 / 1 / 0 | pass, 1.3, 4, 4 / 0 / 0 | pass, 0.9, 4, 4 / 0 / 0 | pass, 1.2, 4, 4 / 0 / 0 | 0.92× | 0.60× | 0.90× | 1.33× | 1.00× |
| bugfix-python-scale-generator | pass, 5.6, 5, 5 / 0 / 0 | pass, 8.6, 7, 7 / 0 / 0 | pass, 9.2, 7, 7 / 0 / 0 | pass, 8.4, 5, 5 / 0 / 0 | pass, 5.0, 5, 5 / 0 / 0 | pass, 3.6, 5, 5 / 0 / 0 | 0.65× | 0.61× | 0.67× | 1.12× | 1.56× |
| bugfix-javascript-complex-numbers | pass, 1.5, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 1.5, 4, 4 / 0 / 0 | pass, 1.5, 4, 4 / 0 / 0 | pass, 1.1, 4, 4 / 0 / 0 | pass, 1.8, 5, 5 / 0 / 0 | 1.33× | 1.01× | 0.98× | 1.34× | 0.83× |
| bugfix-python-pig-latin | pass, 4.5, 7, 7 / 0 / 0 | pass, 2.2, 5, 5 / 0 / 0 | pass, 3.6, 5, 5 / 0 / 0 | pass, 4.1, 7, 7 / 0 / 0 | pass, 4.7, 9, 9 / 0 / 0 | pass, 2.8, 6, 6 / 0 / 0 | 2.07× | 1.25× | 1.09× | 0.95× | 1.62× |
| bugfix-javascript-variable-length-quantity | pass, 2.7, 7, 7 / 0 / 0 | pass, 2.0, 6, 6 / 0 / 0 | pass, 2.6, 6, 6 / 0 / 0 | pass, 2.4, 5, 5 / 0 / 0 | pass, 2.6, 6, 6 / 0 / 0 | pass, 4.4, 6, 6 / 0 / 0 | 1.32× | 1.02× | 1.12× | 1.04× | 0.61× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4a | 12 / 12 | 29.7 | 2.1 | 8.65, 19.22 | 57 | 16.5 | 62 / 0 / 0 | 0 |
| o4sa | 12 / 12 | 27.6 | 1.9 | 3.05, 10.97 | 115 | 16.3 | 64 / 0 / 0 | 0 |
| o3a | 12 / 12 | 35.8 | 2.5 | 7.26, 19.56 | 61 | 16.2 | 71 / 2 / 0 | 0 |
| o4b | 12 / 12 | 31.0 | 2.0 | 9.58, 20.29 | 65 | 16.4 | 57 / 0 / 0 | 0 |
| o4sb | 12 / 12 | 25.9 | 1.5 | 3.27, 11.28 | 114 | 16.5 | 65 / 1 / 0 | 0 |
| o3b | 12 / 12 | 26.6 | 1.8 | 8.80, 19.25 | 58 | 16.5 | 60 / 0 / 0 | 0 |

Tasks scored on every arm: 12.
- o4sa against o4a: 27.6 against 29.7 minutes in all (1.08× faster), per task 1.19× (geometric mean, range 0.65–2.07); passed 12 against 12
- o3a against o4a: 35.8 against 29.7 minutes in all (0.83× faster), per task 0.87× (geometric mean, range 0.60–1.25); passed 12 against 12
- o4b against o4a: 31.0 against 29.7 minutes in all (0.96× faster), per task 1.00× (geometric mean, range 0.67–1.31); passed 12 against 12
- o4sb against o4a: 25.9 against 29.7 minutes in all (1.15× faster), per task 1.24× (geometric mean, range 0.95–1.51); passed 12 against 12
- o3b against o4a: 26.6 against 29.7 minutes in all (1.11× faster), per task 1.07× (geometric mean, range 0.61–1.62); passed 12 against 12

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4a: graph reallocations 0 of 62 requests (0 same size, 0 changed)
- o4sa: graph reallocations 0 of 63 requests (0 same size, 0 changed)
- o3a: graph reallocations 0 of 71 requests (0 same size, 0 changed)
- o4b: graph reallocations 0 of 57 requests (0 same size, 0 changed)
- o4sb: graph reallocations 0 of 65 requests (0 same size, 0 changed)
- o3b: graph reallocations 0 of 60 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4a: tasks, TTFT s, new prompt tokens | o4sa: tasks, TTFT s, new prompt tokens | o3a: tasks, TTFT s, new prompt tokens | o4b: tasks, TTFT s, new prompt tokens | o4sb: tasks, TTFT s, new prompt tokens | o3b: tasks, TTFT s, new prompt tokens |
|---|---|---|---|---|---|---|
| 1 | 12, 16.32, 789 | 12, 8.45, 789 | 12, 16.34, 789 | 12, 16.26, 789 | 12, 8.44, 789 | 12, 16.52, 789 |
| 2 | 12, 18.85, 1452 | 12, 10.19, 1452 | 12, 18.51, 1452 | 12, 18.94, 1452 | 12, 10.15, 1452 | 12, 18.64, 1452 |
| 3 | 12, 7.76, 280 | 12, 2.07, 128 | 12, 7.83, 316 | 12, 5.10, 127 | 12, 2.08, 127 | 12, 7.13, 260 |
| 4 | 12, 3.37, 34 | 12, 1.97, 83 | 12, 5.01, 128 | 12, 7.03, 216 | 12, 2.87, 246 | 12, 4.93, 122 |
| 5 | 7, 3.44, 36 | 7, 1.40, 36 | 8, 3.25, 36 | 6, 3.39, 36 | 6, 1.43, 36 | 7, 3.24, 36 |
| 6 | 4, 4.25, 78 | 4, 2.39, 128 | 5, 3.90, 61 | 2, 3.31, 36 | 3, 2.67, 191 | 3, 4.22, 80 |
| 7 | 3, 3.35, 36 | 2, 1.30, 30 | 4, 4.19, 76 | 1, 6.60, 188 | 2, 1.83, 61 | 1, 3.35, 32 |
| 8 | 0, —, — | 1, 1.40, 32 | 1, 3.24, 32 | 0, —, — | 2, 1.41, 34 | 1, 3.31, 35 |
| 9 | 0, —, — | 1, 1.42, 35 | 1, 3.29, 35 | 0, —, — | 2, 2.05, 110 | 0, —, — |
| 10 | 0, —, — | 0, —, — | 1, 3.34, 32 | 0, —, — | 1, 1.42, 32 | 0, —, — |
| 11 | 0, —, — | 0, —, — | 1, 3.27, 32 | 0, —, — | 1, 1.88, 52 | 0, —, — |
| 12 | 0, —, — | 0, —, — | 1, 3.29, 35 | 0, —, — | 0, —, — | 0, —, — |
| 13 | 0, —, — | 0, —, — | 1, 3.36, 36 | 0, —, — | 0, —, — | 0, —, — |
