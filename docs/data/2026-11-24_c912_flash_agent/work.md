### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| t1400a | 12 | 66 | 38598 (3216) | 31360 (2613) | 0 (0; 0.0 of 244.5) | 27.37 (bugfix-python-affine-cipher, 1) |
| t1400b | 12 | 67 | 39649 (3304) | 18328 (1527) | 215 (1; 2.9 of 249.6) | 27.53 (bugfix-python-affine-cipher, 1) |
| f5cand | 12 | 62 | 35195 (2933) | 32077 (2673) | 0 (0; 0.0 of 243.2) | 28.58 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| t1400b | bugfix-python-pig-latin | 7 | reply re-read | 5253 | 215 | 5249 | 215 | 411 | 2.88 | 2.94 |
