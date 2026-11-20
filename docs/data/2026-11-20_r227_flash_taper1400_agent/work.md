### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| r075a | 12 | 63 | 39887 (3324) | 20391 (1699) | 0 (0; 0.0 of 258.0) | 27.48 (bugfix-python-affine-cipher, 1) |
| r075b | 12 | 64 | 39104 (3259) | 27522 (2294) | 0 (0; 0.0 of 259.6) | 27.50 (bugfix-python-affine-cipher, 1) |
| c600a | 12 | 60 | 35504 (2959) | 22271 (1856) | 195 (1; 1.9 of 240.1) | 27.46 (bugfix-python-affine-cipher, 1) |
| t1400a | 12 | 66 | 38598 (3216) | 31360 (2613) | 0 (0; 0.0 of 244.5) | 27.37 (bugfix-python-affine-cipher, 1) |
| c600b | 12 | 68 | 39629 (3302) | 29745 (2479) | 0 (0; 0.0 of 256.9) | 27.28 (bugfix-python-affine-cipher, 1) |
| t1400b | 12 | 67 | 39649 (3304) | 18328 (1527) | 215 (1; 2.9 of 249.6) | 27.53 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| c600a | bugfix-python-list-ops | 4 | reply re-read | 5222 | 195 | 5218 | 195 | 230 | 1.91 | 1.96 |
| t1400b | bugfix-python-pig-latin | 7 | reply re-read | 5253 | 215 | 5249 | 215 | 411 | 2.88 | 2.94 |
