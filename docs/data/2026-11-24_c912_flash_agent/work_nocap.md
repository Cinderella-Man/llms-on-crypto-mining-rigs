### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| t1400a | 11 | 60 | 33138 (3013) | 30159 (2742) | 0 (0; 0.0 of 213.7) | 27.37 (bugfix-python-affine-cipher, 1) |
| t1400b | 11 | 62 | 34961 (3178) | 17712 (1610) | 215 (1; 2.9 of 227.2) | 27.53 (bugfix-python-affine-cipher, 1) |
| f5cand | 11 | 55 | 32446 (2950) | 31065 (2824) | 0 (0; 0.0 of 221.7) | 28.58 (bugfix-python-affine-cipher, 1) |
| f5ref | 11 | 55 | 34236 (3112) | 17151 (1559) | 0 (0; 0.0 of 232.6) | 28.52 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| t1400b | bugfix-python-pig-latin | 7 | reply re-read | 5253 | 215 | 5249 | 215 | 411 | 2.88 | 2.94 |
