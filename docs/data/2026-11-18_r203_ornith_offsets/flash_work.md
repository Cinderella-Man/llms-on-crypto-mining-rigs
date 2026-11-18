### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| fnoc0 | 12 | 60 | 36978 (3082) | 22860 (1905) | 0 (0; 0.0 of 261.9) | 28.91 (bugfix-python-affine-cipher, 1) |
| r156a-v4r | 12 | 95 | 52794 (4400) | 41836 (3486) | 4863 (2; 31.0 of 394.2) | — |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| r156a-v4r | bugfix-javascript-complex-numbers | 26 | reply re-read | 22892 | 1077 | 22888 | 1077 | 1775 | 13.59 | — |
| r156a-v4r | bugfix-python-pig-latin | 3 | reply re-read | 3426 | 3786 | 3422 | 3786 | 4012 | 17.44 | — |
