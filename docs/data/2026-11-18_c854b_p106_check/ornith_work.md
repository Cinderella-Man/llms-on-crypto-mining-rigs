### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o4c2 | 12 | 63 | 39103 (3259) | 20266 (1689) | 2143 (1; 13.2 of 406.6) | 28.84 (bugfix-python-affine-cipher, 1) |
| o4sna | 12 | 63 | 40863 (3405) | 21810 (1818) | 4831 (2; 30.6 of 396.6) | 28.14 (bugfix-python-affine-cipher, 1) |
| o4snb | 12 | 73 | 49555 (4130) | 27605 (2300) | 12332 (3; 71.9 of 467.8) | 32.24 (bugfix-python-scale-generator, 7, a re-read) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| o4c2 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 2143 | 3427 | 2143 | 2377 | 13.19 | 13.40 |
| o4sna | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4853 | 2551 | 4849 | 2551 | 2586 | 17.81 | 17.88 |
| o4sna | bugfix-python-pig-latin | 4 | reply re-read | 3607 | 2280 | 3603 | 2280 | 2314 | 12.79 | 12.86 |
| o4snb | bugfix-python-list-ops | 3 | reply re-read | 3431 | 1800 | 3427 | 1800 | 2034 | 11.93 | 12.03 |
| o4snb | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 4253 | 3537 | 4253 | 4286 | 27.86 | 27.95 |
| o4snb | bugfix-python-scale-generator | 7 | reply re-read | 6827 | 6279 | 6823 | 6279 | 6310 | 32.10 | 32.24 |
