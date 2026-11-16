### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| v3base | 12 | 75 | 70159 (5847) | 65307 (5442) | 33728 (7; 191.7 of 390.5) | 73.65 (bugfix-python-list-ops, 5, a re-read) |
| reread | 12 | 77 | 35780 (2982) | 55801 (4650) | 0 (0; 0.0 of 200.1) | 12.47 (bugfix-javascript-complex-numbers, 2) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| v3base | bugfix-javascript-complex-numbers | 4 | reply re-read | 6365 | 591 | 6900 | 56 | 73 | 1.14 | 1.33 |
| v3base | bugfix-python-affine-cipher | 4 | reply re-read | 3534 | 2050 | 3530 | 2050 | 2459 | 11.55 | 11.90 |
| v3base | bugfix-python-affine-cipher | 5 | reply re-read | 5989 | 353 | 6280 | 62 | 76 | 1.27 | 1.53 |
| v3base | bugfix-python-list-ops | 5 | reply re-read | 3484 | 14366 | 3480 | 14366 | 14433 | 73.28 | 73.65 |
| v3base | bugfix-python-pig-latin | 4 | reply re-read | 3085 | 2865 | 5866 | 84 | 98 | 1.50 | 1.69 |
| v3base | bugfix-python-scale-generator | 6 | reply re-read | 4846 | 5307 | 4842 | 5307 | 6308 | 30.43 | 30.67 |
| v3base | bugfix-python-scale-generator | 7 | reply re-read | 11150 | 11803 | 11146 | 11803 | 11980 | 72.57 | 73.04 |
