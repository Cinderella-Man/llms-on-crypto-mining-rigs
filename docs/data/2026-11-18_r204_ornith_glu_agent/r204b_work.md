### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o4g2b | 12 | 59 | 39859 (3322) | 20411 (1701) | 4592 (2; 31.4 of 387.5) | 28.78 (bugfix-python-affine-cipher, 1) |
| o4g2a | 12 | 62 | 38823 (3235) | 16516 (1376) | 3406 (3; 31.2 of 394.3) | 29.87 (bugfix-python-affine-cipher, 1) |
| o4c2 | 12 | 63 | 39103 (3259) | 20266 (1689) | 2143 (1; 13.2 of 406.6) | 28.84 (bugfix-python-affine-cipher, 1) |
| o4c3 | 12 | 72 | 37750 (3146) | 21608 (1801) | 1371 (2; 17.4 of 410.2) | 28.88 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| o4g2b | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4900 | 1908 | 4896 | 1908 | 2053 | 12.65 | 12.74 |
| o4g2b | bugfix-python-scale-generator | 4 | reply re-read | 6787 | 2684 | 6783 | 2684 | 2714 | 18.77 | 18.85 |
| o4g2a | bugfix-python-pig-latin | 4 | reply re-read | 3538 | 565 | 3534 | 565 | 591 | 8.69 | 8.74 |
| o4g2a | bugfix-python-pig-latin | 5 | reply re-read | 4125 | 1859 | 4121 | 1859 | 1894 | 11.98 | 12.05 |
| o4g2a | bugfix-python-scale-generator | 4 | reply re-read | 5195 | 982 | 5191 | 982 | 1187 | 10.53 | 10.61 |
| o4c2 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 2143 | 3427 | 2143 | 2377 | 13.19 | 13.40 |
| o4c3 | bugfix-python-pig-latin | 4 | reply re-read | 3538 | 996 | 3534 | 996 | 1030 | 9.60 | 9.65 |
| o4c3 | bugfix-python-variable-length-quantity | 6 | reply re-read | 5806 | 375 | 5802 | 375 | 413 | 7.79 | 7.87 |
