### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o4oc0 | 12 | 67 | 44697 (3725) | 23764 (1980) | 8860 (5; 65.6 of 437.3) | 28.07 (bugfix-python-affine-cipher, 1) |
| o4g2a | 12 | 62 | 38823 (3235) | 16516 (1376) | 3406 (3; 31.2 of 394.3) | 29.87 (bugfix-python-affine-cipher, 1) |
| o4g2b | 12 | 59 | 39859 (3322) | 20411 (1701) | 4592 (2; 31.4 of 387.5) | 28.78 (bugfix-python-affine-cipher, 1) |
| o5oc0 | 12 | 60 | 38560 (3213) | 14253 (1188) | 3431 (2; 22.8 of 378.4) | 28.11 (bugfix-python-affine-cipher, 1) |
| o5oc1 | 12 | 67 | 37657 (3138) | 20222 (1685) | 2078 (3; 23.4 of 370.6) | 26.88 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| o4oc0 | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 5034 | 1994 | 5030 | 1994 | 2037 | 12.47 | 12.56 |
| o4oc0 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 3034 | 3427 | 3034 | 3067 | 17.52 | 17.61 |
| o4oc0 | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 392 | 3537 | 392 | 418 | 7.41 | 7.46 |
| o4oc0 | bugfix-python-scale-generator | 4 | reply re-read | 5435 | 815 | 5431 | 815 | 870 | 9.43 | 9.50 |
| o4oc0 | bugfix-python-scale-generator | 7 | reply re-read | 7033 | 2625 | 7029 | 2625 | 2657 | 18.82 | 18.92 |
| o4g2a | bugfix-python-pig-latin | 4 | reply re-read | 3538 | 565 | 3534 | 565 | 591 | 8.69 | 8.74 |
| o4g2a | bugfix-python-pig-latin | 5 | reply re-read | 4125 | 1859 | 4121 | 1859 | 1894 | 11.98 | 12.05 |
| o4g2a | bugfix-python-scale-generator | 4 | reply re-read | 5195 | 982 | 5191 | 982 | 1187 | 10.53 | 10.61 |
| o4g2b | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4900 | 1908 | 4896 | 1908 | 2053 | 12.65 | 12.74 |
| o4g2b | bugfix-python-scale-generator | 4 | reply re-read | 6787 | 2684 | 6783 | 2684 | 2714 | 18.77 | 18.85 |
| o5oc0 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 1316 | 3427 | 1316 | 1349 | 10.26 | 10.32 |
| o5oc0 | bugfix-python-pig-latin | 5 | reply re-read | 4090 | 2115 | 4086 | 2115 | 2150 | 12.57 | 12.64 |
| o5oc1 | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4844 | 1306 | 4840 | 1306 | 1341 | 9.97 | 10.04 |
| o5oc1 | bugfix-python-pig-latin | 4 | reply re-read | 3544 | 572 | 3540 | 572 | 599 | 8.27 | 8.32 |
| o5oc1 | bugfix-python-pig-latin | 9 | reply re-read | 6089 | 200 | 6085 | 200 | 234 | 5.20 | 5.26 |
