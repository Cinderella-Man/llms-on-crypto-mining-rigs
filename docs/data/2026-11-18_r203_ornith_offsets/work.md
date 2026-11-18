### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o4sna | 12 | 63 | 40863 (3405) | 21810 (1818) | 4831 (2; 30.6 of 396.6) | 28.14 (bugfix-python-affine-cipher, 1) |
| o4snb | 12 | 73 | 49555 (4130) | 27605 (2300) | 12332 (3; 71.9 of 467.8) | 32.24 (bugfix-python-scale-generator, 7, a re-read) |
| o4oc0 | 12 | 67 | 44697 (3725) | 23764 (1980) | 8860 (5; 65.6 of 437.3) | 28.07 (bugfix-python-affine-cipher, 1) |
| o4oc1 | 12 | 62 | 39552 (3296) | 17016 (1418) | 4074 (1; 28.8 of 389.2) | 28.94 (bugfix-python-scale-generator, 4, a re-read) |
| o4oc2 | 12 | 63 | 43369 (3614) | 21519 (1793) | 8130 (5; 59.1 of 394.5) | 26.67 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| o4sna | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4853 | 2551 | 4849 | 2551 | 2586 | 17.81 | 17.88 |
| o4sna | bugfix-python-pig-latin | 4 | reply re-read | 3607 | 2280 | 3603 | 2280 | 2314 | 12.79 | 12.86 |
| o4snb | bugfix-python-list-ops | 3 | reply re-read | 3431 | 1800 | 3427 | 1800 | 2034 | 11.93 | 12.03 |
| o4snb | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 4253 | 3537 | 4253 | 4286 | 27.86 | 27.95 |
| o4snb | bugfix-python-scale-generator | 7 | reply re-read | 6827 | 6279 | 6823 | 6279 | 6310 | 32.10 | 32.24 |
| o4oc0 | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 5034 | 1994 | 5030 | 1994 | 2037 | 12.47 | 12.56 |
| o4oc0 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 3034 | 3427 | 3034 | 3067 | 17.52 | 17.61 |
| o4oc0 | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 392 | 3537 | 392 | 418 | 7.41 | 7.46 |
| o4oc0 | bugfix-python-scale-generator | 4 | reply re-read | 5435 | 815 | 5431 | 815 | 870 | 9.43 | 9.50 |
| o4oc0 | bugfix-python-scale-generator | 7 | reply re-read | 7033 | 2625 | 7029 | 2625 | 2657 | 18.82 | 18.92 |
| o4oc1 | bugfix-python-scale-generator | 4 | reply re-read | 5427 | 4074 | 5423 | 4074 | 4108 | 28.84 | 28.94 |
| o4oc2 | bugfix-python-list-ops | 3 | reply re-read | 3431 | 604 | 3427 | 604 | 636 | 8.02 | 8.08 |
| o4oc2 | bugfix-python-phone-number | 4 | reply re-read | 5804 | 663 | 5800 | 663 | 693 | 8.91 | 9.02 |
| o4oc2 | bugfix-python-pig-latin | 4 | reply re-read | 4078 | 1725 | 4074 | 1725 | 1760 | 10.89 | 11.02 |
| o4oc2 | bugfix-python-scale-generator | 4 | reply re-read | 5337 | 2818 | 5333 | 2818 | 2969 | 17.37 | 17.46 |
| o4oc2 | bugfix-python-scale-generator | 7 | reply re-read | 9191 | 2320 | 9187 | 2320 | 2351 | 13.91 | 14.04 |
