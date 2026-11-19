### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o5oc1 | 12 | 67 | 37657 (3138) | 20222 (1685) | 2078 (3; 23.4 of 370.6) | 26.88 (bugfix-python-affine-cipher, 1) |
| o5t0a | 12 | 58 | 40888 (3407) | 20554 (1713) | 5321 (2; 29.9 of 370.4) | 26.47 (bugfix-python-affine-cipher, 1) |
| o5t1a | 12 | 64 | 36956 (3080) | 17458 (1455) | 0 (0; 0.0 of 360.4) | 26.70 (bugfix-python-affine-cipher, 1) |
| o5t0b | 12 | 66 | 36100 (3008) | 16209 (1351) | 0 (0; 0.0 of 357.9) | 26.96 (bugfix-python-affine-cipher, 1) |
| o5t1b | 12 | 55 | 34628 (2886) | 21478 (1790) | 0 (0; 0.0 of 326.8) | 26.89 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| o5oc1 | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4844 | 1306 | 4840 | 1306 | 1341 | 9.97 | 10.04 |
| o5oc1 | bugfix-python-pig-latin | 4 | reply re-read | 3544 | 572 | 3540 | 572 | 599 | 8.27 | 8.32 |
| o5oc1 | bugfix-python-pig-latin | 9 | reply re-read | 6089 | 200 | 6085 | 200 | 234 | 5.20 | 5.26 |
| o5t0a | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 1441 | 3537 | 1441 | 1476 | 9.87 | 9.93 |
| o5t0a | bugfix-python-scale-generator | 5 | reply re-read | 6666 | 3880 | 6662 | 3880 | 3913 | 20.07 | 20.17 |
