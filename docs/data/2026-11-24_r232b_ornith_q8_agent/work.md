### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o5t1a | 12 | 64 | 36956 (3080) | 17458 (1455) | 0 (0; 0.0 of 360.4) | 26.70 (bugfix-python-affine-cipher, 1) |
| o5t1b | 12 | 55 | 34628 (2886) | 21478 (1790) | 0 (0; 0.0 of 326.8) | 26.89 (bugfix-python-affine-cipher, 1) |
| o5f16a | 12 | 59 | 35247 (2937) | 18433 (1536) | 0 (0; 0.0 of 337.8) | 26.28 (bugfix-python-affine-cipher, 1) |
| o5q8a | 12 | 65 | 36712 (3059) | 22559 (1880) | 0 (0; 0.0 of 365.5) | 26.48 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

None.
