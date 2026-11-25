### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| o5q8a | 12 | 60 | 35671 (2973) | 25385 (2115) | 0 (0; 0.0 of 341.8) | 25.65 (bugfix-python-affine-cipher, 1) |
| o5kdpa | 12 | 68 | 36743 (3062) | 20995 (1750) | 0 (0; 0.0 of 341.7) | 25.72 (bugfix-python-affine-cipher, 1) |
| o5q8b | 12 | 64 | 36722 (3060) | 23411 (1951) | 0 (0; 0.0 of 354.6) | 26.47 (bugfix-python-affine-cipher, 1) |
| o5kdpb | 12 | 66 | 35062 (2922) | 20467 (1706) | 0 (0; 0.0 of 328.1) | 25.74 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

None.
