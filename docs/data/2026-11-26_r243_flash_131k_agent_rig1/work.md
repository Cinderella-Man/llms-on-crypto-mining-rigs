### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| ca | 12 | 73 | 41589 (3466) | 24497 (2041) | 63 (1; 3.3 of 279.2) | 27.45 (bugfix-python-affine-cipher, 1) |
| l131a | 12 | 64 | 38082 (3174) | 24305 (2025) | 0 (0; 0.0 of 248.1) | 19.54 (bugfix-python-affine-cipher, 1) |
| cb | 12 | 64 | 38531 (3211) | 25305 (2109) | 0 (0; 0.0 of 255.4) | 28.18 (bugfix-python-affine-cipher, 1) |
| l131b | 12 | 86 | 49821 (4152) | 34440 (2870) | 0 (0; 0.0 of 335.3) | 19.57 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| ca | bugfix-python-affine-cipher | 6 | reply re-read | 8621 | 63 | 8617 | 63 | 412 | 3.33 | 3.53 |
