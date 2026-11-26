### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| gmixa | 12 | 67 | 34099 (2842) | 19217 (1601) | 0 (0; 0.0 of 338.2) | 31.10 (bugfix-python-affine-cipher, 1) |
| pmixa | 12 | 65 | 36699 (3058) | 21476 (1790) | 1380 (1; 8.4 of 251.9) | 27.23 (bugfix-python-affine-cipher, 1) |
| gmixb | 12 | 73 | 39068 (3256) | 24765 (2064) | 0 (0; 0.0 of 371.1) | 31.39 (bugfix-python-affine-cipher, 1) |
| pmixb | 12 | 60 | 33052 (2754) | 18459 (1538) | 0 (0; 0.0 of 227.5) | 27.23 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| pmixa | bugfix-python-variable-length-quantity | 5 | reply re-read | 6511 | 1380 | 6507 | 1380 | 1421 | 8.40 | 8.47 |
