### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| f4ta | 12 | 73 | 48907 (4076) | 37846 (3154) | 14632 (4; 67.6 of 222.6) | 24.70 (bugfix-python-affine-cipher, 3, a re-read) |
| f4tb | 12 | 71 | 61112 (5093) | 48656 (4055) | 24755 (5; 141.7 of 299.5) | 93.88 (bugfix-python-scale-generator, 4, a re-read) |
| f4tr | 12 | 77 | 35635 (2970) | 65384 (5449) | 0 (0; 0.0 of 169.7) | 11.68 (bugfix-javascript-complex-numbers, 2) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| f4ta | bugfix-javascript-grade-school | 4 | reply re-read | 4140 | 314 | 4263 | 191 | 205 | 1.71 | 1.87 |
| f4ta | bugfix-python-affine-cipher | 3 | reply re-read | 3524 | 5120 | 3520 | 5120 | 5530 | 24.30 | 24.70 |
| f4ta | bugfix-python-list-ops | 5 | reply re-read | 3417 | 3995 | 3413 | 3995 | 4012 | 17.67 | 17.85 |
| f4ta | bugfix-python-pig-latin | 3 | reply re-read | 2868 | 5326 | 2864 | 5326 | 5519 | 23.93 | 24.15 |
| f4tb | bugfix-javascript-space-age | 5 | reply re-read | 3817 | 972 | 4585 | 204 | 323 | 1.80 | 1.99 |
| f4tb | bugfix-python-pig-latin | 3 | reply re-read | 2872 | 557 | 2868 | 557 | 1386 | 6.10 | 6.28 |
| f4tb | bugfix-python-pig-latin | 4 | reply re-read | 4254 | 2901 | 4250 | 2901 | 2922 | 12.49 | 12.67 |
| f4tb | bugfix-python-scale-generator | 3 | reply re-read | 3693 | 5614 | 3689 | 5614 | 6187 | 27.92 | 28.21 |
| f4tb | bugfix-python-scale-generator | 4 | reply re-read | 9876 | 15479 | 9872 | 15479 | 15689 | 93.41 | 93.88 |
