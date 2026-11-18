### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| v4rc2 | 12 | 92 | 57624 (4802) | 39678 (3306) | 4854 (7; 51.6 of 413.9) | 28.82 (bugfix-python-affine-cipher, 1) |
| r156a-v4r | 12 | 95 | 52794 (4400) | 41836 (3486) | 4863 (2; 31.0 of 394.2) | — |
| r156b-v4r2 | 12 | 68 | 40576 (3381) | 25747 (2146) | 761 (1; 6.9 of 280.6) | — |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| v4rc2 | bugfix-javascript-complex-numbers | 5 | reply re-read | 7777 | 1436 | 7773 | 1436 | 1773 | 9.77 | 9.88 |
| v4rc2 | bugfix-javascript-complex-numbers | 11 | reply re-read | 12505 | 455 | 12501 | 455 | 474 | 3.33 | 3.44 |
| v4rc2 | bugfix-javascript-complex-numbers | 21 | reply re-read | 22211 | 322 | 22207 | 322 | 1478 | 12.56 | 12.72 |
| v4rc2 | bugfix-javascript-grade-school | 4 | reply re-read | 4456 | 150 | 4452 | 150 | 183 | 1.94 | 2.01 |
| v4rc2 | bugfix-python-affine-cipher | 2 | reply re-read | 2780 | 71 | 2776 | 71 | 1257 | 7.67 | 7.76 |
| v4rc2 | bugfix-python-pig-latin | 3 | reply re-read | 3457 | 1620 | 3453 | 1620 | 1846 | 8.75 | 8.82 |
| v4rc2 | bugfix-python-variable-length-quantity | 3 | reply re-read | 4476 | 800 | 4472 | 800 | 1097 | 7.53 | 7.59 |
| r156a-v4r | bugfix-javascript-complex-numbers | 26 | reply re-read | 22892 | 1077 | 22888 | 1077 | 1775 | 13.59 | — |
| r156a-v4r | bugfix-python-pig-latin | 3 | reply re-read | 3426 | 3786 | 3422 | 3786 | 4012 | 17.44 | — |
| r156b-v4r2 | bugfix-python-pig-latin | 3 | reply re-read | 3428 | 761 | 3424 | 761 | 987 | 6.92 | — |
