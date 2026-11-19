### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| v4rc2 | 12 | 92 | 57624 (4802) | 39678 (3306) | 4854 (7; 51.6 of 413.9) | 28.82 (bugfix-python-affine-cipher, 1) |
| f5c1 | 12 | 104 | 57114 (4760) | 40857 (3405) | 1806 (4; 23.6 of 427.2) | 27.97 (bugfix-python-affine-cipher, 1) |
| f5s1 | 12 | 82 | 42492 (3541) | 25651 (2138) | 2071 (3; 19.8 of 309.9) | 27.99 (bugfix-python-affine-cipher, 1) |
| f5t1 | 12 | 103 | 52245 (4354) | 32381 (2698) | 0 (0; 0.0 of 404.0) | 28.19 (bugfix-python-affine-cipher, 1) |

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
| f5c1 | bugfix-javascript-complex-numbers | 34 | reply re-read | 30181 | 200 | 30177 | 200 | 639 | 11.52 | 11.69 |
| f5c1 | bugfix-python-pig-latin | 3 | reply re-read | 3457 | 1291 | 3453 | 1291 | 1514 | 7.97 | 8.02 |
| f5c1 | bugfix-python-pig-latin | 7 | reply re-read | 6032 | 128 | 6028 | 128 | 220 | 2.04 | 2.10 |
| f5c1 | bugfix-python-pig-latin | 8 | reply re-read | 6248 | 187 | 6244 | 187 | 228 | 2.09 | 2.16 |
| f5s1 | bugfix-javascript-variable-length-quantity | 2 | reply re-read | 2126 | 69 | 2122 | 69 | 473 | 3.32 | 3.36 |
| f5s1 | bugfix-python-pig-latin | 4 | reply re-read | 4710 | 552 | 4706 | 552 | 606 | 6.51 | 6.58 |
| f5s1 | bugfix-python-pig-latin | 21 | reply re-read | 12142 | 1450 | 12138 | 1450 | 1500 | 9.92 | 10.02 |
