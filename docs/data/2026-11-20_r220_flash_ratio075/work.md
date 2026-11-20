### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| f5oc1 | 12 | 61 | 37739 (3145) | 22262 (1855) | 941 (1; 6.9 of 247.4) | 27.53 (bugfix-python-affine-cipher, 1) |
| c10a | 12 | 64 | 37117 (3093) | 16406 (1367) | 0 (0; 0.0 of 246.9) | 27.21 (bugfix-python-affine-cipher, 1) |
| r075a | 12 | 63 | 39887 (3324) | 20391 (1699) | 0 (0; 0.0 of 258.0) | 27.48 (bugfix-python-affine-cipher, 1) |
| c10b | 12 | 99 | 56042 (4670) | 37544 (3129) | 0 (0; 0.0 of 393.6) | 27.42 (bugfix-python-affine-cipher, 1) |
| r075b | 12 | 64 | 39104 (3259) | 27522 (2294) | 0 (0; 0.0 of 259.6) | 27.50 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| f5oc1 | bugfix-python-pig-latin | 3 | reply re-read | 3457 | 941 | 3453 | 941 | 1167 | 6.93 | 7.01 |
