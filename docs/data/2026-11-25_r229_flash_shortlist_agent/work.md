### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| s2a | 12 | 62 | 38188 (3182) | 25570 (2131) | 0 (0; 0.0 of 234.5) | 27.71 (bugfix-python-affine-cipher, 1) |
| s1a | 12 | 66 | 40255 (3355) | 27587 (2299) | 0 (0; 0.0 of 247.1) | 27.36 (bugfix-python-affine-cipher, 1) |
| s2b | 12 | 68 | 39794 (3316) | 22062 (1838) | 0 (0; 0.0 of 245.8) | 27.56 (bugfix-python-affine-cipher, 1) |
| s1b | 12 | 67 | 37280 (3107) | 38682 (3224) | 0 (0; 0.0 of 238.5) | 27.35 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

None.
