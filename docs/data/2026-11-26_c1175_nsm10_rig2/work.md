### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| ca | 12 | 85 | 51659 (4305) | 50096 (4175) | 0 (0; 0.0 of 319.7) | 26.77 (bugfix-python-affine-cipher, 1) |
| r05a | 12 | 63 | 36478 (3040) | 19756 (1646) | 0 (0; 0.0 of 229.6) | 27.07 (bugfix-python-affine-cipher, 1) |
| cb | 12 | 71 | 36980 (3082) | 33242 (2770) | 0 (0; 0.0 of 239.3) | 27.23 (bugfix-python-affine-cipher, 1) |
| r05b | 12 | 73 | 43241 (3603) | 40493 (3374) | 0 (0; 0.0 of 272.7) | 27.38 (bugfix-python-affine-cipher, 1) |

### The turns that read a reply again

None.
