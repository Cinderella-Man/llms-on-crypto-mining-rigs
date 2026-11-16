### The work of each run

Prompt tokens read: the prompt tokens the server processed (the rest came from its cache). Reply tokens re-read: on a turn whose cached prefix stops short of the previous reply's end, the tokens of that reply past the cached prefix, which are read again as prompt. The longest first-token wait is as the client sees it.

| arm | tasks | turns | prompt tokens read (per task) | tokens generated (per task) | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|
| f3a | 12 | 66 | 37335 (3111) | 28677 (2390) | 3868 (6; 26.0 of 208.3) | 13.04 (bugfix-javascript-complex-numbers, 2) |
| f3b | 12 | 68 | 49185 (4099) | 41976 (3498) | 16470 (6; 86.3 of 271.2) | 55.47 (bugfix-python-scale-generator, 4, a re-read) |
| f4a | 12 | 73 | 72086 (6007) | 68241 (5687) | 36767 (9; 211.7 of 407.5) | 88.86 (bugfix-python-scale-generator, 6, a re-read) |
| f4b | 12 | 80 | 66718 (5560) | 64569 (5381) | 31643 (5; 167.9 of 373.0) | 78.05 (bugfix-python-scale-generator, 8, a re-read) |
| f4ta | 12 | 73 | 48907 (4076) | 37846 (3154) | 14632 (4; 67.6 of 222.6) | 24.70 (bugfix-python-affine-cipher, 3, a re-read) |
| f4tb | 12 | 71 | 61112 (5093) | 48656 (4055) | 24755 (5; 141.7 of 299.5) | 93.88 (bugfix-python-scale-generator, 4, a re-read) |
| o4sna | 12 | 63 | 40863 (3405) | 21810 (1818) | 4831 (2; 30.6 of 396.6) | 28.14 (bugfix-python-affine-cipher, 1) |
| o4snb | 12 | 73 | 49555 (4130) | 27605 (2300) | 12332 (3; 71.9 of 467.8) | 32.24 (bugfix-python-scale-generator, 7, a re-read) |

### The turns that read a reply again

| arm | task | turn | cache path | previous prompt's end | previous reply's tokens | cached | reply tokens re-read | prompt tokens read | prompt s | first token s |
|---|---|---|---|---|---|---|---|---|---|---|
| f3a | bugfix-javascript-triangle | 3 | reply re-read | 2904 | 194 | 3007 | 91 | 104 | 1.24 | 1.38 |
| f3a | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 6176 | 776 | 6790 | 162 | 178 | 1.85 | 2.03 |
| f3a | bugfix-python-affine-cipher | 4 | reply re-read | 4480 | 885 | 5175 | 190 | 406 | 2.65 | 2.94 |
| f3a | bugfix-python-phone-number | 5 | reply re-read | 4597 | 1767 | 4593 | 1767 | 1788 | 8.80 | 8.99 |
| f3a | bugfix-python-pig-latin | 3 | reply re-read | 2869 | 1752 | 4486 | 135 | 324 | 1.62 | 1.79 |
| f3a | bugfix-python-pig-latin | 4 | reply re-read | 4810 | 1523 | 4806 | 1523 | 1971 | 9.88 | 10.08 |
| f3b | bugfix-javascript-complex-numbers | 4 | reply re-read | 5701 | 198 | 5789 | 110 | 127 | 1.45 | 1.64 |
| f3b | bugfix-javascript-grade-school | 5 | reply re-read | 4090 | 211 | 4205 | 96 | 110 | 1.31 | 1.48 |
| f3b | bugfix-javascript-variable-length-quantity | 5 | reply re-read | 8373 | 260 | 8579 | 54 | 71 | 1.15 | 1.34 |
| f3b | bugfix-python-affine-cipher | 3 | reply re-read | 3524 | 3349 | 3520 | 3349 | 3755 | 17.27 | 17.62 |
| f3b | bugfix-python-pig-latin | 6 | reply re-read | 6035 | 2109 | 6031 | 2109 | 2126 | 10.07 | 10.26 |
| f3b | bugfix-python-scale-generator | 4 | reply re-read | 5265 | 10752 | 5261 | 10752 | 10871 | 55.09 | 55.47 |
| f4a | bugfix-javascript-complex-numbers | 4 | reply re-read | 6215 | 405 | 6566 | 54 | 68 | 1.07 | 1.24 |
| f4a | bugfix-javascript-grade-school | 4 | reply re-read | 4126 | 331 | 4315 | 142 | 159 | 1.67 | 1.84 |
| f4a | bugfix-python-list-ops | 4 | reply re-read | 3406 | 7807 | 3402 | 7807 | 7831 | 36.91 | 37.13 |
| f4a | bugfix-python-pig-latin | 4 | reply re-read | 2873 | 2577 | 2869 | 2577 | 2770 | 12.19 | 12.38 |
| f4a | bugfix-python-pig-latin | 6 | reply re-read | 6289 | 526 | 6569 | 246 | 367 | 1.74 | 1.97 |
| f4a | bugfix-python-pig-latin | 9 | reply re-read | 7808 | 4967 | 7804 | 4967 | 4992 | 25.91 | 26.20 |
| f4a | bugfix-python-pig-latin | 10 | reply re-read | 12796 | 403 | 13110 | 89 | 106 | 1.51 | 1.76 |
| f4a | bugfix-python-scale-generator | 5 | reply re-read | 7584 | 7938 | 7580 | 7938 | 8093 | 42.32 | 42.71 |
| f4a | bugfix-python-scale-generator | 6 | reply re-read | 15673 | 12947 | 15669 | 12947 | 12966 | 88.40 | 88.86 |
| f4b | bugfix-javascript-triangle | 3 | reply re-read | 1968 | 128 | 2007 | 89 | 102 | 1.20 | 1.33 |
| f4b | bugfix-javascript-variable-length-quantity | 3 | reply re-read | 3639 | 11223 | 3635 | 11223 | 11575 | 56.90 | 57.15 |
| f4b | bugfix-python-affine-cipher | 3 | reply re-read | 3544 | 1155 | 4522 | 177 | 581 | 2.84 | 3.19 |
| f4b | bugfix-python-pig-latin | 6 | reply re-read | 3762 | 6283 | 3758 | 6283 | 6306 | 29.22 | 29.47 |
| f4b | bugfix-python-scale-generator | 8 | reply re-read | 7831 | 13871 | 7827 | 13871 | 13893 | 77.75 | 78.05 |
| f4ta | bugfix-javascript-grade-school | 4 | reply re-read | 4140 | 314 | 4263 | 191 | 205 | 1.71 | 1.87 |
| f4ta | bugfix-python-affine-cipher | 3 | reply re-read | 3524 | 5120 | 3520 | 5120 | 5530 | 24.30 | 24.70 |
| f4ta | bugfix-python-list-ops | 5 | reply re-read | 3417 | 3995 | 3413 | 3995 | 4012 | 17.67 | 17.85 |
| f4ta | bugfix-python-pig-latin | 3 | reply re-read | 2868 | 5326 | 2864 | 5326 | 5519 | 23.93 | 24.15 |
| f4tb | bugfix-javascript-space-age | 5 | reply re-read | 3817 | 972 | 4585 | 204 | 323 | 1.80 | 1.99 |
| f4tb | bugfix-python-pig-latin | 3 | reply re-read | 2872 | 557 | 2868 | 557 | 1386 | 6.10 | 6.28 |
| f4tb | bugfix-python-pig-latin | 4 | reply re-read | 4254 | 2901 | 4250 | 2901 | 2922 | 12.49 | 12.67 |
| f4tb | bugfix-python-scale-generator | 3 | reply re-read | 3693 | 5614 | 3689 | 5614 | 6187 | 27.92 | 28.21 |
| f4tb | bugfix-python-scale-generator | 4 | reply re-read | 9876 | 15479 | 9872 | 15479 | 15689 | 93.41 | 93.88 |
| o4sna | bugfix-javascript-variable-length-quantity | 4 | reply re-read | 4853 | 2551 | 4849 | 2551 | 2586 | 17.81 | 17.88 |
| o4sna | bugfix-python-pig-latin | 4 | reply re-read | 3607 | 2280 | 3603 | 2280 | 2314 | 12.79 | 12.86 |
| o4snb | bugfix-python-list-ops | 3 | reply re-read | 3431 | 1800 | 3427 | 1800 | 2034 | 11.93 | 12.03 |
| o4snb | bugfix-python-pig-latin | 4 | reply re-read | 3541 | 4253 | 3537 | 4253 | 4286 | 27.86 | 27.95 |
| o4snb | bugfix-python-scale-generator | 7 | reply re-read | 6827 | 6279 | 6823 | 6279 | 6310 | 32.10 | 32.24 |
