| task | turn | prompt tokens read | from the cache | o4sna prompt ms | o4snb prompt ms | o4snb / o4sna |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8537.5 | 8570.8 | 1.0039 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16318.3 | 16403.0 | 1.0052 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6498.8 | 6474.8 | 0.9963 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 10866.7 | 10872.6 | 1.0005 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8233.6 | 8343.7 | 1.0134 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10088.3 | 10066.4 | 0.9978 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8436.5 | 8453.2 | 1.0020 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8318.4 | 8311.6 | 0.9992 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7105.1 | 6754.4 | 0.9506 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9824.7 | 9836.2 | 1.0012 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8106.5 | 8141.8 | 1.0043 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11630.0 | 11601.2 | 0.9975 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 26726.8 | 26506.6 | 0.9918 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 9713.0 | 9722.7 | 1.0010 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7042.9 | 6849.2 | 0.9725 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9960.7 | 9945.2 | 0.9984 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9281.8 | 9276.3 | 0.9994 |
| bugfix-python-phone-number | 2 | 1906 | 2702 | 11357.8 | 11365.9 | 1.0007 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8335.5 | 8360.2 | 1.0030 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8796.9 | 8809.6 | 1.0014 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9143.7 | 9164.1 | 1.0022 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8385.6 | 8393.8 | 1.0010 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11223.9 | 11241.0 | 1.0015 |

| requests | prompt tokens read | o4sna s | o4snb s | o4snb / o4sna | o4snb slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 23 | 28345 | 233.93 | 233.46 | 0.9980 | 14 / 0 / 9 | 1.0010 |
| first turns, 12 | 10708 | 115.83 | 115.29 | 0.9953 | 7 / 0 / 5 | 1.0015 |
| with a cached prefix (all but a session's cold first request), 22 | 25603 | 207.21 | 206.96 | 0.9988 | 14 / 0 / 8 | 1.0010 |
