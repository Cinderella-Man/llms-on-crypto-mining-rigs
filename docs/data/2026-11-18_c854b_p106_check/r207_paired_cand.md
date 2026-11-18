| task | turn | prompt tokens read | from the cache | o4c2 prompt ms | o4c3 prompt ms | o4c3 / o4c2 |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8615.8 | 8602.4 | 0.9985 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16483.1 | 16450.8 | 0.9980 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6815.3 | 6881.2 | 1.0097 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11054.9 | 11030.9 | 0.9978 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8281.7 | 8010.1 | 0.9672 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10291.1 | 10276.3 | 0.9986 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8672.6 | 8549.5 | 0.9858 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8379.0 | 8409.1 | 1.0036 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7945.9 | 7022.7 | 0.8838 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9956.1 | 9949.7 | 0.9994 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8222.8 | 7855.6 | 0.9554 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11872.5 | 11898.5 | 1.0022 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27818.2 | 27867.1 | 1.0018 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7032.7 | 7402.7 | 1.0526 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10088.2 | 10078.1 | 0.9990 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9550.5 | 9517.7 | 0.9966 |
| bugfix-python-phone-number | 2 | 1906 | 2703 | 11634.2 | 11596.6 | 0.9968 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8419.3 | 8398.3 | 0.9975 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8947.3 | 8964.1 | 1.0019 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9301.9 | 9354.0 | 1.0056 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8525.2 | 8562.2 | 1.0043 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11506.5 | 11504.8 | 0.9998 |

| requests | prompt tokens read | o4c2 s | o4c3 s | o4c3 / o4c2 | o4c3 slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 22 | 27160 | 229.41 | 228.18 | 0.9946 | 8 / 0 / 14 | 0.9988 |
| first turns, 12 | 10708 | 119.20 | 118.02 | 0.9901 | 5 / 0 / 7 | 0.9980 |
| with a cached prefix (all but a session's cold first request), 21 | 24418 | 201.60 | 200.32 | 0.9936 | 7 / 0 / 14 | 0.9986 |
