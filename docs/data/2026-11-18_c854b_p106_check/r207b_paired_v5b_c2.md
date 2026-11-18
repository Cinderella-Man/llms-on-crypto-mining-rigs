| task | turn | prompt tokens read | from the cache | o4v5b prompt ms | o4c2 prompt ms | o4c2 / o4v5b |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8599.7 | 8615.8 | 1.0019 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16603.6 | 16483.1 | 0.9927 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6520.1 | 6815.3 | 1.0453 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11130.3 | 11054.9 | 0.9932 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8353.2 | 8281.7 | 0.9914 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10299.2 | 10291.1 | 0.9992 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8586.5 | 8672.6 | 1.0100 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8454.7 | 8379.0 | 0.9910 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7189.2 | 7945.9 | 1.1053 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9944.3 | 9956.1 | 1.0012 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7872.9 | 8222.8 | 1.0444 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11874.0 | 11872.5 | 0.9999 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27834.8 | 27818.2 | 0.9994 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7120.7 | 7032.7 | 0.9876 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10078.6 | 10088.2 | 1.0009 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9563.0 | 9550.5 | 0.9987 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8406.1 | 8419.3 | 1.0016 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8879.5 | 8947.3 | 1.0076 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9347.0 | 9301.9 | 0.9952 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8453.5 | 8525.2 | 1.0085 |

| requests | prompt tokens read | o4v5b s | o4c2 s | o4c2 / o4v5b | o4c2 slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 20 | 23368 | 205.11 | 206.27 | 1.0057 | 10 / 0 / 10 | 1.0004 |
| first turns, 12 | 10708 | 117.85 | 119.20 | 1.0115 | 7 / 0 / 5 | 1.0017 |
| with a cached prefix (all but a session's cold first request), 19 | 20626 | 177.28 | 178.46 | 1.0067 | 10 / 0 / 9 | 1.0009 |
