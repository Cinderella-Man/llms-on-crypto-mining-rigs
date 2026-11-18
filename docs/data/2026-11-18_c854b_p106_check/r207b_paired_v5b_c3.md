| task | turn | prompt tokens read | from the cache | o4v5b prompt ms | o4c3 prompt ms | o4c3 / o4v5b |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8599.7 | 8602.4 | 1.0003 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16603.6 | 16450.8 | 0.9908 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6520.1 | 6881.2 | 1.0554 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11130.3 | 11030.9 | 0.9911 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8353.2 | 8010.1 | 0.9589 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10299.2 | 10276.3 | 0.9978 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8586.5 | 8549.5 | 0.9957 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8454.7 | 8409.1 | 0.9946 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7189.2 | 7022.7 | 0.9768 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9944.3 | 9949.7 | 1.0005 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7872.9 | 7855.6 | 0.9978 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11874.0 | 11898.5 | 1.0021 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27834.8 | 27867.1 | 1.0012 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7120.7 | 7402.7 | 1.0396 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10078.6 | 10078.1 | 1.0000 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9563.0 | 9517.7 | 0.9953 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8406.1 | 8398.3 | 0.9991 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8879.5 | 8964.1 | 1.0095 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9347.0 | 9354.0 | 1.0007 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8453.5 | 8562.2 | 1.0129 |

| requests | prompt tokens read | o4v5b s | o4c3 s | o4c3 / o4v5b | o4c3 slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 20 | 23368 | 205.11 | 205.08 | 0.9999 | 9 / 0 / 11 | 0.9995 |
| first turns, 12 | 10708 | 117.85 | 118.02 | 1.0015 | 6 / 0 / 6 | 0.9997 |
| with a cached prefix (all but a session's cold first request), 19 | 20626 | 177.28 | 177.21 | 0.9997 | 8 / 0 / 11 | 0.9991 |
