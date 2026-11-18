| task | turn | prompt tokens read | from the cache | o4v5a prompt ms | o4c3 prompt ms | o4c3 / o4v5a |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8597.3 | 8602.4 | 1.0006 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16445.1 | 16450.8 | 1.0003 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6510.1 | 6881.2 | 1.0570 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11027.7 | 11030.9 | 1.0003 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8282.2 | 8010.1 | 0.9671 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10203.8 | 10276.3 | 1.0071 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8630.0 | 8549.5 | 0.9907 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8484.7 | 8409.1 | 0.9911 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7243.7 | 7022.7 | 0.9695 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9952.5 | 9949.7 | 0.9997 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8192.7 | 7855.6 | 0.9589 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11785.6 | 11898.5 | 1.0096 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 29185.8 | 27867.1 | 0.9548 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7105.5 | 7402.7 | 1.0418 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10076.5 | 10078.1 | 1.0002 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9539.2 | 9517.7 | 0.9978 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8407.8 | 8398.3 | 0.9989 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8862.6 | 8964.1 | 1.0115 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9356.7 | 9354.0 | 0.9997 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8540.9 | 8562.2 | 1.0025 |

| requests | prompt tokens read | o4v5a s | o4c3 s | o4c3 / o4v5a | o4c3 slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 20 | 23368 | 206.43 | 205.08 | 0.9935 | 10 / 0 / 10 | 0.9999 |
| first turns, 12 | 10708 | 119.59 | 118.02 | 0.9869 | 4 / 0 / 8 | 0.9983 |
| with a cached prefix (all but a session's cold first request), 19 | 20626 | 177.24 | 177.21 | 0.9998 | 10 / 0 / 9 | 1.0002 |
