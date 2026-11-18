| task | turn | prompt tokens read | from the cache | o4v5a prompt ms | o4c2 prompt ms | o4c2 / o4v5a |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8597.3 | 8615.8 | 1.0021 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16445.1 | 16483.1 | 1.0023 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6510.1 | 6815.3 | 1.0469 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11027.7 | 11054.9 | 1.0025 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8282.2 | 8281.7 | 0.9999 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10203.8 | 10291.1 | 1.0086 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8630.0 | 8672.6 | 1.0049 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8484.7 | 8379.0 | 0.9875 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7243.7 | 7945.9 | 1.0969 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9952.5 | 9956.1 | 1.0004 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8192.7 | 8222.8 | 1.0037 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11785.6 | 11872.5 | 1.0074 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 29185.8 | 27818.2 | 0.9531 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 9954.0 | 10097.6 | 1.0144 |
| bugfix-python-affine-cipher | 3 | 448 | 4052 | 7688.8 | 7675.8 | 0.9983 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7105.5 | 7032.7 | 0.9898 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10076.5 | 10088.2 | 1.0012 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9539.2 | 9550.5 | 1.0012 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8407.8 | 8419.3 | 1.0014 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8862.6 | 8947.3 | 1.0096 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9356.7 | 9301.9 | 0.9941 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8540.9 | 8525.2 | 0.9982 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11439.1 | 11506.5 | 1.0059 |

| requests | prompt tokens read | o4v5a s | o4c2 s | o4c2 / o4v5a | o4c2 slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 23 | 26887 | 235.51 | 235.55 | 1.0002 | 16 / 0 / 7 | 1.0021 |
| first turns, 12 | 10708 | 119.59 | 119.20 | 0.9967 | 7 / 0 / 5 | 1.0013 |
| with a cached prefix (all but a session's cold first request), 22 | 24145 | 206.33 | 207.74 | 1.0068 | 16 / 0 / 6 | 1.0022 |
